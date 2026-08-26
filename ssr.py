"""Realized skew stickiness.

    R = (1 / S_T) * d σ_ATMF / d ln F

S_T is today's ATMF implied skew slope dσ / d ln K (scale only).
Sticky-strike ≈ 1, sticky-delta ≈ 0.

Estimated with rolling OLS of Δσ on (S_T · Δln F): F = ES futures, σ = ATM IV path.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

INVESTING_ES_URL = "https://www.investing.com/indices/us-spx-500-futures-historical-data"
_CACHE = Path(__file__).resolve().parent / ".firecrawl" / "es_futures.parquet"
_MD_ROW = re.compile(
    r"\|\s*([A-Z][a-z]{2} \d{1,2}, \d{4})\s*\|\s*([0-9,]+(?:\.\d+)?)"
)


def atmf_skew_slope(
    atm_iv_pct: float,
    skew_25d_pts: float,
    dte: float = 30.0,
    r: float = 0.045,
) -> float:
    """S_T = dσ/d ln K from 25Δ put vs ATM, in vol pts per unit log-strike."""
    sigma = float(atm_iv_pct) / 100.0
    t = max(float(dte), 1.0) / 365.0
    d1 = float(norm.ppf(0.75))  # put delta = Φ(d1) - 1 = -0.25
    k = (r + 0.5 * sigma * sigma) * t - d1 * sigma * np.sqrt(t)
    if abs(k) < 1e-6:
        return float("nan")
    return float(skew_25d_pts) / k


def parse_investing_markdown(text: str) -> pd.DataFrame:
    rows = []
    for date_s, px_s in _MD_ROW.findall(text):
        rows.append(
            {
                "date": pd.to_datetime(date_s),
                "close": float(px_s.replace(",", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["close"])
    out = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def _fetch_es_yfinance(lookback_days: int) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker("ES=F").history(period=f"{max(lookback_days, 30)}d")
    if hist.empty:
        return pd.DataFrame(columns=["close"])
    out = hist[["Close"]].rename(columns={"Close": "close"})
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def _fetch_es_investing_firecrawl() -> pd.DataFrame:
    out_dir = Path(__file__).resolve().parent / ".firecrawl"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / "es-hist.md"
    stale = True
    # if md_path.is_file():
    #     age = datetime.now() - datetime.fromtimestamp(md_path.stat().st_mtime)
    #     stale = age > timedelta(hours=12)
    if stale:
        cmd = [
            "npx",
            "--yes",
            "firecrawl-cli",
            "scrape",
            INVESTING_ES_URL,
            "--wait-for",
            "4000",
            "--only-main-content",
            "-o",
            str(md_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
    if not md_path.is_file():
        return pd.DataFrame(columns=["close"])
    return parse_investing_markdown(md_path.read_text(encoding="utf-8"))


def fetch_es_futures(lookback_days: int = 90, *, prefer_investing: bool = True) -> tuple[pd.DataFrame, str]:
    """Front ES close. Investing.com when reachable, else Yahoo `ES=F`."""
    # if _CACHE.is_file():
    #     age = datetime.now() - datetime.fromtimestamp(_CACHE.stat().st_mtime)
    #     if age < timedelta(hours=12):
    #         cached = pd.read_parquet(_CACHE)
    #         cached.index = pd.to_datetime(cached.index).tz_localize(None).normalize()
    #         src = str(cached.attrs.get("source", "cache")) if hasattr(cached, "attrs") else "cache"
    #         return cached, f"cache ({src})"

    investing = pd.DataFrame(columns=["close"])
    source = "yahoo ES=F"
    if prefer_investing:
        try:
            investing = _fetch_es_investing_firecrawl()
        except Exception:
            investing = pd.DataFrame(columns=["close"])

    yahoo = _fetch_es_yfinance(lookback_days)
    if not investing.empty and not yahoo.empty:
        combined = yahoo.copy()
        combined.loc[investing.index.intersection(combined.index), "close"] = investing["close"]
        extra = investing.loc[~investing.index.isin(combined.index)]
        if not extra.empty:
            combined = pd.concat([combined, extra]).sort_index()
        out, source = combined, "investing.com + yahoo ES=F"
    elif not investing.empty:
        out, source = investing, "investing.com"
    else:
        out, source = yahoo, "yahoo ES=F"

    if out.empty:
        raise RuntimeError("Could not load ES futures from investing.com or Yahoo.")
    # _CACHE.parent.mkdir(parents=True, exist_ok=True)
    # out.to_parquet(_CACHE)
    return out, source


def rolling_ssr(
    fut_close: pd.Series,
    atm_iv: pd.Series,
    skew_slope: float,
    window: int = 10,
) -> pd.DataFrame:
    """Rolling OLS of Δσ on S_T · Δln F. `atm_iv` and `fut_close` in calendar dates."""
    frame = pd.DataFrame({"F": fut_close, "iv": atm_iv}).dropna().sort_index()
    if len(frame) < 5 or not np.isfinite(skew_slope) or abs(skew_slope) < 1e-8:
        return pd.DataFrame(columns=["dlnF", "div", "ssr"])
    dlnF = np.log(frame["F"]).diff()
    div = frame["iv"].diff()
    # The independent variable for the regression is: x = S_T * dlnF
    # Since S_T (skew slope) is typically negative for equities, and dlnF is negative during selloffs,
    # x is typically positive during selloffs.
    # The dependent variable is: y = div (change in ATM IV), which is typically positive during selloffs.
    # Thus, beta (OLS slope of y on x) is typically positive.
    # If beta > 1, SSR = beta will be > 1 (hyper-sticky-strike).
    # 1.0 is sticky-strike, 0.0 is sticky-delta (when using log-moneyness).
    x = float(skew_slope) * dlnF
    work = pd.DataFrame({"dlnF": dlnF, "div": div, "x": x}).dropna()
    win = min(int(window), max(len(work) - 1, 2))
    min_obs = max(5, win // 2)
    ssr = np.full(len(work), np.nan)
    xv = work["x"].to_numpy()
    yv = work["div"].to_numpy()
    for i in range(len(work)):
        lo = max(0, i + 1 - win)
        if i + 1 - lo < min_obs:
            continue
        xx = float(np.dot(xv[lo : i + 1], xv[lo : i + 1]))
        if xx < 1e-12:
            continue
        ssr[i] = float(np.dot(xv[lo : i + 1], yv[lo : i + 1]) / xx)
    out = work[["dlnF", "div"]].copy()
    out["ssr"] = ssr
    out["ssr_1d"] = yv / np.where(np.abs(xv) < 1e-12, np.nan, xv)
    return out


def smile_atmf_skew(
    ks: np.ndarray | list[float],
    dtes: np.ndarray | list[float],
    iv: np.ndarray | list,
    *,
    k_band: float = 0.05,
) -> pd.DataFrame:
    """ATMF skew S_T = dσ/d ln K at K/S=1 from a (DTE × K/S) IV grid, vol pts."""
    ks = np.asarray(ks, dtype=float)
    dtes = np.asarray(dtes, dtype=float)
    iv = np.asarray(iv, dtype=float)
    empty = pd.DataFrame(columns=["dte", "T", "atm_iv", "S"])
    if iv.ndim != 2 or len(dtes) < 2 or len(ks) < 3:
        return empty
    k = np.log(np.clip(ks, 1e-8, None))
    order = np.argsort(k)
    k, iv = k[order], iv[:, order]
    rows = []
    for i, t_d in enumerate(dtes):
        y = iv[i]
        ok = np.isfinite(y) & np.isfinite(k)
        if int(ok.sum()) < 3:
            continue
        kk, yy = k[ok], y[ok]
        atm = float(np.interp(0.0, kk, yy))
        near = np.abs(kk) <= k_band
        if int(near.sum()) >= 2:
            slope = float(np.polyfit(kk[near], yy[near], 1)[0])
        else:
            slope = float(np.gradient(yy, kk)[int(np.argmin(np.abs(kk)))])
        rows.append({"dte": float(t_d), "T": max(float(t_d), 1.0) / 365.0, "atm_iv": atm, "S": slope})
    return pd.DataFrame(rows)


def implied_ssr(
    ks: np.ndarray | list[float],
    dtes: np.ndarray | list[float],
    iv: np.ndarray | list,
    *,
    min_dte: float = 14.0,
) -> pd.DataFrame:
    """Bergomi time-homogeneous implied SSR from the live skew term structure.

    R_T = 2 + d ln|S_T| / d ln T
    (Smile Dynamics IV §3). 1 sticky-strike, 0 sticky-delta, 2 short-dated local vol.
    No futures / VIX path — today's IV grid only.
    """
    curve = smile_atmf_skew(ks, dtes, iv)
    if curve.empty:
        return pd.DataFrame(columns=["dte", "T", "atm_iv", "S", "ssr", "ssr_power", "gamma"])
    work = curve[curve["dte"] >= min_dte].dropna(subset=["S"]).copy()
    work = work[np.abs(work["S"]) > 1e-8]
    if len(work) < 3:
        work["ssr"] = np.nan
        work["ssr_power"] = np.nan
        work["gamma"] = np.nan
        return work
    log_t = np.log(work["T"].to_numpy())
    log_s = np.log(np.abs(work["S"].to_numpy()))
    x = log_t - log_t.mean()
    y = log_s - log_s.mean()
    xx = float(np.dot(x, x))
    dln_s_dln_t = float(np.dot(x, y) / xx) if xx > 1e-12 else float("nan")
    work["ssr"] = 2.0 + np.gradient(log_s, log_t)
    work["ssr_power"] = 2.0 + dln_s_dln_t
    work["gamma"] = -dln_s_dln_t
    return work
