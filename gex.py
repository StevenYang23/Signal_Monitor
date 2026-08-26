"""Dealer gamma exposure (GEX) from a live option chain.

Dollar GEX per 1% spot move, SpotGamma-style:

    GEX = Γ × OI × M × S² × 0.01 × sign

Calls are +1 and puts are −1 (dealers assumed short the public's net options).
Positive net GEX: dealers long gamma, buy dips / sell rips.
Negative net GEX: dealers short gamma, amplify the move.

TTM buckets are exact business-day tenors (Mon–Fri, not calendar days).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

# Remaining business days to expiry (numpy weekdays; 0 = today). Front week only.
GEX_TTM_BD: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

DEFAULT_MULTIPLIER = 100.0
MAX_PROFILE_BARS = 160


def _is_call(option_type: Any) -> bool:
    text = str(option_type).upper()
    return "CALL" in text or text in {"C", "2"}


def _is_put(option_type: Any) -> bool:
    text = str(option_type).upper()
    return "PUT" in text or text in {"P", "3"}


def _year_fraction(dte: float) -> float:
    """Calendar year fraction. 0DTE uses 1/365 so ATM gamma stays finite."""
    return max(float(dte), 1.0) / 365.0


def bs_gamma(spot: float, strike: float, dte: float, iv_pct: float, r: float = 0.045) -> float:
    """Black-Scholes per-share gamma. `iv_pct` is percent (16.0 = 16%)."""
    sigma = float(iv_pct) / 100.0
    t = _year_fraction(dte)
    if spot <= 0 or strike <= 0 or sigma <= 1e-8:
        return float("nan")
    vol_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    return float(norm.pdf(d1) / (spot * vol_sqrt_t))


def _row_multiplier(row: pd.Series) -> float:
    for col in ("multiplier", "option_contract_multiplier", "contract_size", "option_contract_size"):
        if col not in row.index:
            continue
        val = row[col]
        if pd.notna(val):
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if num > 0:
                return num
    return DEFAULT_MULTIPLIER


def _strike_bin(spot: float) -> float:
    if spot >= 3000:
        return 5.0
    if spot >= 200:
        return 1.0
    return 0.5


def _fmt_notional(value: float | None) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    ax = abs(float(value))
    if ax >= 1e9:
        return f"{value / 1e9:+.2f}B"
    if ax >= 1e6:
        return f"{value / 1e6:+.1f}M"
    if ax >= 1e3:
        return f"{value / 1e3:+.0f}K"
    return f"{value:+.0f}"


def _ttm_key(n: int) -> str:
    return str(n)


def _ttm_label(n: int) -> str:
    return "0 (today)" if n == 0 else str(n)


def _as_day64(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.dt.normalize().to_numpy(dtype="datetime64[D]")


def _business_dte(df: pd.DataFrame) -> pd.Series:
    """Weekdays from asof to expiry, excluding expiry: today=0, next session=1."""
    n = len(df)
    bd = np.zeros(n, dtype=np.int32)
    if "expiry" in df.columns and "asof_date" in df.columns:
        asof = pd.to_datetime(df["asof_date"], errors="coerce")
        exp = pd.to_datetime(df["expiry"], errors="coerce")
        ok = asof.notna() & exp.notna()
        if ok.any():
            bd[ok.to_numpy()] = np.maximum(
                np.busday_count(_as_day64(asof[ok]), _as_day64(exp[ok])),
                0,
            )
        return pd.Series(bd, index=df.index)
    cal = pd.to_numeric(df["dte"], errors="coerce").fillna(0).to_numpy()
    asof = np.datetime64(pd.Timestamp.today().date(), "D")
    starts = np.full(n, asof, dtype="datetime64[D]")
    ends = starts + cal.astype(int).astype("timedelta64[D]")
    bd[:] = np.maximum(np.busday_count(starts, ends), 0)
    return pd.Series(bd, index=df.index)


def _prepare_chain(df: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    out = df.copy()
    if "ks_ratio" not in out.columns:
        out["ks_ratio"] = out["strike"] / float(spot)
    if "oi" not in out.columns:
        out["oi"] = 0.0
    out["oi"] = pd.to_numeric(out["oi"], errors="coerce").fillna(0.0)
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["dte"] = pd.to_numeric(out["dte"], errors="coerce")
    out = out.dropna(subset=["strike", "dte"])
    out["dte_bd"] = _business_dte(out)
    out = out[out["oi"] > 0]

    if "gamma" not in out.columns:
        out["gamma"] = np.nan
    else:
        out["gamma"] = pd.to_numeric(out["gamma"], errors="coerce")

    iv = pd.to_numeric(out["iv"], errors="coerce") if "iv" in out.columns else pd.Series(np.nan, index=out.index)
    need_bs = ~np.isfinite(out["gamma"]) | (out["gamma"].abs() < 1e-12)
    if need_bs.any():
        computed = [
            bs_gamma(spot, float(k), float(d), float(v), r=r)
            if np.isfinite(v)
            else np.nan
            for k, d, v in zip(out.loc[need_bs, "strike"], out.loc[need_bs, "dte"], iv.loc[need_bs])
        ]
        out.loc[need_bs, "gamma"] = computed

    out["multiplier"] = out.apply(_row_multiplier, axis=1)
    scale = float(spot) * float(spot) * 0.01
    signed = np.where(out["option_type"].map(_is_call), 1.0, np.where(out["option_type"].map(_is_put), -1.0, np.nan))
    out["gex"] = out["gamma"] * out["oi"] * out["multiplier"] * scale * signed
    out = out[np.isfinite(out["gex"])]
    return out


def _nearest_expiry_slice(df: pd.DataFrame, ttm_bd: int) -> pd.DataFrame:
    """Contracts on the listed expiry whose business-day TTM is closest to target.

    Only expiries inside `GEX_TTM_BD` (0–5 BD) are eligible.
    """
    empty = df.iloc[0:0]
    if df.empty:
        return empty
    target = int(ttm_bd)
    lo, hi = int(GEX_TTM_BD[0]), int(GEX_TTM_BD[-1])
    in_window = (df["dte_bd"] >= lo) & (df["dte_bd"] <= hi)
    if "expiry" in df.columns:
        exp_d = pd.to_datetime(_as_day64(df["expiry"]))
        per = pd.DataFrame({
            "expiry_d": exp_d.to_numpy(),
            "dte_bd": df["dte_bd"].to_numpy(),
        })
        per = per.loc[in_window.to_numpy()].dropna()
        if not per.empty:
            per = per.groupby("expiry_d", as_index=False)["dte_bd"].first()
            per["dist"] = (per["dte_bd"] - target).abs()
            nearest = per.sort_values(["dist", "expiry_d"])["expiry_d"].iloc[0]
            return df.loc[exp_d == nearest]
        return empty
    work = df.loc[in_window]
    if work.empty:
        return empty
    nearest_bd = int(work["dte_bd"].iloc[(work["dte_bd"] - target).abs().to_numpy().argmin()])
    return work[work["dte_bd"] == nearest_bd]


def _aggregate_strikes(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["strike", "call_gex", "put_gex", "net_gex", "cum_gex"])
    step = _strike_bin(spot)
    work = df.copy()
    work["strike_bin"] = (work["strike"] / step).round() * step
    is_call = work["option_type"].map(_is_call)
    grouped = pd.DataFrame(
        {
            "call_gex": work.loc[is_call].groupby("strike_bin")["gex"].sum(),
            "put_gex": work.loc[~is_call].groupby("strike_bin")["gex"].sum(),
        }
    ).fillna(0.0)
    grouped["net_gex"] = grouped["call_gex"] + grouped["put_gex"]
    grouped = grouped.reset_index().rename(columns={"strike_bin": "strike"}).sort_values("strike")
    # Keep a dense band around spot so the chart can zoom; drop far-OTM dust.
    band = grouped[(grouped["strike"] - spot).abs() <= max(spot * 0.08, 400)]
    if band.empty:
        band = grouped
    if len(band) > MAX_PROFILE_BARS:
        step = _strike_bin(spot)
        coarser = max(step * 2, 10.0)
        band = band.copy()
        band["strike"] = (band["strike"] / coarser).round() * coarser
        band = band.groupby("strike", as_index=False)[["call_gex", "put_gex", "net_gex"]].sum()
    band = band.sort_values("strike")
    band["cum_gex"] = band["net_gex"].cumsum()
    return band[["strike", "call_gex", "put_gex", "net_gex", "cum_gex"]]


def _gamma_flip(profile: pd.DataFrame) -> float | None:
    if profile.empty:
        return None
    cum = profile["net_gex"].cumsum().to_numpy()
    strikes = profile["strike"].to_numpy()
    sign0 = np.sign(cum[0]) if cum[0] != 0 else 0.0
    for i in range(1, len(cum)):
        if cum[i] == 0:
            return float(strikes[i])
        if sign0 != 0 and np.sign(cum[i]) != sign0 and np.sign(cum[i]) != 0:
            x0, x1 = float(strikes[i - 1]), float(strikes[i])
            y0, y1 = float(cum[i - 1]), float(cum[i])
            if y1 == y0:
                return x1
            return x0 + (0.0 - y0) / (y1 - y0) * (x1 - x0)
    return None


def _bucket_payload(df: pd.DataFrame, spot: float, key: str, label: str) -> dict[str, Any]:
    profile = _aggregate_strikes(df, spot)
    call_total = float(df.loc[df["option_type"].map(_is_call), "gex"].sum()) if not df.empty else 0.0
    put_total = float(df.loc[df["option_type"].map(_is_put), "gex"].sum()) if not df.empty else 0.0
    net = call_total + put_total
    call_wall = None
    put_wall = None
    if not profile.empty:
        call_idx = profile["call_gex"].idxmax()
        put_idx = profile["put_gex"].idxmin()
        if profile.loc[call_idx, "call_gex"] > 0:
            call_wall = float(profile.loc[call_idx, "strike"])
        if profile.loc[put_idx, "put_gex"] < 0:
            put_wall = float(profile.loc[put_idx, "strike"])
    flip = _gamma_flip(profile)
    ttm_actual = int(df["dte_bd"].iloc[0]) if not df.empty and "dte_bd" in df.columns else None
    expiry = None
    if not df.empty and "expiry" in df.columns:
        expiry_ts = pd.to_datetime(df["expiry"].iloc[0], errors="coerce")
        if pd.notna(expiry_ts):
            expiry = pd.Timestamp(expiry_ts).strftime("%Y-%m-%d")
    return {
        "key": key,
        "label": label,
        "n_contracts": int(len(df)),
        "ttm_actual": ttm_actual,
        "expiry": expiry,
        "net": net,
        "call": call_total,
        "put": put_total,
        "net_label": _fmt_notional(net),
        "call_label": _fmt_notional(call_total),
        "put_label": _fmt_notional(put_total),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "flip": flip,
        "regime": "long_gamma" if net > 0 else ("short_gamma" if net < 0 else "flat"),
        "strikes": [float(v) for v in profile["strike"]],
        "call_gex": [float(v) for v in profile["call_gex"]],
        "put_gex": [float(v) for v in profile["put_gex"]],
        "net_gex": [float(v) for v in profile["net_gex"]],
        "cum_gex": [float(v) for v in profile["cum_gex"]],
    }


def build_gex_report(
    df: pd.DataFrame,
    *,
    spot: float | None = None,
    r: float = 0.045,
    default_bucket: str = "0",
) -> dict[str, Any]:
    """Build per-TTM GEX profiles for the dashboard."""
    if df is None or df.empty:
        return {"exists": False, "error": "empty chain", "buckets": {}, "default_bucket": default_bucket, "bucket_order": [_ttm_key(n) for n in GEX_TTM_BD]}

    if spot is None:
        spot = float(df["spot"].iloc[0])
    prepared = _prepare_chain(df, float(spot), r)
    buckets: dict[str, dict[str, Any]] = {}
    for n in GEX_TTM_BD:
        key = _ttm_key(n)
        sliced = _nearest_expiry_slice(prepared, n)
        buckets[key] = _bucket_payload(sliced, float(spot), key, _ttm_label(n))

    default = default_bucket
    if not buckets.get(default, {}).get("n_contracts"):
        for n in GEX_TTM_BD:
            key = _ttm_key(n)
            if buckets.get(key, {}).get("n_contracts"):
                default = key
                break

    return {
        "exists": True,
        "spot": float(spot),
        "n_contracts": int(len(prepared)),
        "gamma_source": "vendor+bs_fallback",
        "default_bucket": default,
        "bucket_order": [_ttm_key(n) for n in GEX_TTM_BD],
        "buckets": buckets,
        "note": "TTM is business days (Mon–Fri); each bucket uses the closest listed expiry. OI is the prior-session print. Call+/put− is a dealer-inventory convention.",
    }
