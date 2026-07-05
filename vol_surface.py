from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm

import futu as ft
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_UNDERLYING = "US..SPX"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111
SNAPSHOT_BATCH = 200
SNAPSHOT_PAUSE_SEC = 0.55
TRADING_DAYS = 252
VIX_TICKER = "^VIX"
SPX_TICKER = "^SPX"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Index underlyings: Futu index snapshots often require extra permissions yfinance.
SPOT_YF_DEFAULT: dict[str, str] = {
    "US..SPX": "^SPX",
    "US..NDX": "^NDX",
    "US..IXIC": "^IXIC",
    "US..DJI": "^DJI",
    "US.QQQ": "QQQ",
    "US.DIA": "DIA",
}
# Back-compat alias
SPOT_YF_FALLBACK = SPOT_YF_DEFAULT


@dataclass
class VolSurfaceConfig:
    underlying: str = DEFAULT_UNDERLYING
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    lookback_days: int = 5
    max_dte: int = 60
    min_dte: int = 1
    moneyness_min: float = 0.88
    moneyness_max: float = 1.12
    min_open_interest: int = 1
    min_iv: float = 1.0
    max_iv: float = 200.0
    risk_free_rate: float = 0.045
    term_dtes: tuple[int, ...] = (7, 10, 14, 21, 30, 45, 60)
    event_hump_dte_range: tuple[int, int] = (5, 20)
    anchor_dtes: tuple[int, ...] = (7, 30, 60)
    anchor_delta: float = 0.25
    anchor_hv_period: int = 22
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve() / "research" / "data" / "vol_surface")
    use_deepseek: bool = False

    def cache_path(self, asof: date | str) -> Path:
        asof_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
        safe_code = self.underlying.replace(".", "_")
        return self.cache_dir / safe_code / f"{asof_str}.parquet"


def _require_futu() -> None:
    if ft is None:
        raise ImportError("futu-api is not installed. Run: pip install futu-api")


def futu_anchor_iv_supported() -> bool:
    """True if this futu-api build exposes option IV history (get_option_volatility)."""
    if hasattr(ft.OpenQuoteContext, "get_option_volatility"):
        return True
    try:
        from futu.quote.quote_query import GetOptionVolatilityQuery  # noqa: F401
        return True
    except ImportError:
        return False


def _futu_get_option_volatility(
    quote_ctx: Any,
    code: str,
    query_time_period: int | None = None,
    hv_time_period: int | None = None,
) -> tuple[int, pd.DataFrame | str]:
    """Compat wrapper: SDK method (>=10.07) or direct query fallback."""
    if hasattr(quote_ctx, "get_option_volatility"):
        return quote_ctx.get_option_volatility(
            code,
            query_time_period=query_time_period,
            hv_time_period=hv_time_period,
        )

    try:
        from futu.quote.quote_query import GetOptionVolatilityQuery
    except ImportError:
        ver = getattr(ft, "__version__", "unknown")
        return ft.RET_ERROR, (
            f"Anchor IV history requires futu-api >= 10.07 (installed {ver}). "
            "In this Jupyter kernel run: pip install -U futu-api"
        )

    query_processor = quote_ctx._get_sync_query_processor(
        GetOptionVolatilityQuery.pack_req,
        GetOptionVolatilityQuery.unpack,
    )
    kargs = {
        "code": code,
        "conn_id": quote_ctx.get_sync_conn_id(),
        "query_time_period": query_time_period,
        "hv_time_period": hv_time_period,
    }
    ret_code, msg, ret = query_processor(**kargs)
    if ret_code == ft.RET_ERROR:
        return ret_code, msg

    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if isinstance(ret, dict):
        meta["average_impvol"] = ret.get("average_impvol")
        meta["impvol_status"] = ret.get("impvol_status")
        meta["analysis"] = ret.get("analysis", "")
        for item in ret.get("item_list", []):
            row = dict(item)
            row.update(meta)
            rows.append(row)

    col_list = [
        "timestamp",
        "timestamp_str",
        "implied_volatility",
        "history_volatility",
        "volatility_premium",
        "average_impvol",
        "impvol_status",
        "analysis",
    ]
    if not rows:
        return ft.RET_OK, pd.DataFrame(columns=col_list)
    frame = pd.DataFrame(rows)
    for c in col_list:
        if c not in frame.columns:
            frame[c] = None
    return ft.RET_OK, frame[col_list]


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_asof(update_time: str | None) -> date:
    if not update_time:
        return date.today()
    return pd.Timestamp(update_time).date()


def _standardize_chain(raw: pd.DataFrame, spot: float, asof: date) -> pd.DataFrame:
    df = raw.copy()
    df["spot"] = float(spot)
    df["asof_date"] = pd.Timestamp(asof)
    df["expiry"] = pd.to_datetime(df["strike_time"])
    df["dte"] = (df["expiry"] - df["asof_date"]).dt.days.clip(lower=0)
    df["moneyness"] = np.where(
        df["option_type"].str.upper().str.contains("CALL"),
        df["spot"] / df["option_strike_price"],
        df["option_strike_price"] / df["spot"],
    )
    df["log_moneyness"] = np.log(df["option_strike_price"] / df["spot"])
    df["iv"] = pd.to_numeric(df["option_implied_volatility"], errors="coerce")
    df["delta"] = pd.to_numeric(df["option_delta"], errors="coerce")
    df["oi"] = pd.to_numeric(df["option_open_interest"], errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["bid"] = pd.to_numeric(df["bid_price"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask_price"], errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["strike"] = df["option_strike_price"]
    df["ks_ratio"] = df["option_strike_price"] / df["spot"]
    return df


def clean_option_chain(df: pd.DataFrame, cfg: VolSurfaceConfig) -> pd.DataFrame:
    out = df.copy()
    if "option_valid" in out.columns:
        out = out[out["option_valid"].fillna(False)]
    out = out[out["dte"].between(cfg.min_dte, cfg.max_dte)]
    out = out[out["moneyness"].between(cfg.moneyness_min, cfg.moneyness_max)]
    out = out[out["iv"].between(cfg.min_iv, cfg.max_iv)]
    if cfg.min_open_interest > 0:
        out = out[out["oi"] >= cfg.min_open_interest]
    out = out.dropna(subset=["iv", "strike", "dte"])
    return out.sort_values(["dte", "strike"]).reset_index(drop=True)


def fetch_spot_yfinance(ticker: str) -> tuple[float, date]:
    if yf is None:
        raise ImportError("yfinance is required for index spot fallback")
    raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError(f"No yfinance close for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    price = float(close.dropna().iloc[-1])
    asof = pd.Timestamp(close.dropna().index[-1]).date()
    return price, asof


def fetch_spot(quote_ctx: Any, code: str) -> tuple[float, date]:
    yf_ticker = SPOT_YF_DEFAULT.get(code)
    if yf_ticker:
        return fetch_spot_yfinance(yf_ticker)

    ret, snap = quote_ctx.get_market_snapshot([code])
    if ret == ft.RET_OK and not snap.empty:
        row = snap.iloc[0]
        price = float(row.get("last_price") or row.get("prev_close_price"))
        asof = _parse_asof(row.get("update_time"))
        return price, asof

    raise RuntimeError(f"Failed to fetch spot for {code}: {snap}")


def fetch_option_chain_futu(
    quote_ctx: Any,
    cfg: VolSurfaceConfig,
    spot: float | None = None,
    asof: date | None = None,
) -> pd.DataFrame:
    _require_futu()
    if spot is None or asof is None:
        spot, asof = fetch_spot(quote_ctx, cfg.underlying)

    ret, exp = quote_ctx.get_option_expiration_date(cfg.underlying)
    if ret != ft.RET_OK or exp.empty:
        raise RuntimeError(f"Failed to fetch option expirations for {cfg.underlying}: {exp}")

    expiries = pd.to_datetime(exp["strike_time"])
    end_date = (asof + timedelta(days=cfg.max_dte)).strftime("%Y-%m-%d")
    start_date = asof.strftime("%Y-%m-%d")
    valid = expiries[(expiries >= pd.Timestamp(asof)) & (expiries <= pd.Timestamp(end_date))]
    if valid.empty:
        raise RuntimeError(f"No option expiries within {cfg.max_dte}d for {cfg.underlying}")

    chain_parts: list[pd.DataFrame] = []
    window_start = pd.Timestamp(asof)
    window_end_limit = pd.Timestamp(end_date)
    while window_start <= window_end_limit:
        batch_end = min(window_start + timedelta(days=29), window_end_limit)
        ret, chain = quote_ctx.get_option_chain(
            cfg.underlying,
            start=window_start.strftime("%Y-%m-%d"),
            end=batch_end.strftime("%Y-%m-%d"),
        )
        if ret == ft.RET_OK and not chain.empty:
            chain_parts.append(chain)
        window_start = batch_end + timedelta(days=1)

    if not chain_parts:
        raise RuntimeError(f"Empty option chain for {cfg.underlying}")

    chain_meta = pd.concat(chain_parts, ignore_index=True).drop_duplicates(subset=["code"])
    chain_meta["expiry"] = pd.to_datetime(chain_meta["strike_time"])
    chain_meta["dte"] = (chain_meta["expiry"] - pd.Timestamp(asof)).dt.days
    strike_lo = spot / cfg.moneyness_max
    strike_hi = spot / cfg.moneyness_min
    chain_meta = chain_meta[
        chain_meta["dte"].between(cfg.min_dte, cfg.max_dte)
        & chain_meta["strike_price"].between(strike_lo, strike_hi)
    ]
    codes = chain_meta["code"].dropna().unique().tolist()
    logger.info("Requesting snapshots for %d option contracts (DTE<=%d)", len(codes), cfg.max_dte)

    snap_parts: list[pd.DataFrame] = []
    for i, batch in enumerate(_chunked(codes, SNAPSHOT_BATCH)):
        if i > 0:
            time.sleep(SNAPSHOT_PAUSE_SEC)
        ret, snap = quote_ctx.get_market_snapshot(batch)
        if ret == ft.RET_OK and not snap.empty:
            snap_parts.append(snap)
        else:
            logger.warning("Snapshot batch failed: %s", snap)
            time.sleep(1.0)

    if not snap_parts:
        raise RuntimeError("All option snapshot requests failed.")

    snap = pd.concat(snap_parts, ignore_index=True).drop_duplicates(subset=["code"])
    std = _standardize_chain(snap, spot=spot, asof=asof)
    return clean_option_chain(std, cfg)


def ensure_runtime_deps(*, upgrade_futu: bool = True, install_pyarrow: bool = True) -> str:
    """Install/upgrade packages in the active Jupyter kernel. Returns futu version."""
    import subprocess

    global ft

    if install_pyarrow:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "-q"])

    if ft is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "futu-api", "-q"])
        for mod in list(sys.modules):
            if mod == "futu" or mod.startswith("futu."):
                del sys.modules[mod]
        import futu as ft  # noqa: F811

    if upgrade_futu and not futu_anchor_iv_supported():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "futu-api", "-q"])
        for mod in list(sys.modules):
            if mod == "futu" or mod.startswith("futu."):
                del sys.modules[mod]
        import futu as ft  # noqa: F811

    if not futu_anchor_iv_supported():
        ver = getattr(ft, "__version__", "unknown")
        raise RuntimeError(
            f"futu-api {ver} in {sys.executable} still lacks get_option_volatility. "
            "Use Kernel �?Restart, then re-run from cell 1."
        )
    return getattr(ft, "__version__", "unknown")


def save_surface(df: pd.DataFrame, cfg: VolSurfaceConfig, asof: date | None = None) -> Path:
    asof = asof or pd.Timestamp(df["asof_date"].iloc[0]).date()
    path = cfg.cache_path(asof)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    meta = {
        "underlying": cfg.underlying,
        "asof": str(asof),
        "rows": len(df),
        "spot": float(df["spot"].iloc[0]),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_surface(cfg: VolSurfaceConfig, asof: date | str) -> pd.DataFrame | None:
    path = cfg.cache_path(asof)
    csv_path = path.with_suffix(".csv")
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "asof_date" in df.columns:
            df["asof_date"] = pd.to_datetime(df["asof_date"])
        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"])
        return df
    return None


def list_cached_surface_dates(cfg: VolSurfaceConfig) -> list[date]:
    cache_dir = cfg.cache_dir / cfg.underlying.replace(".", "_")
    if not cache_dir.exists():
        return []
    dates: list[date] = []
    for path in sorted(cache_dir.glob("*.parquet")):
        try:
            dates.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(dates)


def business_days_only(dates: Iterable[date]) -> list[date]:
    return sorted(d for d in dates if d.weekday() < 5)


def latest_business_session(surfaces: dict[date, pd.DataFrame]) -> date:
    biz = business_days_only(list(surfaces.keys()))
    if biz:
        return biz[-1]
    return sorted(surfaces.keys())[-1]


def trim_surfaces_to_sessions(
    surfaces: dict[date, pd.DataFrame],
    n_sessions: int,
    *,
    business_days: bool = True,
) -> dict[date, pd.DataFrame]:
    dates = business_days_only(list(surfaces.keys())) if business_days else sorted(surfaces.keys())
    keep = dates[-n_sessions:]
    return {d: surfaces[d] for d in keep}


def load_surface_history(cfg: VolSurfaceConfig, n_days: int | None = None) -> dict[date, pd.DataFrame]:
    n_days = n_days or cfg.lookback_days
    cached = list_cached_surface_dates(cfg)
    if cached:
        use_dates = cached[-(n_days + 1) :]
        history: dict[date, pd.DataFrame] = {}
        for d in use_dates:
            df = load_surface(cfg, d)
            if df is not None:
                history[d] = df
        if history:
            return history
    history = {}
    for offset in range(n_days + 1):
        d = date.today() - timedelta(days=offset)
        df = load_surface(cfg, d)
        if df is not None:
            history[d] = df
    return history


def fetch_and_cache(cfg: VolSurfaceConfig | None = None, *, save: bool = True) -> pd.DataFrame:
    cfg = cfg or VolSurfaceConfig()
    _require_futu()
    quote_ctx = ft.OpenQuoteContext(host=cfg.host, port=cfg.port)
    try:
        df = fetch_option_chain_futu(quote_ctx, cfg)
        if save:
            save_surface(df, cfg)
        return df
    finally:
        quote_ctx.close()


def bs_call_price(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    tiny = (T <= 1e-6) | (sigma <= 1e-6)
    vol_sqrt_t = sigma * np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(tiny, intrinsic, price)


def _interp_at_target(
    df: pd.DataFrame,
    target_dte: float,
    *,
    on: str,
    target_value: float,
    option_type: str | None = None,
) -> float:
    sub = df[np.isfinite(df[on]) & np.isfinite(df["iv"])].copy()
    if option_type is not None:
        sub = sub[sub["option_type"].str.upper().str.contains(option_type.upper())]
    if sub.empty:
        return np.nan

    dtes = sub["dte"].unique()
    if len(dtes) == 0:
        return np.nan
    nearest_dte = dtes[np.argmin(np.abs(dtes - target_dte))]
    slice_df = sub[sub["dte"] == nearest_dte].sort_values(on)
    if len(slice_df) < 2:
        return float(slice_df["iv"].iloc[0]) if len(slice_df) == 1 else np.nan

    x = slice_df[on].to_numpy()
    y = slice_df["iv"].to_numpy()
    if target_value < x.min() or target_value > x.max():
        return np.nan
    return float(np.interp(target_value, x, y))


def atm_iv(df: pd.DataFrame, target_dte: float) -> float:
    dtes = sorted(df["dte"].unique())
    xs: list[float] = []
    ys: list[float] = []
    for d in dtes:
        val = _interp_at_target(df, float(d), on="log_moneyness", target_value=0.0)
        if np.isfinite(val):
            xs.append(float(d))
            ys.append(float(val))
    if not xs:
        return np.nan
    if target_dte <= xs[0]:
        return ys[0]
    if target_dte >= xs[-1]:
        return ys[-1]
    return float(np.interp(target_dte, xs, ys))


def delta_iv(df: pd.DataFrame, target_dte: float, target_delta: float, option_type: str) -> float:
    sub = df[df["option_type"].str.upper().str.contains(option_type.upper())].copy()
    if sub.empty:
        return np.nan
    dtes = sub["dte"].unique()
    nearest_dte = dtes[np.argmin(np.abs(dtes - target_dte))]
    slice_df = sub[sub["dte"] == nearest_dte].sort_values("delta")
    if len(slice_df) < 2:
        return np.nan
    x = slice_df["delta"].to_numpy()
    y = slice_df["iv"].to_numpy()
    if option_type.upper().startswith("P"):
        target_delta = -abs(target_delta)
    else:
        target_delta = abs(target_delta)
    if target_delta < x.min() or target_delta > x.max():
        return np.nan
    return float(np.interp(target_delta, x, y))


def compute_surface_features(df: pd.DataFrame, cfg: VolSurfaceConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VolSurfaceConfig()
    spot = float(df["spot"].iloc[0])
    asof = pd.Timestamp(df["asof_date"].iloc[0]).date()

    term: dict[str, float] = {}
    for dte in cfg.term_dtes:
        term[f"atm_iv_{dte}d"] = atm_iv(df, dte)

    front_key = f"atm_iv_{cfg.term_dtes[0]}d"
    back_key = f"atm_iv_{cfg.term_dtes[-1]}d"
    front_iv = term.get(front_key, np.nan)
    back_iv = term.get(back_key, np.nan)

    iv_30 = term.get("atm_iv_30d", atm_iv(df, 30))
    put_25 = delta_iv(df, 30, 0.25, "PUT")
    call_25 = delta_iv(df, 30, 0.25, "CALL")
    put_10 = delta_iv(df, 30, 0.10, "PUT")
    call_10 = delta_iv(df, 30, 0.10, "CALL")

    skew_25 = put_25 - iv_30 if np.isfinite(put_25) and np.isfinite(iv_30) else np.nan
    call_skew_25 = call_25 - iv_30 if np.isfinite(call_25) and np.isfinite(iv_30) else np.nan
    butterfly = (
        (put_25 + call_25) / 2.0 - iv_30
        if all(np.isfinite(v) for v in (put_25, call_25, iv_30))
        else np.nan
    )
    put_slope = (put_10 - iv_30) if np.isfinite(put_10) and np.isfinite(iv_30) else np.nan
    call_slope = (call_10 - iv_30) if np.isfinite(call_10) and np.isfinite(iv_30) else np.nan

    term_curve = [term[f"atm_iv_{d}d"] for d in cfg.term_dtes if np.isfinite(term[f"atm_iv_{d}d"])]
    term_dtes_valid = [d for d in cfg.term_dtes if np.isfinite(term[f"atm_iv_{d}d"])]
    term_slope = np.nan
    term_curvature = np.nan
    if len(term_curve) >= 2:
        term_slope = term_curve[0] - term_curve[-1]
    if len(term_curve) >= 3:
        mid = len(term_curve) // 2
        term_curvature = term_curve[mid] - (term_curve[0] + term_curve[-1]) / 2.0

    return {
        "asof_date": str(asof),
        "spot": spot,
        "atm_iv_30d": iv_30,
        "skew_25d": skew_25,
        "call_skew_25d": call_skew_25,
        "butterfly_25d": butterfly,
        "put_slope_10d": put_slope,
        "call_slope_10d": call_slope,
        "term_slope": term_slope,
        "term_curvature": term_curvature,
        "front_iv": front_iv,
        "back_iv": back_iv,
        **term,
        "term_dtes": term_dtes_valid,
        "term_curve": term_curve,
    }


def build_iv_grid(
    df: pd.DataFrame,
    dte_grid: np.ndarray | None = None,
    ks_grid: np.ndarray | None = None,
    max_dte: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate IV onto a (DTE x K/S) mesh from OTM options only.

    Uses OTM puts for strikes below spot (K/S < 1) and OTM calls for
    strikes above spot (K/S > 1), with a small ATM buffer where both
    types are included for a smooth interpolation across the smile.
    """
    dte_grid = dte_grid if dte_grid is not None else np.array([7, 10, 14, 21, 30, 45, 60])
    dte_grid = dte_grid[dte_grid <= max_dte]
    ks_grid = ks_grid if ks_grid is not None else np.linspace(0.85, 1.15, 13)

    sub = df[np.isfinite(df["iv"]) & np.isfinite(df["dte"])].copy()
    if "ks_ratio" not in sub.columns:
        sub["ks_ratio"] = sub["option_strike_price"] / sub["spot"]

    # OTM selection with ATM buffer: puts on left wing, calls on right wing
    atm_half_width = 0.03
    is_put = sub["option_type"].str.upper().str.contains("PUT")
    is_call = sub["option_type"].str.upper().str.contains("CALL")
    otm_mask = (
        (sub["ks_ratio"] < 1.0 - atm_half_width) & is_put
    ) | (
        (sub["ks_ratio"] > 1.0 + atm_half_width) & is_call
    ) | (
        sub["ks_ratio"].between(1.0 - atm_half_width, 1.0 + atm_half_width)
    )
    sub = sub[otm_mask]

    if len(sub) < 4:
        raise ValueError("Not enough OTM option quotes to build IV grid.")

    points = sub[["dte", "ks_ratio", "iv"]].dropna().to_numpy()
    grid_dte, grid_ks = np.meshgrid(dte_grid, ks_grid, indexing="ij")
    iv_grid = interpolate.griddata(
        points[:, :2],
        points[:, 2],
        (grid_dte, grid_ks),
        method="linear",
    )
    if np.isnan(iv_grid).any():
        iv_grid = interpolate.griddata(
            points[:, :2],
            points[:, 2],
            (grid_dte, grid_ks),
            method="nearest",
        )
    iv_grid = np.clip(iv_grid, 1.0, 200.0)
    return grid_dte, grid_ks, iv_grid


def smooth_iv_grid_quadratic(
    df: pd.DataFrame,
    grid_dte: np.ndarray,
    grid_ks: np.ndarray,
    iv_grid: np.ndarray,
    *,
    dte_window: int = 4,
    iv_lo: float = 5.0,
    iv_hi: float = 150.0,
) -> np.ndarray:
    """Per-DTE quadratic smile in log-moneyness; falls back to the raw IV slice."""
    sub = df[np.isfinite(df["iv"]) & np.isfinite(df["dte"])].copy()
    if "ks_ratio" not in sub.columns:
        sub["ks_ratio"] = sub["option_strike_price"] / sub["spot"]
    grouped = sub.groupby(["dte", "ks_ratio"])["iv"].mean().reset_index()
    dte_axis = grid_dte[:, 0]
    ks_axis = grid_ks[0, :]
    iv_smooth = np.zeros_like(iv_grid)
    for i, dte in enumerate(dte_axis):
        slice_df = grouped[np.abs(grouped["dte"] - dte) <= dte_window]
        if len(slice_df) >= 3:
            x_pts = np.log(slice_df["ks_ratio"].to_numpy())
            y_pts = slice_df["iv"].to_numpy()
            coeffs = np.polyfit(x_pts, y_pts, 2)
            if coeffs[0] < 0:
                coeffs = np.polyfit(x_pts, y_pts, 1)
                coeffs = np.array([0.0, *coeffs])
            for j, ks in enumerate(ks_axis):
                iv_smooth[i, j] = np.clip(np.polyval(coeffs, np.log(ks)), iv_lo, iv_hi)
        else:
            iv_smooth[i, :] = iv_grid[i, :]
    return iv_smooth


def _sanitize_dupire_local_vol(
    local_vol: np.ndarray,
    iv_grid: np.ndarray,
    denom: np.ndarray,
    d2C_dK2: np.ndarray,
) -> np.ndarray:
    """Drop butterfly-violating / boundary nodes and cap vs local implied vol."""
    bad = (denom <= 1e-4) | (d2C_dK2 <= 0) | ~np.isfinite(local_vol)
    out = np.where(bad, np.nan, local_vol)
    out[0, :] = np.nan
    out[-1, :] = np.nan
    out[:, 0] = np.nan
    out[:, -1] = np.nan
    cap = np.clip(iv_grid * 2.5, 40.0, 180.0)
    return np.minimum(out, cap)


def dupire_local_vol(
    spot: float,
    grid_dte: np.ndarray,
    grid_ks: np.ndarray,
    iv_grid: np.ndarray,
    r: float = 0.045,
) -> np.ndarray:
    """Dupire local vol (%) from an implied-vol grid."""
    t_years = np.maximum(grid_dte / TRADING_DAYS, 1 / TRADING_DAYS)
    strike_grid = grid_ks * spot
    call_grid = bs_call_price(spot, strike_grid, t_years, r, iv_grid / 100.0)

    t_axis = t_years[:, 0]
    k_axis = strike_grid[0, :]
    dC_dT = np.gradient(call_grid, t_axis, axis=0)
    dC_dK = np.gradient(call_grid, k_axis, axis=1)
    d2C_dK2 = np.gradient(dC_dK, k_axis, axis=1)

    # Dupire: σ² = 2(∂C/∂T + rK ∂C/∂K) / (K² ∂²C/∂K²)
    numer = dC_dT + r * strike_grid * dC_dK
    denom = strike_grid**2 * d2C_dK2
    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * numer / denom
    local_vol = np.sqrt(np.clip(local_var, 0.0, None)) * 100.0
    return _sanitize_dupire_local_vol(local_vol, iv_grid, denom, d2C_dK2)


def strike_from_delta(
    spot: float,
    t_years: float,
    delta: float,
    iv_pct: float,
    r: float = 0.045,
) -> float:
    """Invert Black–Scholes delta to strike (call delta for δ>0, put delta for δ<0)."""
    sigma = max(float(iv_pct) / 100.0, 1e-4)
    t_years = max(float(t_years), 1.0 / TRADING_DAYS)
    sqrt_t = np.sqrt(t_years)
    target = float(delta)

    def bs_call_delta(k: float) -> float:
        d1 = (np.log(spot / k) + (r + 0.5 * sigma**2) * t_years) / (sigma * sqrt_t)
        return float(norm.cdf(d1))

    def objective(k: float) -> float:
        if target >= 0:
            return bs_call_delta(k) - target
        return bs_call_delta(k) - 1.0 - target

    lo, hi = spot * 0.55, spot * 1.45
    f_lo, f_hi = objective(lo), objective(hi)
    expand = 0
    while f_lo * f_hi > 0 and expand < 6:
        lo *= 0.85
        hi *= 1.15
        f_lo, f_hi = objective(lo), objective(hi)
        expand += 1
    if f_lo * f_hi > 0:
        return spot
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if objective(lo) * objective(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_iv_grid_delta(
    df: pd.DataFrame,
    delta_grid: np.ndarray | None = None,
    dte_grid: np.ndarray | None = None,
    max_dte: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate IV onto a (DTE x delta) mesh, OTM put/call + ATM buffer.

    Uses OTM puts (delta < 0) for the left wing and OTM calls (delta > 0)
    for the right wing, with a small ATM buffer for smooth interpolation.
    """
    dte_grid = dte_grid if dte_grid is not None else np.array([7, 10, 14, 21, 30, 45, 60])
    dte_grid = dte_grid[dte_grid <= max_dte]
    delta_grid = delta_grid if delta_grid is not None else np.linspace(-0.5, 0.5, 21)

    sub = df[np.isfinite(df["iv"]) & np.isfinite(df["delta"]) & np.isfinite(df["dte"])].copy()

    # OTM selection: puts on left wing (delta < 0), calls on right wing (delta > 0), ATM buffer
    atm_half_width = 0.03
    is_put = sub["option_type"].str.upper().str.contains("PUT")
    is_call = sub["option_type"].str.upper().str.contains("CALL")
    otm_mask = (
        (sub["delta"] < -atm_half_width) & is_put
    ) | (
        (sub["delta"] > atm_half_width) & is_call
    ) | (
        sub["delta"].abs() <= atm_half_width
    )
    sub = sub[otm_mask]

    if len(sub) < 4:
        raise ValueError("Not enough OTM option quotes to build delta IV grid.")

    points = sub[["dte", "delta", "iv"]].to_numpy()
    grid_dte, grid_delta = np.meshgrid(dte_grid, delta_grid, indexing="ij")
    iv_grid = interpolate.griddata(
        points[:, :2],
        points[:, 2],
        (grid_dte, grid_delta),
        method="linear",
    )
    if np.isnan(iv_grid).any():
        iv_grid = interpolate.griddata(
            points[:, :2],
            points[:, 2],
            (grid_dte, grid_delta),
            method="nearest",
        )
    iv_grid = np.clip(iv_grid, 1.0, 200.0)
    return grid_dte, grid_delta, iv_grid


def dupire_local_vol_delta(
    spot: float,
    grid_dte: np.ndarray,
    grid_delta: np.ndarray,
    iv_grid: np.ndarray,
    r: float = 0.045,
) -> np.ndarray:
    """Dupire local vol (%) from an IV surface gridded on (DTE, delta)."""
    t_years = np.maximum(grid_dte / TRADING_DAYS, 1.0 / TRADING_DAYS)
    strike_grid = np.zeros_like(iv_grid)
    n_dte, n_delta = iv_grid.shape
    for i in range(n_dte):
        for j in range(n_delta):
            strike_grid[i, j] = strike_from_delta(
                spot,
                t_years[i, j],
                grid_delta[i, j],
                iv_grid[i, j],
                r=r,
            )

    call_grid = bs_call_price(spot, strike_grid, t_years, r, iv_grid / 100.0)
    t_axis = t_years[:, 0]
    dC_dT = np.gradient(call_grid, t_axis, axis=0)
    n_dte, n_delta = call_grid.shape
    dC_dK = np.empty_like(call_grid)
    d2C_dK2 = np.empty_like(call_grid)
    for i in range(n_dte):
        k_row = strike_grid[i, :]
        dC_dK[i, :] = np.gradient(call_grid[i, :], k_row)
        d2C_dK2[i, :] = np.gradient(dC_dK[i, :], k_row)

    numer = dC_dT + r * strike_grid * dC_dK
    denom = strike_grid**2 * d2C_dK2
    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * numer / denom
    local_vol = np.sqrt(np.clip(local_var, 0.0, None)) * 100.0
    return _sanitize_dupire_local_vol(local_vol, iv_grid, denom, d2C_dK2)


def _delta_band_stats(df: pd.DataFrame, delta_lo: float, delta_hi: float) -> dict[str, float]:
    sub = df[
        np.isfinite(df["iv"])
        & np.isfinite(df["delta"])
        & (df["delta"] >= delta_lo)
        & (df["delta"] <= delta_hi)
    ]
    puts = sub[sub["delta"] <= -0.005]
    calls = sub[sub["delta"] >= 0.005]
    put_mean = float(puts["iv"].mean()) if len(puts) else np.nan
    call_mean = float(calls["iv"].mean()) if len(calls) else np.nan
    return {
        "mean_iv": float(sub["iv"].mean()) if len(sub) else np.nan,
        "put_mean_iv": put_mean,
        "call_mean_iv": call_mean,
        "put_call_spread": put_mean - call_mean if np.isfinite(put_mean) and np.isfinite(call_mean) else np.nan,
        "n_quotes": len(sub),
        "n_puts": len(puts),
        "n_calls": len(calls),
    }


def analyze_delta_band_5d(
    surfaces: dict[date, pd.DataFrame],
    delta_lo: float = -0.05,
    delta_hi: float = 0.05,
) -> dict[str, Any]:
    """Compare ATM-near IV level and put/call balance over cached business days."""
    dates = sorted(surfaces.keys())
    if not dates:
        return {}
    stats_by_day = {d: _delta_band_stats(surfaces[d], delta_lo, delta_hi) for d in dates}
    today_d = dates[-1]
    today = stats_by_day[today_d]
    hist = [stats_by_day[d] for d in dates[:-1]]
    hist_mean_iv = float(np.nanmean([h["mean_iv"] for h in hist])) if hist else np.nan
    hist_put_spread = float(np.nanmean([h["put_call_spread"] for h in hist])) if hist else np.nan
    prev = stats_by_day[dates[-2]] if len(dates) >= 2 else None

    return {
        "today_date": str(today_d),
        "delta_band": (delta_lo, delta_hi),
        "today": today,
        "history_mean_iv": hist_mean_iv,
        "history_put_call_spread": hist_put_spread,
        "iv_vs_5d_mean": today["mean_iv"] - hist_mean_iv if np.isfinite(hist_mean_iv) else np.nan,
        "put_spread_vs_5d_mean": today["put_call_spread"] - hist_put_spread if np.isfinite(hist_put_spread) else np.nan,
        "put_spread_vs_prev": (
            today["put_call_spread"] - prev["put_call_spread"]
            if prev and np.isfinite(prev["put_call_spread"])
            else np.nan
        ),
        "stats_by_day": {str(d): stats_by_day[d] for d in dates},
    }


def detect_iv_ml_anomalies(
    df: pd.DataFrame,
    residual_pct: float = 98.0,
    residual_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Flag option quotes whose IV deviates strongly from a smooth ML surface fit."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict
    except ImportError:
        return {"anomalies": pd.DataFrame(), "n_anomalies": 0, "threshold": np.nan, "skipped": "sklearn missing"}

    sub = df[np.isfinite(df["iv"]) & np.isfinite(df["delta"]) & np.isfinite(df["dte"])].copy()
    if len(sub) < 80:
        return {"anomalies": pd.DataFrame(), "n_anomalies": 0, "threshold": np.nan, "skipped": "too few quotes"}

    sub["is_put"] = sub["option_type"].str.upper().str.contains("PUT").astype(int)
    x = sub[["dte", "delta", "is_put"]].to_numpy()
    y = sub["iv"].to_numpy()
    model = GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=42)
    y_hat = cross_val_predict(model, x, y, cv=min(5, max(2, len(sub) // 40)))
    residual = np.abs(y - y_hat)
    threshold = float(np.nanpercentile(residual, residual_pct) * residual_multiplier)
    mask = residual > threshold
    anomalies = sub.loc[mask, ["code", "option_type", "dte", "delta", "iv", "strike"]].copy()
    anomalies["predicted_iv"] = y_hat[mask]
    anomalies["residual"] = residual[mask]
    return {
        "anomalies": anomalies.sort_values("residual", ascending=False),
        "n_anomalies": int(mask.sum()),
        "threshold": threshold,
        "median_residual": float(np.median(residual)),
    }


def detect_local_vol_spikes(
    local_vol: np.ndarray,
    grid_dte: np.ndarray,
    grid_delta: np.ndarray,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Find grid nodes where local vol spikes vs a local neighborhood."""
    lv = local_vol.copy()
    finite = np.isfinite(lv)
    if not finite.any():
        return pd.DataFrame()
    med = float(np.nanmedian(lv))
    mad = float(np.nanmedian(np.abs(lv - med))) or 1.0
    rows: list[dict[str, Any]] = []
    for i in range(1, lv.shape[0] - 1):
        for j in range(1, lv.shape[1] - 1):
            center = lv[i, j]
            if not np.isfinite(center):
                continue
            hood = lv[i - 1 : i + 2, j - 1 : j + 2]
            hood_med = float(np.nanmedian(hood))
            hood_std = float(np.nanstd(hood)) or mad
            z = (center - hood_med) / max(hood_std, 0.5)
            if z >= z_threshold and center > hood_med + 2.0:
                rows.append(
                    {
                        "dte": float(grid_dte[i, j]),
                        "delta": float(grid_delta[i, j]),
                        "local_vol": float(center),
                        "neighbor_median": hood_med,
                        "z_score": float(z),
                    }
                )
    return pd.DataFrame(rows).sort_values("z_score", ascending=False) if rows else pd.DataFrame()


def deepseek_study_conclusion(
    draft: str,
    delta_ctx: dict[str, Any],
    ml_ctx: dict[str, Any],
    lv_spikes: pd.DataFrame,
    sentiment: dict[str, str],
    api_key: str | None = None,
) -> str:
    api_key = api_key or load_deepseek_api_key()
    if not api_key:
        return draft
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an equity derivatives strategist. Write a concise plain-English conclusion "
                    "as bullet points (4�? lines, each starting with '- '). Cover vol level, put/call skew "
                    "shift, sentiment, and any local-vol anomalies. Use only supplied facts. "
                    "No section headings or numbered lists."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft:\n{draft}\n\n"
                    f"Delta-band stats: {json.dumps(delta_ctx, default=str)}\n"
                    f"ML IV outliers: n={ml_ctx.get('n_anomalies', 0)}\n"
                    f"Local vol spikes: {len(lv_spikes)} points\n"
                    f"Sentiment: {json.dumps(sentiment)}"
                ),
            },
        ],
        "temperature": 0.25,
    }
    req = request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except (error.URLError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("DeepSeek conclusion failed: %s", exc)
        return draft


def build_study_conclusion(
    study: "VolSurfaceStudy",
    delta_lo: float = -0.05,
    delta_hi: float = 0.05,
    use_deepseek: bool | None = None,
) -> str:
    """Plain-word conclusion: vol level, put/call skew, sentiment, local-vol checks."""
    if not study.surfaces:
        study.load_history()
    dates = sorted(study.surfaces.keys())
    if not dates:
        return "No cached vol surface data available."

    today_d = latest_business_session(study.surfaces)
    today_df = study.surfaces[today_d]
    delta_ctx = analyze_delta_band_5d(study.surfaces, delta_lo, delta_hi)
    ml_ctx = detect_iv_ml_anomalies(today_df)
    lv_spikes = pd.DataFrame()
    if today_d in study.local_vols and today_d in study.grids:
        g_dte, g_delta, _ = study.delta_grids.get(today_d, (None, None, None))
        if g_dte is not None:
            lv_spikes = detect_local_vol_spikes(study.local_vols[today_d], g_dte, g_delta)

    result = study.analyze()
    sentiment = result["sentiment"]
    vix = result["vix_context"]
    today_stats = delta_ctx.get("today", {})
    iv_chg = delta_ctx.get("iv_vs_5d_mean", np.nan)
    skew_chg = delta_ctx.get("put_spread_vs_5d_mean", np.nan)

    vol_level = (
        "higher than the recent range"
        if (np.isfinite(iv_chg) and iv_chg > 0.3)
        else ("lower than the recent range" if (np.isfinite(iv_chg) and iv_chg < -0.3) else "in line with the recent range")
    )
    skew_tilt = "puts are relatively richer vs calls" if (np.isfinite(skew_chg) and skew_chg > 0.2) else (
        "calls are relatively richer vs puts" if (np.isfinite(skew_chg) and skew_chg < -0.2) else "put/call balance is little changed"
    )

    bullets: list[str] = [
        f"As of {today_d}, SPX near {today_df['spot'].iloc[0]:,.0f}.",
    ]
    iv_line = (
        f"In the {delta_lo:+.2f} to {delta_hi:+.2f} delta band, mean IV is "
        f"{today_stats.get('mean_iv', np.nan):.1f}%"
    )
    if np.isfinite(iv_chg):
        iv_line += f" ({iv_chg:+.1f} vol pts vs the prior {len(dates) - 1} session average)."
    else:
        iv_line += "."
    bullets.append(f"Vol level: {iv_line}")

    if vix:
        bullets.append(
            f"VIX {vix.get('vix', np.nan):.1f}% ({vix.get('vix_change_5d', np.nan):+.1f} pts over 5d); "
            f"overall vol is {vol_level}."
        )
    else:
        bullets.append(f"Overall vol is {vol_level}.")

    bullets.append(
        f"Put/call skew: spread {today_stats.get('put_call_spread', np.nan):+.1f} vol pts in band; {skew_tilt}."
    )
    bullets.append(
        f"Sentiment �?fear: {sentiment.get('fear', '').replace(chr(8212), '-')}; "
        f"calls: {sentiment.get('call_opportunity', '').replace(chr(8212), '-')}; "
        f"event: {sentiment.get('event_risk', '').replace(chr(8212), '-')}."
    )

    n_ml = ml_ctx.get("n_anomalies", 0)
    n_lv = len(lv_spikes)
    if n_ml or n_lv:
        bullets.append(
            f"Local-vol diagnostics: {n_ml} ML IV outlier(s), {n_lv} local-vol spike(s) "
            f"�?check for stale quotes or event pockets."
        )
    else:
        bullets.append("Local-vol diagnostics: no extreme ML IV outliers or local-vol spikes.")

    draft = "\n".join(f"- {b}" for b in bullets)
    if use_deepseek if use_deepseek is not None else study.cfg.use_deepseek:
        return deepseek_study_conclusion(draft, delta_ctx, ml_ctx, lv_spikes, sentiment)
    return draft


def plot_linked_delta_surfaces(
    grid_dte: np.ndarray,
    grid_delta: np.ndarray,
    iv_grid: np.ndarray,
    local_vol_grid: np.ndarray,
    *,
    title: str = "",
    link_cameras: bool = True,
) -> Any:
    """Side-by-side 3D IV and local-vol surfaces (delta × DTE) with linked rotation."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required for 3D linked surfaces: pip install plotly") from exc

    x_delta = grid_delta[0, :]
    y_dte = grid_dte[:, 0]
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Implied vol (%)", "Dupire local vol (%)"),
        horizontal_spacing=0.06,
    )
    iv_lo, iv_hi = float(np.nanmin(iv_grid)), float(np.nanmax(iv_grid))
    lv_lo, lv_hi = float(np.nanmin(local_vol_grid)), float(np.nanmax(local_vol_grid))
    fig.add_trace(
        go.Surface(
            x=x_delta,
            y=y_dte,
            z=iv_grid,
            cmin=iv_lo,
            cmax=iv_hi,
            colorscale="Viridis",
            showscale=True,
            name="IV",
            colorbar=dict(
                title=dict(text="IV (%)"),
                len=0.55,
                thickness=16,
                x=0.47,
                xanchor="left",
                y=0.5,
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=x_delta,
            y=y_dte,
            z=local_vol_grid,
            cmin=lv_lo,
            cmax=lv_hi,
            colorscale="Magma",
            showscale=True,
            name="Local vol",
            colorbar=dict(
                title=dict(text="Local vol (%)"),
                len=0.55,
                thickness=16,
                x=1.02,
                xanchor="left",
                y=0.5,
            ),
        ),
        row=1,
        col=2,
    )
    layout = dict(
        title=title or "IV vs local vol (linked 3D)",
        height=620,
        width=1320,
        scene=dict(
            xaxis_title="Delta",
            yaxis_title="DTE",
            zaxis_title="IV (%)",
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1, z=0.6),
        ),
        scene2=dict(
            xaxis_title="Delta",
            yaxis_title="DTE",
            zaxis_title="Local vol (%)",
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1, z=0.6),
        ),
    )
    fig.update_layout(**layout)

    if link_cameras:
        try:
            import ipywidgets  # noqa: F401
            fig_w = go.FigureWidget(fig)
            fig_w.layout.on_change(
                lambda layout, key: setattr(fig_w.layout.scene2, "camera", layout.scene.camera),
                "scene.camera",
            )
            return fig_w
        except Exception:
            logger.debug("FigureWidget camera sync unavailable; returning static Figure")
    return fig


def plot_single_delta_surface(
    grid_dte: np.ndarray,
    grid_delta: np.ndarray,
    z_grid: np.ndarray,
    *,
    local: bool = False,
    title: str = "",
) -> Any:
    """One 3D surface on delta × DTE �?IV or Dupire local vol."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for 3D surfaces: pip install plotly") from exc

    x_delta = grid_delta[0, :]
    y_dte = grid_dte[:, 0]
    z_lo, z_hi = float(np.nanmin(z_grid)), float(np.nanmax(z_grid))
    if not np.isfinite(z_lo) or not np.isfinite(z_hi):
        raise ValueError("Surface grid has no finite values to plot.")
    if local:
        z_label = "Local vol (%)"
        colorscale = "Magma"
        trace_name = "Local vol"
    else:
        z_label = "IV (%)"
        colorscale = "Viridis"
        trace_name = "IV"

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_delta,
                y=y_dte,
                z=z_grid,
                cmin=z_lo,
                cmax=z_hi,
                colorscale=colorscale,
                showscale=True,
                name=trace_name,
                colorbar=dict(title=dict(text=z_label), len=0.65, thickness=18),
            )
        ]
    )
    fig.update_layout(
        title=title or trace_name,
        height=580,
        width=820,
        scene=dict(
            xaxis_title="Delta",
            yaxis_title="DTE",
            zaxis_title=z_label,
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1, z=0.6),
        ),
    )
    try:
        import ipywidgets  # noqa: F401
        return go.FigureWidget(fig)
    except Exception:
        return fig


def plot_surface_decomposition(
    grid_dte: np.ndarray,
    grid_delta: np.ndarray,
    iv_today: np.ndarray,
    iv_prev: np.ndarray,
    *,
    today_label: str,
    prev_label: str,
    title: str = "IV surface change (today �?previous session)",
) -> Any:
    """3D decomposition: IV change from previous business day to today."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required: pip install plotly") from exc

    diff = iv_today - iv_prev
    x_delta = grid_delta[0, :]
    y_dte = grid_dte[:, 0]
    fig = go.Figure(
        data=[
            go.Surface(
                x=x_delta,
                y=y_dte,
                z=diff,
                colorscale="RdBu_r",
                colorbar=dict(title="Δ IV (pts)"),
            )
        ]
    )
    fig.update_layout(
        title=f"{title}\n{today_label} minus {prev_label}",
        scene=dict(xaxis_title="Delta", yaxis_title="DTE", zaxis_title="Δ IV (vol pts)"),
        height=560,
        width=900,
    )
    return fig


def detect_term_hump(
    features: dict[str, Any],
    cfg: VolSurfaceConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or VolSurfaceConfig()
    dtes = features.get("term_dtes") or list(cfg.term_dtes)
    curve = features.get("term_curve") or [features.get(f"atm_iv_{d}d", np.nan) for d in dtes]
    curve = np.asarray(curve, dtype=float)
    valid = np.isfinite(curve)
    if valid.sum() < 3:
        return {"event_hump": False, "hump_dte": np.nan, "hump_iv": np.nan, "hump_z": np.nan}

    dtes_arr = np.asarray(dtes)[valid]
    curve = curve[valid]
    # Local maxima excluding edges
    peaks: list[tuple[int, float, float]] = []
    for i in range(1, len(curve) - 1):
        if curve[i] > curve[i - 1] and curve[i] > curve[i + 1]:
            peaks.append((i, float(dtes_arr[i]), float(curve[i])))

    if not peaks:
        return {"event_hump": False, "hump_dte": np.nan, "hump_iv": np.nan, "hump_z": np.nan}

    lo, hi = cfg.event_hump_dte_range
    in_window = [(i, d, v) for i, d, v in peaks if lo <= d <= hi]
    if not in_window:
        return {"event_hump": False, "hump_dte": np.nan, "hump_iv": np.nan, "hump_z": np.nan}

    idx, hump_dte, hump_iv = max(in_window, key=lambda x: x[2])
    baseline = np.mean(np.delete(curve, idx))
    spread = np.std(curve) or 1.0
    hump_z = (hump_iv - baseline) / spread
    return {
        "event_hump": bool(hump_z >= 1.0 and hump_iv - baseline >= 1.5),
        "hump_dte": hump_dte,
        "hump_iv": hump_iv,
        "hump_z": float(hump_z),
        "hump_baseline_iv": float(baseline),
    }


def compare_features(
    today: dict[str, Any],
    history: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    keys = [
        "atm_iv_30d",
        "skew_25d",
        "call_skew_25d",
        "butterfly_25d",
        "put_slope_10d",
        "call_slope_10d",
        "term_slope",
        "term_curvature",
    ]
    for key in keys:
        row: dict[str, Any] = {"feature": key, "today": today.get(key, np.nan)}
        for h in history:
            d = h.get("asof_date", "?")
            row[f"delta_{d}"] = today.get(key, np.nan) - h.get(key, np.nan) if key in h else np.nan
        if history:
            oldest = history[0]
            row["delta_5d"] = today.get(key, np.nan) - oldest.get(key, np.nan)
            row["delta_1d"] = today.get(key, np.nan) - history[-1].get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _yf_close(ticker: str, period: str = "3mo") -> pd.Series:
    if yf is None:
        raise ImportError("yfinance is required")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty or "Close" not in raw:
        raise RuntimeError(f"No yfinance data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().astype(float)


def fetch_vix_context(lookback_days: int = 5, trading_days: int = TRADING_DAYS) -> dict[str, Any]:
    """VIX / realized-vol context for multi-day IV level comparison when surface cache is sparse."""
    vix = _yf_close(VIX_TICKER)
    spx = _yf_close(SPX_TICKER)
    df = pd.concat({"VIX": vix, "SPX": spx}, axis=1, join="inner").dropna()
    df["log_ret"] = np.log(df["SPX"] / df["SPX"].shift(1))
    df["RV_22"] = df["log_ret"].rolling(22).std() * np.sqrt(trading_days) * 100

    latest = df.iloc[-1]
    idx_1d = max(len(df) - 2, 0)
    idx_5d = max(len(df) - 1 - lookback_days, 0)
    row_1d = df.iloc[idx_1d]
    row_5d = df.iloc[idx_5d]

    vix_today = float(latest["VIX"])
    rv_22 = float(latest["RV_22"]) if np.isfinite(latest["RV_22"]) else np.nan
    vrp = vix_today - rv_22 if np.isfinite(rv_22) else np.nan

    return {
        "asof": str(df.index[-1].date()),
        "vix": vix_today,
        "vix_1d_ago": float(row_1d["VIX"]),
        "vix_5d_ago": float(row_5d["VIX"]),
        "vix_change_1d": vix_today - float(row_1d["VIX"]),
        "vix_change_5d": vix_today - float(row_5d["VIX"]),
        "rv_22": rv_22,
        "vrp": vrp,
        "spx": float(latest["SPX"]),
        "history": df.tail(lookback_days + 1)[["VIX", "RV_22"]].reset_index(),
    }


def load_deepseek_api_key(env_path: Path | None = None) -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("deepseek")
    if key:
        return key.strip()
    path = env_path or Path(".env")
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("deepseek"):
            _, _, val = line.partition("=")
            return val.strip().strip('"').strip("'")
    return None


def deepseek_enhance_commentary(
    rule_commentary: str,
    features: dict[str, Any],
    vix_ctx: dict[str, Any],
    anomalies: dict[str, Any],
    sentiment: dict[str, str],
    api_key: str | None = None,
) -> str:
    """Optional LLM polish on top of rule-based commentary."""
    api_key = api_key or load_deepseek_api_key()
    if not api_key:
        return rule_commentary

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an equity derivatives strategist. Refine the rule-based draft "
                    "using only the supplied metrics. Do not invent data. No price targets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft:\n{rule_commentary}\n\n"
                    f"Features: {json.dumps({k: features[k] for k in features if k not in ('term_dtes', 'term_curve')}, default=str)}\n"
                    f"VIX context: {json.dumps({k: v for k, v in vix_ctx.items() if k != 'history'}, default=str)}\n"
                    f"Anomalies: {json.dumps(anomalies, default=str)}\n"
                    f"Sentiment: {json.dumps(sentiment)}"
                ),
            },
        ],
        "temperature": 0.3,
    }
    req = request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except (error.URLError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("DeepSeek commentary failed: %s", exc)
        return rule_commentary


def classify_sentiment(
    today: dict[str, Any],
    changes: pd.DataFrame,
    anomalies: dict[str, Any],
    vix_ctx: dict[str, Any] | None = None,
    anchor_ctx: dict[str, Any] | None = None,
) -> dict[str, str]:
    def _delta(name: str, default: float = 0.0) -> float:
        row = changes.loc[changes["feature"] == name]
        if row.empty or "delta_5d" not in row.columns:
            return default
        val = row["delta_5d"].iloc[0]
        return float(val) if np.isfinite(val) else default

    d_skew = _delta("skew_25d")
    d_call = _delta("call_skew_25d")
    d_atm = _delta("atm_iv_30d")
    d_put_slope = _delta("put_slope_10d")
    d_bfly = _delta("butterfly_25d")

    anchor_changes = (anchor_ctx or {}).get("changes") or {}
    if anchor_changes.get("atm_30d_call") is not None and not np.isfinite(d_atm):
        d_atm = float(anchor_changes["atm_30d_call"])
    elif vix_ctx and not np.isfinite(d_atm):
        d_atm = float(vix_ctx.get("vix_change_5d", 0.0))
    elif vix_ctx and len(changes.columns) <= 4:
        d_atm = float(vix_ctx.get("vix_change_5d", d_atm))

    skew_proxy_chg = anchor_changes.get("skew_proxy")
    if skew_proxy_chg is not None and np.isfinite(skew_proxy_chg):
        if not np.isfinite(d_skew) or d_skew == 0.0:
            d_skew = float(skew_proxy_chg)

    fear_score = 0
    if d_atm > 1.0:
        fear_score += 1
    if d_skew > 0.5:
        fear_score += 2
    if d_put_slope > 0.5:
        fear_score += 1
    if d_bfly > 0.3:
        fear_score += 1

    call_opportunity_score = 0
    if d_call > 0.3:
        call_opportunity_score += 1
    if d_atm < -1.0:
        call_opportunity_score += 1
    if today.get("call_skew_25d", 0) > today.get("skew_25d", 0) * -0.3:
        call_opportunity_score += 1

    if fear_score >= 3:
        fear_label = "elevated �?downside protection being bid"
    elif fear_score >= 1:
        fear_label = "moderate �?some hedging demand"
    else:
        fear_label = "calm �?limited crash-insurance premium"

    if call_opportunity_score >= 2 and fear_score <= 1:
        call_label = "favorable �?upside vol relatively cheap vs puts"
    elif call_opportunity_score >= 1:
        call_label = "mixed �?selective call structures may work"
    else:
        call_label = "cautious �?calls not clearly cheap vs hedging demand"

    if anomalies.get("event_hump"):
        event_label = f"event priced near {anomalies['hump_dte']:.0f}d (IV hump)"
    elif today.get("term_slope", 0) > 2:
        event_label = "front-end IV elevated �?near-term stress"
    elif today.get("term_slope", 0) < -2:
        event_label = "inverted term structure �?immediate concern"
    else:
        event_label = "no dominant event hump in term structure"

    return {
        "fear": fear_label,
        "call_opportunity": call_label,
        "event_risk": event_label,
        "fear_score": str(fear_score),
        "call_score": str(call_opportunity_score),
    }


def _change_val(changes: pd.DataFrame, feature: str, col: str = "delta_5d") -> float | None:
    if col not in changes.columns or feature not in changes["feature"].values:
        return None
    val = changes.loc[changes["feature"] == feature, col].iloc[0]
    return float(val) if np.isfinite(val) else None


def _pick_nearest_dte_slice(df: pd.DataFrame, target_dte: float, option_type: str) -> pd.DataFrame:
    sub = df[df["option_type"].str.upper().str.contains(option_type.upper())].copy()
    if sub.empty:
        return sub
    nearest = sub["dte"].iloc[(sub["dte"] - target_dte).abs().argsort().iloc[0]]
    return sub[sub["dte"] == nearest]


def pick_anchor_contracts(df: pd.DataFrame, cfg: VolSurfaceConfig | None = None) -> dict[str, pd.Series]:
    """Select representative option codes for Futu IV history (today's chain)."""
    cfg = cfg or VolSurfaceConfig()
    anchors: dict[str, pd.Series] = {}

    for dte in cfg.anchor_dtes:
        slice_df = _pick_nearest_dte_slice(df, float(dte), "CALL")
        if slice_df.empty:
            continue
        row = slice_df.loc[slice_df["log_moneyness"].abs().idxmin()]
        anchors[f"atm_{int(dte)}d_call"] = row

    put_slice = _pick_nearest_dte_slice(df, 30.0, "PUT")
    if not put_slice.empty and put_slice["delta"].notna().any():
        target = -abs(cfg.anchor_delta)
        row = put_slice.iloc[(put_slice["delta"] - target).abs().argsort().iloc[0]]
        anchors["put_25d_30d"] = row

    call_slice = _pick_nearest_dte_slice(df, 30.0, "CALL")
    if not call_slice.empty and call_slice["delta"].notna().any():
        target = abs(cfg.anchor_delta)
        row = call_slice.iloc[(call_slice["delta"] - target).abs().argsort().iloc[0]]
        anchors["call_25d_30d"] = row

    return anchors


def fetch_anchor_iv_histories(
    df: pd.DataFrame,
    cfg: VolSurfaceConfig | None = None,
    query_time_period: int = 1,
) -> dict[str, Any]:
    """Daily IV history for anchor contracts via Futu get_option_volatility (1=Week)."""
    cfg = cfg or VolSurfaceConfig()
    _require_futu()
    if not futu_anchor_iv_supported():
        ver = getattr(ft, "__version__", "unknown")
        warnings.warn(
            f"futu-api {ver} has no get_option_volatility �?anchor IV history skipped. "
            "Upgrade in your notebook kernel: pip install -U futu-api",
            stacklevel=2,
        )
        return {
            "anchors": {},
            "series": pd.DataFrame(),
            "changes": {},
            "meta": {"supported": False, "futu_version": ver},
        }

    anchors = pick_anchor_contracts(df, cfg)
    if not anchors:
        return {"anchors": {}, "series": pd.DataFrame(), "changes": {}, "meta": {}}

    ctx = ft.OpenQuoteContext(host=cfg.host, port=cfg.port)
    anchor_meta: dict[str, dict[str, Any]] = {}
    series_parts: list[pd.DataFrame] = []
    try:
        for name, row in anchors.items():
            code = str(row["code"])
            if code.startswith("DEMO."):
                logger.warning("Skipping demo anchor %s", name)
                continue
            time.sleep(SNAPSHOT_PAUSE_SEC)
            ret, vol = _futu_get_option_volatility(
                ctx,
                code,
                query_time_period=query_time_period,
                hv_time_period=cfg.anchor_hv_period,
            )
            if ret != ft.RET_OK:
                logger.warning("Anchor IV history failed for %s (%s): %s", name, code, vol)
                continue
            if not isinstance(vol, pd.DataFrame) or vol.empty:
                logger.warning("Anchor IV history failed for %s (%s): %s", name, code, vol)
                continue

            anchor_meta[name] = {
                "code": code,
                "strike": float(row["strike"]),
                "dte": int(row["dte"]),
                "iv_today": float(row["iv"]),
                "option_type": str(row.get("option_type", "")),
            }
            part = vol[["timestamp_str", "implied_volatility", "history_volatility", "volatility_premium"]].copy()
            part = part.rename(
                columns={
                    "timestamp_str": "date",
                    "implied_volatility": name,
                    "history_volatility": f"{name}_hv",
                    "volatility_premium": f"{name}_premium",
                }
            )
            series_parts.append(part)
    finally:
        ctx.close()

    if not series_parts:
        return {"anchors": anchor_meta, "series": pd.DataFrame(), "changes": {}, "meta": {}}

    merged = series_parts[0]
    for part in series_parts[1:]:
        merged = merged.merge(part, on="date", how="outer")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)

    if "put_25d_30d" in merged.columns and "atm_30d_call" in merged.columns:
        merged["skew_proxy"] = merged["put_25d_30d"] - merged["atm_30d_call"]

    changes: dict[str, float] = {}
    iv_cols = [
        c
        for c in merged.columns
        if c not in ("date",) and not c.endswith("_hv") and not c.endswith("_premium")
    ]
    for col in iv_cols:
        s = merged[col].dropna()
        if len(s) >= 2:
            changes[col] = float(s.iloc[-1] - s.iloc[0])
        elif len(s) == 1:
            changes[col] = 0.0

    return {
        "anchors": anchor_meta,
        "series": merged,
        "changes": changes,
        "meta": {"query_days": len(merged), "query_time_period": query_time_period},
    }


def format_anchor_commentary(anchor_ctx: dict[str, Any]) -> list[str]:
    if anchor_ctx.get("meta", {}).get("supported") is False:
        ver = anchor_ctx.get("meta", {}).get("futu_version", "?")
        return [
            "## Anchor IV history",
            f"Unavailable: futu-api {ver} is too old (need >= 10.07). "
            "Run in notebook kernel: `pip install -U futu-api` then restart kernel.",
        ]
    if anchor_ctx.get("series") is None or anchor_ctx["series"].empty:
        return ["Anchor IV history unavailable (Futu get_option_volatility returned no data)."]

    lines = ["## Anchor IV history (Futu, ~1 week)"]
    for name, meta in anchor_ctx.get("anchors", {}).items():
        chg = anchor_ctx.get("changes", {}).get(name, np.nan)
        chg_s = f"{chg:+.1f} vol pts over window" if np.isfinite(chg) else "n/a"
        lines.append(
            f"- **{name}** `{meta['code']}` strike {meta['strike']:,.0f} "
            f"(DTE {meta['dte']}d): IV {meta['iv_today']:.1f}%, {chg_s}"
        )
    skew_chg = anchor_ctx.get("changes", {}).get("skew_proxy", np.nan)
    if np.isfinite(skew_chg):
        if skew_chg > 0.3:
            direction = "steeper downside"
        elif skew_chg < -0.3:
            direction = "flatter downside"
        else:
            direction = "stable skew"
        lines.append(
            f"- **Skew proxy** (25Δ put �?30d ATM call): window change {skew_chg:+.1f} vol pts �?{direction}"
        )
    lines.append("Note: each series tracks one fixed contract; DTE/moneyness drift over the window.")
    return lines


def _series_daily_deltas(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.array([])
    return np.diff(s.to_numpy(dtype=float))


def _consecutive_streak(deltas: np.ndarray) -> tuple[int, str]:
    if len(deltas) == 0:
        return 0, "flat"
    sign = float(np.sign(deltas[-1]))
    if sign == 0:
        return 0, "flat"
    streak = 1
    for i in range(len(deltas) - 2, -1, -1):
        if np.sign(deltas[i]) == sign:
            streak += 1
        else:
            break
    return streak, "up" if sign > 0 else "down"


def _today_vs_recent(deltas: np.ndarray) -> tuple[str, float]:
    """Classify whether the latest daily move is typical or an outlier."""
    if len(deltas) < 2:
        return "insufficient history", 0.0
    today = float(deltas[-1])
    prior = deltas[:-1]
    if len(prior) == 0:
        return "insufficient history", 0.0
    mean = float(np.mean(prior))
    std = float(np.std(prior)) or max(abs(mean), 0.05)
    z = (today - mean) / std
    if abs(z) >= 2.0:
        tag = "today is an outlier vs the recent rhythm"
    elif abs(z) >= 1.2:
        tag = "today is somewhat sharper than recent days"
    elif abs(today) < 0.05 and abs(mean) < 0.05:
        tag = "surface has been quiet with little day-to-day change"
    else:
        tag = "today continues the recent day-to-day pattern"
    return tag, z


def _pc_intensity_label(z: float, *, pos_label: str, neg_label: str, mild_threshold: float = 1.0) -> str:
    if z >= mild_threshold:
        return pos_label
    if z <= -mild_threshold:
        return neg_label
    return "little net tilt in this mode today"


def build_structure_metrics_insights(
    today: dict[str, Any],
    vix_ctx: dict[str, Any] | None,
    anchor_ctx: dict[str, Any] | None,
    today_scores: list[float] | np.ndarray,
    pca_sentiment: dict[str, Any] | None,
    pca_score_history: np.ndarray | None = None,
    changes: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    """Rule-based metric lines + natural-language insights for the structure panel."""
    vix_ctx = vix_ctx or {}
    anchor_ctx = anchor_ctx or {}
    scores = np.asarray(today_scores, dtype=float) if today_scores is not None else np.zeros(4)
    pc1 = float(scores[0]) if len(scores) > 0 else 0.0
    pc2 = float(scores[1]) if len(scores) > 1 else 0.0
    pc3 = float(scores[2]) if len(scores) > 2 else 0.0

    aiv = float(today.get("atm_iv_30d", np.nan))
    vix = float(vix_ctx.get("vix", np.nan))
    vrp = float(vix_ctx.get("vrp", np.nan)) if np.isfinite(vix_ctx.get("vrp", np.nan)) else (
        vix - aiv if np.isfinite(vix) and np.isfinite(aiv) else np.nan
    )
    vix_chg_5d = float(vix_ctx.get("vix_change_5d", np.nan))
    psk = float(today.get("skew_25d", np.nan))
    call_sk = float(today.get("call_skew_25d", np.nan))
    bfly = float(today.get("butterfly_25d", np.nan))
    tsl = float(today.get("term_slope", np.nan))

    def _delta5(name: str) -> float:
        if changes is None or changes.empty or "delta_5d" not in changes.columns:
            return 0.0
        row = changes.loc[changes["feature"] == name]
        if row.empty:
            return 0.0
        val = row["delta_5d"].iloc[0]
        return float(val) if np.isfinite(val) else 0.0

    d_skew = _delta5("skew_25d")
    d_bfly = _delta5("butterfly_25d")
    d_call = _delta5("call_skew_25d")
    d_atm = _delta5("atm_iv_30d")

    anchor_series = anchor_ctx.get("series")
    anchor_changes = anchor_ctx.get("changes") or {}
    atm_deltas = np.array([])
    skew_deltas = np.array([])
    call_deltas = np.array([])
    if isinstance(anchor_series, pd.DataFrame) and not anchor_series.empty:
        if "atm_30d_call" in anchor_series.columns:
            atm_deltas = _series_daily_deltas(anchor_series["atm_30d_call"])
        if "skew_proxy" in anchor_series.columns:
            skew_deltas = _series_daily_deltas(anchor_series["skew_proxy"])
        elif "put_25d_30d" in anchor_series.columns and "atm_30d_call" in anchor_series.columns:
            skew_deltas = _series_daily_deltas(
                anchor_series["put_25d_30d"] - anchor_series["atm_30d_call"]
            )
        if "call_25d_30d" in anchor_series.columns and "atm_30d_call" in anchor_series.columns:
            call_deltas = _series_daily_deltas(
                anchor_series["call_25d_30d"] - anchor_series["atm_30d_call"]
            )

    # --- Vol level ---
    vol_metric = f"Implied Vol (ATM 30d): {aiv:.1f}% | VIX Base level: {vix:.1f}%"
    vol_parts: list[str] = []
    if np.isfinite(vrp):
        if vrp > 2:
            vol_parts.append(f"VIX trades {vrp:.1f} vol pts above 22d realized — options embed a fear premium.")
        elif vrp < -1:
            vol_parts.append(f"ATM IV is {abs(vrp):.1f} pts above VIX — front-end implied vol looks rich vs the fear gauge.")
        else:
            vol_parts.append("ATM IV and VIX are roughly aligned — no extreme vol-risk premium dislocation.")
    if np.isfinite(vix_chg_5d):
        if vix_chg_5d > 1.5:
            vol_parts.append(f"VIX has drifted up {vix_chg_5d:+.1f} pts over 5 sessions — vol level is creeping higher.")
        elif vix_chg_5d < -1.5:
            vol_parts.append(f"VIX has eased {vix_chg_5d:+.1f} pts over 5 sessions — the macro vol backdrop is softening.")
    if len(atm_deltas) >= 2:
        streak, direction = _consecutive_streak(atm_deltas)
        rhythm, _ = _today_vs_recent(atm_deltas)
        if streak >= 3:
            vol_parts.append(
                f"ATM 30d IV has moved {direction} for {streak} straight anchor sessions — a persistent level shift, not a one-day blip."
            )
        else:
            vol_parts.append(f"Anchor ATM IV: {rhythm}.")
    elif np.isfinite(d_atm) and abs(d_atm) > 0.3:
        vol_parts.append(f"30d ATM IV is {d_atm:+.1f} vol pts vs the 5d reference — level is migrating.")
    if not vol_parts:
        vol_parts.append("Vol level looks stable; no strong multi-day drift in ATM IV or VIX.")

    # --- Skew ---
    skew_metric = (
        f"Skew Steepness Premium: 25d option slope is {psk:.1f} vol pts (5d Change: {d_skew:.1f} pts)"
    )
    skew_parts: list[str] = []
    if psk > 3:
        skew_parts.append("Put wing is materially steeper than ATM — downside hedging is bid.")
    elif psk < 0:
        skew_parts.append("Skew is inverted/flat — puts are not commanding a large premium vs ATM.")
    if np.isfinite(call_sk):
        if call_sk > 1.5 and d_call > 0.5:
            skew_parts.append(
                f"25Δ call IV is elevated ({call_sk:+.1f} vol pts vs ATM) and rising — upside/convexity demand is lifting call wing IV."
            )
        elif call_sk > 1.5:
            skew_parts.append(
                f"Call wing IV sits {call_sk:+.1f} vol pts above ATM — upside strikes are relatively bid."
            )
        elif d_call < -0.5:
            skew_parts.append("Call skew has compressed over 5d — upside hedges are cheaper vs puts.")
    if len(skew_deltas) >= 2:
        streak, direction = _consecutive_streak(skew_deltas)
        rhythm, z = _today_vs_recent(skew_deltas)
        if streak >= 3:
            skew_parts.append(
                f"Skew proxy has {'steepened' if direction == 'up' else 'flattened'} for {streak} consecutive sessions — a sustained tilt in the put/call balance."
            )
        elif abs(z) >= 1.5:
            skew_parts.append(f"Today's skew move looks unusual ({rhythm}).")
        elif abs(d_skew) > 0.5:
            skew_parts.append(f"5d skew change of {d_skew:+.1f} pts confirms the wing repricing.")
    elif abs(d_skew) > 0.3:
        skew_parts.append(f"Skew has shifted {d_skew:+.1f} vol pts over 5d — watch whether puts or calls are driving the wing move.")
    if not skew_parts:
        skew_parts.append("Skew shape is near recent norms — no dominant put- or call-wing shock.")

    # --- Butterfly ---
    bfly_metric = (
        f"Wings Fly Curvature: 25d butterfly represents {bfly:.1f} vol pts (5d Change: {d_bfly:.1f} pts)"
    )
    bfly_parts: list[str] = []
    if bfly > 1.5:
        bfly_parts.append("Wings are lifted vs belly — tail/convexity is being priced in (smile wings rich).")
    elif bfly < -1.0:
        bfly_parts.append("Butterfly is negative — wings trade cheap vs ATM; the smile is tight in the tails.")
    if abs(d_bfly) > 0.5:
        bfly_parts.append(f"5d butterfly change ({d_bfly:+.1f} pts) shows wings {'richening' if d_bfly > 0 else 'cheapening'} vs the body.")
    if not bfly_parts:
        bfly_parts.append("Wing curvature is muted — tail pricing is not the main story today.")

    # --- PCA ---
    pca_metric = f"PCA Delta Systemic Shocks: PC1 (Shift) = {pc1:.2f} | PC2 (Skew Tilt) = {pc2:.2f}"
    pca_parts: list[str] = []
    z_scores = (pca_sentiment or {}).get("z_scores") or []
    z1 = float(z_scores[0]) if len(z_scores) > 0 else pc1
    z2 = float(z_scores[1]) if len(z_scores) > 1 else pc2
    z3 = float(z_scores[2]) if len(z_scores) > 2 else pc3

    pca_parts.append(
        _pc_intensity_label(
            z1,
            pos_label="PC1: broad parallel shift UP — the whole surface repriced higher across strikes/tenors (vol level shock).",
            neg_label="PC1: broad parallel shift DOWN — systemic vol compression across the surface.",
        )
    )
    pca_parts.append(
        _pc_intensity_label(
            z2,
            pos_label="PC2: skew steepening — put-side IV rising faster than calls; downside protection demand dominates.",
            neg_label="PC2: skew flattening — call wing IV bid up relative to puts; upside/convexity catching a bid.",
            mild_threshold=0.8,
        )
    )
    if abs(z3) >= 0.8:
        pca_parts.append(
            _pc_intensity_label(
                z3,
                pos_label="PC3: wing/tail curvature expanding — far OTM options repricing faster (tail risk bid).",
                neg_label="PC3: wing curvature compressing — tails cheapening vs the belly.",
                mild_threshold=0.8,
            )
        )

    if pca_score_history is not None and len(pca_score_history) >= 3:
        pc1_hist = pca_score_history[:, 0]
        pc1_d = np.diff(pc1_hist)
        streak, direction = _consecutive_streak(pc1_d)
        rhythm, z = _today_vs_recent(pc1_d)
        if streak >= 3:
            pca_parts.append(
                f"PC1 has moved the same direction for {streak} sessions — surface level shifts are stacking, not reversing."
            )
        elif abs(z) >= 1.5:
            pca_parts.append(f"Today's PC1 move breaks the recent pattern ({rhythm}).")
    elif len(atm_deltas) >= 2:
        streak, _ = _consecutive_streak(atm_deltas)
        rhythm, z = _today_vs_recent(atm_deltas)
        if streak >= 3:
            pca_parts.append(
                f"Without full surface history, anchor ATM IV shows {streak} days of same-direction level drift — likely a sustained PC1-style shift."
            )
        elif abs(z) >= 1.5:
            pca_parts.append(f"Today's ATM IV change stands out vs the past week ({rhythm}).")

    if len(call_deltas) >= 2:
        streak, direction = _consecutive_streak(call_deltas)
        if streak >= 3 and direction == "up":
            pca_parts.append(
                f"Call-wing IV vs ATM has risen for {streak} straight sessions — consistent with PC2 call-skew bid."
            )

    if np.isfinite(tsl) and abs(tsl) > 2:
        pca_parts.append(
            f"Term slope {tsl:+.1f} vol pts — "
            + ("front-end IV elevated vs back (event/near-term fear)." if tsl > 0 else "back-end IV holds up vs front (longer-dated uncertainty).")
        )

    return [
        {"key": "vol_level", "metric": vol_metric, "insight": " ".join(vol_parts)},
        {"key": "skew", "metric": skew_metric, "insight": " ".join(skew_parts)},
        {"key": "butterfly", "metric": bfly_metric, "insight": " ".join(bfly_parts)},
        {"key": "pca", "metric": pca_metric, "insight": " ".join(pca_parts)},
    ]


def deepseek_enhance_structure_insights(
    metrics: list[dict[str, str]],
    context: dict[str, Any],
    api_key: str | None = None,
) -> list[dict[str, str]]:
    """Rewrite insight sentences with DeepSeek; falls back to rule-based text."""
    api_key = api_key or load_deepseek_api_key()
    if not api_key:
        return metrics

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an equity index options strategist writing dashboard copy. "
                    "For each metric block, rewrite ONLY the insight field: 1-2 crisp sentences, "
                    "trader-facing, no bullet lists, no invented numbers, no price targets. "
                    "Emphasize whether moves are multi-day streaks vs today's outlier, and PCA surface dynamics "
                    "(level shift, skew tilt, call vs put wing). Return valid JSON array of "
                    "objects with keys: key, metric, insight. Keep metric strings unchanged."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"blocks": metrics, "context": context}, default=str),
            },
        ],
        "temperature": 0.35,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        rows = parsed if isinstance(parsed, list) else parsed.get("blocks") or parsed.get("metrics") or []
        if not isinstance(rows, list) or not rows:
            return metrics
        by_key = {str(r.get("key")): r for r in rows if isinstance(r, dict) and r.get("key")}
        out: list[dict[str, str]] = []
        for block in metrics:
            key = block["key"]
            if key in by_key and by_key[key].get("insight"):
                out.append({
                    "key": key,
                    "metric": block["metric"],
                    "insight": str(by_key[key]["insight"]).strip(),
                })
            else:
                out.append(block)
        return out
    except (error.URLError, KeyError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("DeepSeek structure insights failed: %s", exc)
        return metrics


def build_commentary(
    today: dict[str, Any],
    changes: pd.DataFrame,
    anomalies: dict[str, Any],
    sentiment: dict[str, str],
    vix_ctx: dict[str, Any] | None = None,
    surface_days: int = 1,
    anchor_ctx: dict[str, Any] | None = None,
) -> str:
    skew_chg = _change_val(changes, "skew_25d")
    if skew_chg is None and anchor_ctx:
        proxy = anchor_ctx.get("changes", {}).get("skew_proxy")
        if proxy is not None and np.isfinite(proxy):
            skew_chg = float(proxy)
    call_chg = _change_val(changes, "call_skew_25d")
    tail_line = (
        f"Butterfly {today.get('butterfly_25d', np.nan):+.1f} vol pts; "
        f"10Δ put wing slope {today.get('put_slope_10d', np.nan):+.1f}."
    )
    if skew_chg is not None:
        tail_line += f" 5d skew change {skew_chg:+.1f} vol pts."

    call_line = ""
    if call_chg is not None:
        call_line = (
            f"25Δ call skew {today.get('call_skew_25d', np.nan):+.1f} vol pts; "
            f"5d change {call_chg:+.1f} vol pts."
        )

    lines = [
        "## Executive Summary",
        (
            f"As of {today.get('asof_date')}, SPX {today.get('spot', 0):,.0f}, "
            f"30d ATM IV {today.get('atm_iv_30d', np.nan):.1f}%"
            + (f" (VIX {vix_ctx['vix']:.1f}%)" if vix_ctx else "")
            + f", 25Δ put skew {today.get('skew_25d', np.nan):+.1f} vol pts."
        ),
        "",
        "## Vol Level (VIX context)",
        (
            f"VIX {vix_ctx['vix']:.1f}% vs 5d ago {vix_ctx['vix_5d_ago']:.1f}% "
            f"({vix_ctx['vix_change_5d']:+.1f} pts). "
            f"22d RV {vix_ctx['rv_22']:.1f}%, VRP {vix_ctx['vrp']:+.1f} vol pts."
            if vix_ctx
            else "VIX context unavailable."
        ),
        *(
            [
                f"Note: only {surface_days} cached SPX surface day(s); surface-shape deltas use anchor IV history below."
            ]
            if surface_days < 2
            else []
        ),
        "",
        *(format_anchor_commentary(anchor_ctx) if anchor_ctx else []),
        "",
        "## Fear Gauge",
        sentiment["fear"],
        "",
        "## Tail Risk",
        tail_line,
        "",
        "## Event Risk",
        sentiment["event_risk"],
        *(
            [
                f"Detected hump: {anomalies['hump_iv']:.1f}% at ~{anomalies['hump_dte']:.0f}d "
                f"(z={anomalies['hump_z']:.1f} vs local term curve)."
            ]
            if anomalies.get("event_hump")
            else []
        ),
        "",
        "## Call vs Put Balance",
        sentiment["call_opportunity"],
        call_line,
        "",
        "## Term Structure",
        (
            f"Front {today.get('front_iv', np.nan):.1f}% vs back {today.get('back_iv', np.nan):.1f}%; "
            f"slope {today.get('term_slope', np.nan):+.1f} vol pts."
        ),
    ]
    return "\n".join(line for line in lines if line)


def generate_demo_surfaces(
    cfg: VolSurfaceConfig | None = None,
    n_days: int = 6,
    *,
    save: bool = True,
) -> dict[date, pd.DataFrame]:
    """Synthetic surfaces for offline runs when OpenD/cache is unavailable."""
    cfg = cfg or VolSurfaceConfig()
    yf_spot = SPOT_YF_DEFAULT.get(cfg.underlying)
    spot0 = 5400.0
    if yf_spot:
        try:
            spot0, _ = fetch_spot_yfinance(yf_spot)
        except Exception:
            spot0 = {"US..NDX": 21000.0, "US.DIA": 420.0}.get(cfg.underlying, 5400.0)
    strike_step = max(1.0, round(spot0 * 0.005, 2))
    strikes = np.arange(
        spot0 * cfg.moneyness_min,
        spot0 * cfg.moneyness_max + strike_step,
        strike_step,
    )
    history: dict[date, pd.DataFrame] = {}
    expiries = [7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]

    for day_offset in range(n_days - 1, -1, -1):
        asof = date.today() - timedelta(days=day_offset)
        spot = spot0 * (1.0 - 0.002 * (n_days - 1 - day_offset))
        rows: list[dict[str, Any]] = []
        fear_ramp = (n_days - 1 - day_offset) * 0.15
        hump_day = day_offset == 2

        for dte in expiries:
            t = dte / TRADING_DAYS
            base_iv = 16.0 + 3.0 * np.sqrt(t) + fear_ramp
            if hump_day and 8 <= dte <= 14:
                base_iv += 6.0 * np.exp(-0.5 * ((dte - 10) / 1.5) ** 2)
            for strike in strikes:
                mny = spot / strike
                log_m = np.log(strike / spot)
                skew = 8.0 + fear_ramp * 2.0
                smile = 2.5 * (log_m**2) * 100
                wing = skew * max(-log_m, 0) * 100 - 0.5 * max(log_m, 0) * 100
                iv = base_iv + smile + wing

                for opt_type, delta_sign in (("CALL", 1), ("PUT", -1)):
                    delta = delta_sign * norm.cdf((np.log(spot / strike) + 0.5 * (iv / 100) ** 2 * t) / ((iv / 100) * np.sqrt(t)))
                    rows.append(
                        {
                            "asof_date": pd.Timestamp(asof),
                            "code": f"DEMO.{strike}.{dte}.{opt_type[0]}",
                            "option_type": opt_type,
                            "strike_time": str(asof + timedelta(days=dte)),
                            "option_strike_price": strike,
                            "strike": strike,
                            "expiry": pd.Timestamp(asof + timedelta(days=dte)),
                            "dte": dte,
                            "spot": spot,
                            "moneyness": spot / strike if opt_type == "CALL" else strike / spot,
                           "log_moneyness": log_m,
                            "ks_ratio": strike / spot,
                           "iv": iv,
                            "option_implied_volatility": iv,
                            "delta": delta,
                            "option_delta": delta,
                            "oi": 500,
                            "option_open_interest": 500,
                            "volume": 50,
                            "bid": 1.0,
                            "ask": 1.2,
                            "bid_price": 1.0,
                            "ask_price": 1.2,
                            "option_valid": True,
                        }
                    )
        df = pd.DataFrame(rows)
        history[asof] = clean_option_chain(df, cfg)
        if save:
            save_surface(df, cfg, asof=asof)
    return history


class VolSurfaceStudy:
    """Fetch/cache IV surfaces and produce multi-day analysis."""

    def __init__(self, cfg: VolSurfaceConfig | None = None):
        self.cfg = cfg or VolSurfaceConfig()
        self.surfaces: dict[date, pd.DataFrame] = {}
        self.features: dict[date, dict[str, Any]] = {}
        self.local_vols: dict[date, np.ndarray] = {}
        self.grids: dict[date, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.delta_grids: dict[date, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._delta_plot_views: dict[tuple[bool, bool], dict[str, Any]] | None = None

    def _invalidate_plot_cache(self) -> None:
        self._delta_plot_views = None

    def _compute_surface(self, d: date, df: pd.DataFrame) -> None:
        self.surfaces[d] = df
        self.features[d] = compute_surface_features(df, self.cfg)
        spot = float(df["spot"].iloc[0])
        try:
            grid = build_iv_grid(df, max_dte=self.cfg.max_dte)
            self.grids[d] = grid
        except ValueError as exc:
            logger.warning("Moneyness IV grid skipped for %s: %s", d, exc)
        try:
            delta_grid = build_iv_grid_delta(df, max_dte=self.cfg.max_dte)
            self.delta_grids[d] = delta_grid
            g_dte, g_delta, iv_g = delta_grid
            self.local_vols[d] = dupire_local_vol_delta(
                spot, g_dte, g_delta, iv_g, r=self.cfg.risk_free_rate
            )
        except ValueError as exc:
            logger.warning("Delta/local vol skipped for %s: %s", d, exc)
            if d in self.grids:
                self.local_vols[d] = dupire_local_vol(spot, *self.grids[d], r=self.cfg.risk_free_rate)

    def load_history(self, use_demo_if_empty: bool = False) -> None:
        self._invalidate_plot_cache()
        self.surfaces = load_surface_history(self.cfg, self.cfg.lookback_days)
        if (not self.surfaces or len(self.surfaces) < 6) and use_demo_if_empty:
            warnings.warn(
                "No cached surfaces or too few sessions - loading demo SPX surfaces.",
                stacklevel=2,
            )
            self.surfaces = generate_demo_surfaces(self.cfg, n_days=self.cfg.lookback_days + 1)
        self.surfaces = trim_surfaces_to_sessions(
            self.surfaces, self.cfg.lookback_days + 1, business_days=True
        )
        self.features.clear()
        self.local_vols.clear()
        self.grids.clear()
        self.delta_grids.clear()
        for d, df in self.surfaces.items():
            self._compute_surface(d, df)

    def ensure_history(
        self,
        min_sessions: int = 6,
        *,
        fetch_live: bool = True,
        demo_backfill: bool = True,
    ) -> None:
        """Load cache, fetch live if empty, backfill with demo when history is too short."""
        self.load_history(use_demo_if_empty=False)

        if not self.surfaces and fetch_live:
            try:
                self.fetch_live()
            except Exception as exc:
                logger.warning("Live surface fetch failed for %s: %s", self.cfg.underlying, exc)
            self.load_history(use_demo_if_empty=False)

        if len(self.surfaces) < min_sessions and demo_backfill:
            real = dict(self.surfaces)
            demo = generate_demo_surfaces(
                self.cfg,
                n_days=self.cfg.lookback_days + 1,
                save=False,
            )
            merged = trim_surfaces_to_sessions(
                {**demo, **real},
                self.cfg.lookback_days + 1,
                business_days=True,
            )
            self._invalidate_plot_cache()
            self.surfaces = merged
            self.features.clear()
            self.local_vols.clear()
            self.grids.clear()
            self.delta_grids.clear()
            for d, df in self.surfaces.items():
                self._compute_surface(d, df)

        if not self.surfaces:
            raise RuntimeError(f"No surface history available for {self.cfg.underlying}")

    def fetch_live(self, *, save: bool = False) -> pd.DataFrame:
        """Fetch today's option chain from Futu; by default no disk cache."""
        self._invalidate_plot_cache()
        self.surfaces.clear()
        self.features.clear()
        self.local_vols.clear()
        self.grids.clear()
        self.delta_grids.clear()
        df = fetch_and_cache(self.cfg, save=save)
        asof = pd.Timestamp(df["asof_date"].iloc[0]).date()
        self._compute_surface(asof, df)
        return df

    def analyze(self) -> dict[str, Any]:
        if not self.features:
            raise RuntimeError("No surface loaded. Call fetch_live() first.")
        dates = sorted(self.features.keys())
        today_d = dates[-1]
        today = self.features[today_d]
        hist = [self.features[d] for d in dates[:-1]]
        changes = compare_features(today, hist[-5:] if len(hist) >= 5 else hist)
        anomalies = detect_term_hump(today, self.cfg)
        hump_history = {
            str(d): detect_term_hump(self.features[d], self.cfg) for d in dates[:-1]
        }
        vix_ctx = fetch_vix_context(self.cfg.lookback_days)
        today_df = self.surfaces[today_d]
        anchor_ctx = fetch_anchor_iv_histories(today_df, self.cfg, query_time_period=1)
        sentiment = classify_sentiment(today, changes, anomalies, vix_ctx, anchor_ctx)
        rule_commentary = build_commentary(
            today, changes, anomalies, sentiment, vix_ctx, len(dates), anchor_ctx
        )
        commentary = rule_commentary
        if self.cfg.use_deepseek:
            commentary = deepseek_enhance_commentary(
                rule_commentary, today, vix_ctx, anomalies, sentiment
            )
        return {
            "today": today,
            "changes": changes,
            "anomalies": anomalies,
            "hump_history": hump_history,
            "vix_context": vix_ctx,
            "anchor_context": anchor_ctx,
            "sentiment": sentiment,
            "commentary": commentary,
            "rule_commentary": rule_commentary,
            "dates": dates,
        }

    def plot_term_structure(self, ax: plt.Axes | None = None) -> plt.Axes:
        if not self.features:
            self.load_history()
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))
        dates = sorted(self.features.keys())
        for d in dates:
            f = self.features[d]
            dtes = f.get("term_dtes") or list(self.cfg.term_dtes)
            curve = f.get("term_curve") or [f.get(f"atm_iv_{x}d", np.nan) for x in dtes]
            label = pd.Timestamp(d).strftime("%Y-%m-%d")
            lw = 2.5 if d == dates[-1] else 1.2
            ax.plot(dtes, curve, marker="o", label=label, linewidth=lw)
        ax.set_xlabel("Days to expiry")
        ax.set_ylabel("ATM IV (%)")
        ax.set_title(f"{self.cfg.underlying} �?ATM term structure")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_skew_evolution(self, ax: plt.Axes | None = None) -> plt.Axes:
        if not self.features:
            self.load_history()
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        dates = sorted(self.features.keys())
        skew = [self.features[d].get("skew_25d", np.nan) for d in dates]
        call_sk = [self.features[d].get("call_skew_25d", np.nan) for d in dates]
        x = [pd.Timestamp(d) for d in dates]
        ax.plot(x, skew, marker="o", label="25Δ put skew (vs ATM)")
        ax.plot(x, call_sk, marker="s", label="25Δ call skew (vs ATM)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("Vol pts")
        ax.set_title("Skew evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_local_vol_heatmap(self, asof: date | None = None, ax: plt.Axes | None = None) -> plt.Axes:
        if not self.local_vols:
            self.load_history()
        asof = asof or sorted(self.local_vols.keys())[-1]
        grid_dte, grid_ks, _ = self.grids[asof]
        lv = self.local_vols[asof]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            lv.T,
            aspect="auto",
            origin="lower",
            extent=[grid_dte.min(), grid_dte.max(), grid_ks.min(), grid_ks.max()],
            cmap="magma",
        )
        plt.colorbar(im, ax=ax, label="Local vol (%)")
        ax.set_xlabel("Days to expiry")
        ax.set_ylabel("K/S (strike/spot)")
        ax.set_title(f"Dupire local vol �?{asof}")
        return ax

    def plot_anchor_iv_history(self, anchor_ctx: dict[str, Any], ax: plt.Axes | None = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        series = anchor_ctx.get("series")
        if series is None or series.empty:
            ax.set_title("Anchor IV history (no data)")
            return ax

        plot_cols = [
            c
            for c in series.columns
            if c not in ("date", "skew_proxy") and not c.endswith("_hv") and not c.endswith("_premium")
        ]
        for col in plot_cols:
            ax.plot(series["date"], series[col], marker="o", label=col)
        ax.set_ylabel("Implied vol (%)")
        ax.set_title("Anchor contract IV (Futu, ~1 week)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        if "skew_proxy" in series.columns:
            ax2 = ax.twinx()
            ax2.plot(series["date"], series["skew_proxy"], "k--", marker="x", label="skew proxy", alpha=0.7)
            ax2.set_ylabel("Skew proxy (vol pts)", color="black")
            ax2.tick_params(axis="y", labelcolor="black")
        return ax

    def plot_vix_context(self, vix_ctx: dict[str, Any], ax: plt.Axes | None = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        hist = vix_ctx.get("history")
        if hist is None or hist.empty:
            return ax
        x = pd.to_datetime(hist.iloc[:, 0])
        ax.plot(x, hist["VIX"], marker="o", label="VIX", color="tab:red")
        if "RV_22" in hist.columns:
            ax.plot(x, hist["RV_22"], marker="s", label="22d RV", color="tab:blue", alpha=0.8)
        ax.set_ylabel("Vol (%)")
        ax.set_title("VIX vs realized vol (5d window)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_all(self, result: dict[str, Any] | None = None) -> None:
        result = result or self.analyze()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.plot_term_structure(axes[0, 0])
        self.plot_anchor_iv_history(result["anchor_context"], ax=axes[0, 1])
        self.plot_local_vol_heatmap(ax=axes[1, 0])
        self.plot_vix_context(result["vix_context"], ax=axes[1, 1])
        fig.suptitle(f"Vol surface study �?{self.cfg.underlying} (≤{self.cfg.max_dte}d)", fontsize=13)
        fig.tight_layout()
        plt.show()

    def build_delta_surface(
        self,
        asof: date | None = None,
        *,
        delta_lo: float = -0.5,
        delta_hi: float = 0.5,
        max_dte: int | None = None,
        n_delta: int = 25,
        n_dte: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (grid_dte, grid_delta, iv_grid, local_vol) for one session."""
        if not self.surfaces:
            self.load_history()
        asof = asof or latest_business_session(self.surfaces)
        max_dte = max_dte if max_dte is not None else self.cfg.max_dte
        df = self.surfaces[asof]
        dte_vals = np.array(sorted(df["dte"].unique()))
        dte_vals = dte_vals[dte_vals <= max_dte]
        if n_dte is not None and len(dte_vals) > n_dte:
            dte_vals = np.linspace(dte_vals.min(), dte_vals.max(), n_dte).astype(int)
            dte_vals = np.unique(dte_vals)
        delta_vals = np.linspace(delta_lo, delta_hi, n_delta)
        g_dte, g_delta, iv_g = build_iv_grid_delta(
            df, delta_grid=delta_vals, dte_grid=dte_vals, max_dte=max_dte
        )
        spot = float(df["spot"].iloc[0])
        lv = dupire_local_vol_delta(spot, g_dte, g_delta, iv_g, r=self.cfg.risk_free_rate)
        return g_dte, g_delta, iv_g, lv

    def _build_delta_plot_views(
        self,
        *,
        full_delta: tuple[float, float] = (-0.95, 0.95),
        zoom_delta: tuple[float, float] = (-0.55, 0.55),
        zoom_max_dte: int = 30,
        full_n_delta: int = 40,
        zoom_n_delta: int = 25,
    ) -> dict[tuple[bool, bool], dict[str, Any]]:
        """Precompute the 4 surfaces: (full|zoom) × (IV|local vol)."""
        if not self.surfaces:
            self.load_history()
        today_d = latest_business_session(self.surfaces)
        views: dict[tuple[bool, bool], dict[str, Any]] = {}

        g_full, d_full, iv_full, lv_full = self.build_delta_surface(
            today_d,
            delta_lo=full_delta[0],
            delta_hi=full_delta[1],
            n_delta=full_n_delta,
        )
        full_label = f"δ∈[{full_delta[0]:+.2f},{full_delta[1]:+.2f}], ≤{self.cfg.max_dte}d"
        views[(True, False)] = {
            "grid_dte": g_full,
            "grid_delta": d_full,
            "z": iv_full,
            "local": False,
            "title": f"IV �?{today_d} ({full_label})",
        }
        views[(True, True)] = {
            "grid_dte": g_full,
            "grid_delta": d_full,
            "z": lv_full,
            "local": True,
            "title": f"Local vol �?{today_d} ({full_label})",
        }

        g_zoom, d_zoom, iv_zoom, lv_zoom = self.build_delta_surface(
            today_d,
            delta_lo=zoom_delta[0],
            delta_hi=zoom_delta[1],
            max_dte=zoom_max_dte,
            n_delta=zoom_n_delta,
        )
        zoom_label = f"δ∈[{zoom_delta[0]:+.2f},{zoom_delta[1]:+.2f}], DTE<{zoom_max_dte}d"
        views[(False, False)] = {
            "grid_dte": g_zoom,
            "grid_delta": d_zoom,
            "z": iv_zoom,
            "local": False,
            "title": f"IV �?{today_d} ({zoom_label})",
        }
        views[(False, True)] = {
            "grid_dte": g_zoom,
            "grid_delta": d_zoom,
            "z": lv_zoom,
            "local": True,
            "title": f"Local vol �?{today_d} ({zoom_label})",
        }
        return views

    def plot_delta_surface(
        self,
        *,
        full: bool = True,
        local: bool = False,
        full_delta: tuple[float, float] = (-0.95, 0.95),
        zoom_delta: tuple[float, float] = (-0.55, 0.55),
        zoom_max_dte: int = 30,
        full_n_delta: int = 40,
        zoom_n_delta: int = 25,
    ) -> Any:
        """Pick one of 4 views: full/zoom × IV/local vol."""
        if self._delta_plot_views is None:
            self._delta_plot_views = self._build_delta_plot_views(
                full_delta=full_delta,
                zoom_delta=zoom_delta,
                zoom_max_dte=zoom_max_dte,
                full_n_delta=full_n_delta,
                zoom_n_delta=zoom_n_delta,
            )
        key = (bool(full), bool(local))
        if key not in self._delta_plot_views:
            self._invalidate_plot_cache()
            self._delta_plot_views = self._build_delta_plot_views(
                full_delta=full_delta,
                zoom_delta=zoom_delta,
                zoom_max_dte=zoom_max_dte,
                full_n_delta=full_n_delta,
                zoom_n_delta=zoom_n_delta,
            )
        view = self._delta_plot_views[key]
        return plot_single_delta_surface(
            view["grid_dte"],
            view["grid_delta"],
            view["z"],
            local=view["local"],
            title=view["title"],
        )

    def plot_delta_study(
        self,
        *,
        full_delta: tuple[float, float] = (-0.95, 0.95),
        zoom_delta: tuple[float, float] = (-0.55, 0.55),
        zoom_max_dte: int = 30,
        full_n_delta: int = 40,
    ) -> dict[str, Any]:
        """Linked 3D IV/local-vol pairs (full + zoom)."""
        if not self.surfaces:
            self.load_history()
        biz = business_days_only(self.surfaces.keys())
        today_d = biz[-1] if biz else sorted(self.surfaces.keys())[-1]
        g_dte, g_delta, iv_g, lv = self.build_delta_surface(
            today_d,
            delta_lo=full_delta[0],
            delta_hi=full_delta[1],
            n_delta=full_n_delta,
        )
        full_fig = plot_linked_delta_surfaces(
            g_dte,
            g_delta,
            iv_g,
            lv,
            title=(
                f"IV & local vol �?{today_d} "
                f"(δ∈[{full_delta[0]:+.2f},{full_delta[1]:+.2f}], ≤{self.cfg.max_dte}d)"
            ),
        )
        z_dte, z_delta, z_iv, z_lv = self.build_delta_surface(
            today_d, delta_lo=zoom_delta[0], delta_hi=zoom_delta[1], max_dte=zoom_max_dte
        )
        zoom_fig = plot_linked_delta_surfaces(
            z_dte,
            z_delta,
            z_iv,
            z_lv,
            title=(
                f"IV & local vol �?{today_d} "
                f"(zoom δ∈[{zoom_delta[0]:+.1f},{zoom_delta[1]:+.1f}], DTE<{zoom_max_dte}d)"
            ),
        )
        return {
            "full_fig": full_fig,
            "zoom_fig": zoom_fig,
            "today": today_d,
        }

    def plot_bloomberg_3d_surface(
        self,
        asof: date | None = None,
        max_dte: int | None = None,
    ) -> Any:
        """Bloomberg-style 3D IV and Local Vol surface with selection gridded on Moneyness (K/S)."""
        import plotly.graph_objects as go
        if not self.surfaces:
            self.load_history()
        asof = asof or latest_business_session(self.surfaces)
        df_today = self.surfaces[asof]
        spot = float(df_today["spot"].iloc[0])

        sub = df_today[np.isfinite(df_today["iv"]) & np.isfinite(df_today["dte"])].copy()
        raw_dtes = np.unique(sub["dte"].dropna().to_numpy())
        # Use full dynamic range of expires up to the max available to reveal the entire surface
        dte_grid = raw_dtes if max_dte is None else raw_dtes[raw_dtes <= max_dte]

        g_dte, g_ks, iv_g = build_iv_grid(df_today, dte_grid=dte_grid, max_dte=max_dte or int(raw_dtes.max()))

        x_ks = g_ks[0, :]
        y_dte = g_dte[:, 0]

        iv_g_smooth = smooth_iv_grid_quadratic(df_today, g_dte, g_ks, iv_g)
        lv = dupire_local_vol(spot, g_dte, g_ks, iv_g_smooth, r=self.cfg.risk_free_rate)

        fig = go.Figure()
        
        # 1. Raw market IV (with beautiful organic wrinkles)
        fig.add_trace(go.Surface(
            x=x_ks, y=y_dte, z=iv_g,
            name="Raw Implied Vol", colorscale="Viridis", visible=True,
            colorbar=dict(title="Raw IV (%)", x=1.02),
            opacity=1.0,
        ))

        # 2. Processed smooth Model IV (SVI-style quadratic regression)
        fig.add_trace(go.Surface(
            x=x_ks, y=y_dte, z=iv_g_smooth,
            name="Smooth Implied Vol", colorscale="Cividis", visible=False,
            colorbar=dict(title="Smooth IV (%)", x=1.02),
            opacity=1.0,
        ))

        # 3. Arbitrage-free Dupire Local Vol
        fig.add_trace(go.Surface(
            x=x_ks, y=y_dte, z=lv,
            name="Local Vol", colorscale="Magma", visible=False,
            colorbar=dict(title="Local Vol (%)", x=1.02),
            opacity=1.0,
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Moneyness (K/S)", yaxis_title="DTE", zaxis_title="Vol (%)",
                aspectmode="manual", aspectratio=dict(x=1.0, y=1.2, z=0.6),
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8)),
            ),
            title=f"{self.cfg.underlying} - {asof} (Moneyness K/S)",
            updatemenus=[dict(
                type="buttons", direction="right", x=0.5, y=1.1, xanchor="center",
                buttons=[
                    dict(label="Raw IV", method="update",
                         args=[{"visible": [True, False, False]},
                               {"scene.zaxis.title": "Raw IV (%)"}]),
                    dict(label="Smooth IV", method="update",
                         args=[{"visible": [False, True, False]},
                               {"scene.zaxis.title": "Smooth IV (%)"}]),
                    dict(label="Arb-free Local Vol", method="update",
                         args=[{"visible": [False, False, True]},
                               {"scene.zaxis.title": "Local Vol (%)"}]),
                ],
            )],
            width=820, height=580,
        )
        return fig

    def plot_sentiment_gauge(
        self,
        result: dict[str, Any],
        today_scores: np.ndarray,
        sentiment: dict[str, Any],
    ) -> Any:
        """Render a dual-panel Matte-Speedometer Gauge + Short Environment Summary."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        today_d = latest_business_session(self.surfaces)
        today_f = self.features[today_d]
        aiv = float(today_f.get("atm_iv_30d", np.nan))
        psk = float(today_f.get("skew_25d", np.nan))
        tsl = float(today_f.get("term_slope", np.nan))
        vix = float(result.get("vix_context", {}).get("vix", np.nan))

        anomalies = result["anomalies"]
        if result["changes"] is not None and not result["changes"].empty:
            chg = result["changes"]
            def d5(name):
                row = chg.loc[chg["feature"]==name]
                return float(row["delta_5d"].iloc[0]) if not row.empty and np.isfinite(row["delta_5d"].iloc[0]) else 0.0
            d_skew = d5("skew_25d")
            d_bfly = d5("butterfly_25d")
        else:
            d_skew = 0.0
            d_bfly = 0.0

        vrp_val = result.get("vix_context", {}).get("vrp", vix - aiv)
        anchor_ctx = result.get("anchor_context", {})
        anchor_changes = anchor_ctx.get("changes", {})
        skew_proxy_chg = anchor_changes.get("skew_proxy", 0.0)

        ps = np.clip(-psk * 4.0, -35, 35)
        ts_factor = np.clip(tsl * 5.0, -25, 25)
        vix_component = np.clip((20.0 - vix) * 2.0, -20, 20) if np.isfinite(vix) else 0.0
        vrp_factor = np.clip((6.0 - vrp_val) * 2.5, -15, 15) if np.isfinite(vrp_val) else 0.0
        anchor_skew_factor = np.clip(-skew_proxy_chg * 5.0, -10, 10) if np.isfinite(skew_proxy_chg) else 0.0

        sentiment_score = float(np.clip(ps + ts_factor + vix_component + vrp_factor + anchor_skew_factor, -100, 100))

        if sentiment_score > 50:
            sentiment_label = "Extremely Bullish"
            sentiment_color = "#27ae60"
        elif sentiment_score > 15:
            sentiment_label = "Slightly Bullish"
            sentiment_color = "#2ecc71"
        elif sentiment_score > -15:
            sentiment_label = "Neutral"
            sentiment_color = "#f1c40f"
        elif sentiment_score > -50:
            sentiment_label = "Slightly Bearish"
            sentiment_color = "#e67e22"
        else:
            sentiment_label = "Extremely Bearish"
            sentiment_color = "#e74c3c"

        fig, (ax_gauge, ax_text) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1.25]})

        # Speedometer Gauge
        r_outer = 1.0
        r_inner = 0.7
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
        angles = np.linspace(np.pi, 0, 6)

        for i in range(5):
            t_seg = np.linspace(angles[i], angles[i+1], 50)
            x_outer = r_outer * np.cos(t_seg)
            y_outer = r_outer * np.sin(t_seg)
            x_inner = r_inner * np.cos(t_seg)
            y_inner = r_inner * np.sin(t_seg)
            x_polygon = np.concatenate([x_outer, x_inner[::-1]])
            y_polygon = np.concatenate([y_outer, y_inner[::-1]])
            ax_gauge.fill(x_polygon, y_polygon, color=colors[i], alpha=0.85)

        normalized_score = (sentiment_score + 100.0) / 200.0
        needle_angle = np.pi - normalized_score * np.pi

        needle_len = 0.95
        ax_gauge.annotate(
            "",
            xy=(needle_len * np.cos(needle_angle), needle_len * np.sin(needle_angle)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="wedge,tail_width=0.35,shrink_factor=0.5", color="#2c3e50", zorder=10)
        )

        center_circle = patches.Circle((0, 0), 0.12, color="#2c3e50", zorder=11)
        center_rim = patches.Circle((0, 0), 0.15, fill=False, edgecolor="#7f8c8d", lw=1.5, zorder=12)
        ax_gauge.add_patch(center_circle)
        ax_gauge.add_patch(center_rim)

        ticks = [-100, -50, 0, 50, 100]
        for val in ticks:
            t_val = np.pi - ((val + 100.0) / 200.0) * np.pi
            tx = (r_outer + 0.12) * np.cos(t_val)
            ty = (r_outer + 0.12) * np.sin(t_val)
            ax_gauge.text(tx, ty, str(val), ha="center", va="center", fontsize=9, fontweight="bold", color="#7f8c8d")

        ax_gauge.text(0, -0.15, sentiment_label, ha="center", va="center", fontsize=15, fontweight="bold", color=sentiment_color)
        ax_gauge.text(0, -0.32, f"Sentiment Score: {sentiment_score:+.1f} / 100", ha="center", va="center", fontsize=11, fontweight="bold", color="#34495e")

        ax_gauge.set_xlim(-1.25, 1.25)
        ax_gauge.set_ylim(-0.45, 1.25)
        ax_gauge.axis("off")

        # Bullets Summary
        ax_text.axis("off")
        box = patches.FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98,
            boxstyle="round,pad=0.03",
            facecolor="#fdfefe", edgecolor="#d5dbdb", lw=1.5
        )
        ax_text.add_patch(box)

        joint_flag_str = sentiment["joint_flags"][0].upper().replace("_", " ") if sentiment["joint_flags"] else "NEUTRAL"
        pc1_score = today_scores[0]
        pc2_score = today_scores[1]

        hump_active = anomalies.get("event_hump", False)
        term_narrative = f"Hump @ {anomalies.get('hump_dte', 0):.0f}d" if hump_active else "No Hump"
        bfly_today = today_f.get("butterfly_25d", np.nan)

        bullet_points = [
            f"# Market Volatility Structure (As of {today_d})",
            f"• Implied Pricing: ATM 30d IV = {aiv:.1f}% | VIX index baseline = {vix:.1f}%",
            f"• Vol Risk Premium (VRP): {vrp_val:+.1f} pts (VIX premium over 22d Realized Vol)",
            f"• Put/Call Skew Premium: 25d put spread skew = {psk:+.1f} vol pts (5d delta: {d_skew:+.1f} pts)",
            f"• Curvature & Fly Convexity: 25d butterfly = {bfly_today:+.1f} vol pts (5d delta: {d_bfly:+.1f} pts)",
            f"• Term Structure Tenor: Front vs back roll spread = {tsl:+.1f} vol pts ({term_narrative})",
            f"• Surface Delta PCA Shocks: PC1 parallel shift = {pc1_score:+.2f} | PC2 skew tilt = {pc2_score:+.2f}",
        ]

        y_pos = 0.88
        for line in bullet_points:
            is_header = line.startswith("#")
            display_line = line.replace("# ", "") if is_header else line
            font_w = "bold" if is_header else "normal"
            font_s = 12 if is_header else 10.2
            font_c = "#2c3e50" if is_header else "#34495e"
            ax_text.text(0.05, y_pos, display_line, fontsize=font_s, fontweight=font_w, color=font_c, va="center")
            y_pos -= 0.12

        plt.tight_layout()
        plt.show()

    def print_conclusion(self, *, delta_lo: float = -0.05, delta_hi: float = 0.05) -> None:
        print(build_study_conclusion(self, delta_lo=delta_lo, delta_hi=delta_hi))

    def print_report(self) -> None:
        result = self.analyze()
        print(result["commentary"])
        print("\n--- Feature changes (5d) ---")
        print(result["changes"].to_string(index=False))
        print("\n--- Anomalies ---")
        for k, v in result["anomalies"].items():
            print(f"  {k}: {v}")
        print("\n--- Sentiment ---")
        for k, v in result["sentiment"].items():
            print(f"  {k}: {v}")
