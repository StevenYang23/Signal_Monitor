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
from typing import Any
from urllib import error, request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm

try:
    import futu as ft
except ImportError:
    ft = None

try:
    import yfinance as yf
except ImportError:
    yf = None

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

# Index underlyings: Futu index snapshots often require extra permissions — use yfinance.
SPOT_YF_DEFAULT: dict[str, str] = {
    "US..SPX": "^SPX",
    "US..IXIC": "^IXIC",
    "US..DJI": "^DJI",
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
    cache_dir: Path = field(default_factory=lambda: Path("research/data/vol_surface"))
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
    if ft is None:
        return False
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
    if raw.empty or "Close" not in raw:
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
            "Use Kernel → Restart, then re-run from cell 1."
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


def business_days_only(dates: list[date] | set[date]) -> list[date]:
    return sorted(d for d in dates if d.weekday() < 5)


def latest_business_session(surfaces: dict[date, pd.DataFrame]) -> date:
    biz = business_days_only(surfaces.keys())
    if biz:
        return biz[-1]
    return sorted(surfaces.keys())[-1]


def trim_surfaces_to_sessions(
    surfaces: dict[date, pd.DataFrame],
    n_sessions: int,
    *,
    business_days: bool = True,
) -> dict[date, pd.DataFrame]:
    dates = business_days_only(surfaces.keys()) if business_days else sorted(surfaces.keys())
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


def fetch_and_cache(cfg: VolSurfaceConfig | None = None) -> pd.DataFrame:
    cfg = cfg or VolSurfaceConfig()
    _require_futu()
    quote_ctx = ft.OpenQuoteContext(host=cfg.host, port=cfg.port)
    try:
        df = fetch_option_chain_futu(quote_ctx, cfg)
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
) -> float | np.nan:
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
    moneyness_grid: np.ndarray | None = None,
    max_dte: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dte_grid = dte_grid if dte_grid is not None else np.array([7, 10, 14, 21, 30, 45, 60])
    dte_grid = dte_grid[dte_grid <= max_dte]
    moneyness_grid = (
        moneyness_grid if moneyness_grid is not None else np.linspace(0.85, 1.15, 13)
    )

    points = df[["dte", "moneyness", "iv"]].dropna().to_numpy()
    if len(points) < 4:
        raise ValueError("Not enough option quotes to build IV grid.")

    grid_dte, grid_mny = np.meshgrid(dte_grid, moneyness_grid, indexing="ij")
    iv_grid = interpolate.griddata(
        points[:, :2],
        points[:, 2],
        (grid_dte, grid_mny),
        method="linear",
    )
    if np.isnan(iv_grid).any():
        iv_grid = interpolate.griddata(
            points[:, :2],
            points[:, 2],
            (grid_dte, grid_mny),
            method="nearest",
        )
    iv_grid = np.clip(iv_grid, 1.0, 200.0)
    return grid_dte, grid_mny, iv_grid


def dupire_local_vol(
    spot: float,
    grid_dte: np.ndarray,
    grid_moneyness: np.ndarray,
    iv_grid: np.ndarray,
    r: float = 0.045,
) -> np.ndarray:
    """Dupire local vol (%) from an implied-vol grid."""
    t_years = np.maximum(grid_dte / TRADING_DAYS, 1 / TRADING_DAYS)
    strike_grid = grid_moneyness * spot
    call_grid = bs_call_price(spot, strike_grid, t_years, r, iv_grid / 100.0)

    t_axis = t_years[:, 0]
    k_axis = strike_grid[0, :]
    dC_dT = np.gradient(call_grid, t_axis, axis=0)
    dC_dK = np.gradient(call_grid, k_axis, axis=1)
    d2C_dK2 = np.gradient(dC_dK, k_axis, axis=1)

    denom = strike_grid**2 * d2C_dK2
    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * dC_dT / denom
    local_vol = np.sqrt(np.clip(local_var, 0.0, None)) * 100.0
    return np.where(np.isfinite(local_vol), local_vol, np.nan)


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
    """Interpolate IV onto a (DTE × delta) mesh from the option chain."""
    dte_grid = dte_grid if dte_grid is not None else np.array([7, 10, 14, 21, 30, 45, 60])
    dte_grid = dte_grid[dte_grid <= max_dte]
    delta_grid = delta_grid if delta_grid is not None else np.linspace(-0.5, 0.5, 21)

    sub = df[np.isfinite(df["iv"]) & np.isfinite(df["delta"]) & np.isfinite(df["dte"])].copy()
    if len(sub) < 4:
        raise ValueError("Not enough option quotes to build delta IV grid.")

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
    k_axis = strike_grid[0, :]
    dC_dT = np.gradient(call_grid, t_axis, axis=0)
    dC_dK = np.gradient(call_grid, k_axis, axis=1)
    d2C_dK2 = np.gradient(dC_dK, k_axis, axis=1)

    denom = strike_grid**2 * d2C_dK2
    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * dC_dT / denom
    local_vol = np.sqrt(np.clip(local_var, 0.0, None)) * 100.0
    return np.where(np.isfinite(local_vol), local_vol, np.nan)


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
                    "as bullet points (4–7 lines, each starting with '- '). Cover vol level, put/call skew "
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
        f"Sentiment — fear: {sentiment.get('fear', '').replace(chr(8212), '-')}; "
        f"calls: {sentiment.get('call_opportunity', '').replace(chr(8212), '-')}; "
        f"event: {sentiment.get('event_risk', '').replace(chr(8212), '-')}."
    )

    n_ml = ml_ctx.get("n_anomalies", 0)
    n_lv = len(lv_spikes)
    if n_ml or n_lv:
        bullets.append(
            f"Local-vol diagnostics: {n_ml} ML IV outlier(s), {n_lv} local-vol spike(s) "
            f"— check for stale quotes or event pockets."
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
    """One 3D surface on delta × DTE — IV or Dupire local vol."""
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
    title: str = "IV surface change (today − previous session)",
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
        fear_label = "elevated — downside protection being bid"
    elif fear_score >= 1:
        fear_label = "moderate — some hedging demand"
    else:
        fear_label = "calm — limited crash-insurance premium"

    if call_opportunity_score >= 2 and fear_score <= 1:
        call_label = "favorable — upside vol relatively cheap vs puts"
    elif call_opportunity_score >= 1:
        call_label = "mixed — selective call structures may work"
    else:
        call_label = "cautious — calls not clearly cheap vs hedging demand"

    if anomalies.get("event_hump"):
        event_label = f"event priced near {anomalies['hump_dte']:.0f}d (IV hump)"
    elif today.get("term_slope", 0) > 2:
        event_label = "front-end IV elevated — near-term stress"
    elif today.get("term_slope", 0) < -2:
        event_label = "inverted term structure — immediate concern"
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
            f"futu-api {ver} has no get_option_volatility — anchor IV history skipped. "
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
            f"- **Skew proxy** (25Δ put − 30d ATM call): window change {skew_chg:+.1f} vol pts → {direction}"
        )
    lines.append("Note: each series tracks one fixed contract; DTE/moneyness drift over the window.")
    return lines


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


def generate_demo_surfaces(cfg: VolSurfaceConfig | None = None, n_days: int = 6) -> dict[date, pd.DataFrame]:
    """Synthetic SPX-like surfaces for offline notebook runs when OpenD is unavailable."""
    cfg = cfg or VolSurfaceConfig()
    spot0 = 5400.0
    history: dict[date, pd.DataFrame] = {}
    strikes = np.arange(4800, 6010, 25)
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
        if not self.surfaces and use_demo_if_empty:
            warnings.warn(
                "No cached surfaces — loading demo SPX surfaces.",
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

    def fetch_live(self) -> pd.DataFrame:
        df = fetch_and_cache(self.cfg)
        asof = pd.Timestamp(df["asof_date"].iloc[0]).date()
        self._invalidate_plot_cache()
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
        ax.set_title(f"{self.cfg.underlying} — ATM term structure")
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
        grid_dte, grid_mny, _ = self.grids[asof]
        lv = self.local_vols[asof]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            lv.T,
            aspect="auto",
            origin="lower",
            extent=[grid_dte.min(), grid_dte.max(), grid_mny.min(), grid_mny.max()],
            cmap="magma",
        )
        plt.colorbar(im, ax=ax, label="Local vol (%)")
        ax.set_xlabel("Days to expiry")
        ax.set_ylabel("Moneyness (K/S approx)")
        ax.set_title(f"Dupire local vol — {asof}")
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
        fig.suptitle(f"Vol surface study — {self.cfg.underlying} (≤{self.cfg.max_dte}d)", fontsize=13)
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
            "title": f"IV — {today_d} ({full_label})",
        }
        views[(True, True)] = {
            "grid_dte": g_full,
            "grid_delta": d_full,
            "z": lv_full,
            "local": True,
            "title": f"Local vol — {today_d} ({full_label})",
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
            "title": f"IV — {today_d} ({zoom_label})",
        }
        views[(False, True)] = {
            "grid_dte": g_zoom,
            "grid_delta": d_zoom,
            "z": lv_zoom,
            "local": True,
            "title": f"Local vol — {today_d} ({zoom_label})",
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
                f"IV & local vol — {today_d} "
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
                f"IV & local vol — {today_d} "
                f"(zoom δ∈[{zoom_delta[0]:+.1f},{zoom_delta[1]:+.1f}], DTE<{zoom_max_dte}d)"
            ),
        )
        return {
            "full_fig": full_fig,
            "zoom_fig": zoom_fig,
            "today": today_d,
        }

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
