"""SPX Quant Signal Hub (simple) — live IV / local-vol surface + anomaly list."""

from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
import traceback
import urllib.parse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib_temp"),
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vol_surface import (  # noqa: E402
    VolSurfaceConfig,
    VolSurfaceStudy,
    build_iv_grid,
    detect_surface_anomalies,
    detect_term_hump,
    dupire_local_vol,
    fetch_aux_vol_context,
    fetch_spot_yfinance,
    fetch_vix_context,
    smooth_iv_grid_quadratic,
    fetch_anchor_iv_histories,
)
from volatility_regime import HMMVolatilityRegime  # noqa: E402

PORT = int(os.environ.get("PORT", 8050))
INDEX_NAME = "SPX"
OPTION_TICKER = "US..SPX"
SPOT_TICKER = "^SPX"
HMM_NAME = "SPX"


def _futu_reachable(host: str = "127.0.0.1", port: int = 11111, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _prepare_study(study: VolSurfaceStudy, cfg: VolSurfaceConfig, *, futu_up: bool) -> None:
    if not futu_up:
        raise ConnectionError(
            f"Futu OpenD is offline at {cfg.host}:{cfg.port}; live option data is required."
        )
    # Live chain only — no multi-day surface cache / day-over-day deltas.
    study.fetch_live(save=False)


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _aux_level(aux: dict, key: str) -> float | None:
    block = aux.get(key) or {}
    level = block.get("level")
    try:
        v = float(level)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class QuantEngine:
    def process_index(self, short_name: str = INDEX_NAME) -> dict:
        if short_name != INDEX_NAME:
            raise ValueError(f"Simple hub is SPX-only; got {short_name}")
        print(f"[pipeline] {short_name} start", flush=True)
        try:
            payload = self._build_payload()
            print(f"[pipeline] {short_name} OK", flush=True)
            return payload
        except Exception as exc:
            print(f"[pipeline] {short_name} failed: {exc}", flush=True)
            traceback.print_exc()
            return {"exists": False, "error": str(exc)}

    def _build_payload(self) -> dict:
        warnings_list: list[str] = []

        cfg = VolSurfaceConfig(underlying=OPTION_TICKER, max_dte=60, lookback_days=15)
        futu_up = _futu_reachable(cfg.host, cfg.port)
        study = VolSurfaceStudy(cfg)
        _prepare_study(study, cfg, futu_up=futu_up)

        dates = sorted(study.surfaces.keys())
        today_d = dates[-1]
        df_today = study.surfaces[today_d]
        option_spot = float(df_today["spot"].iloc[0])
        try:
            spot, _ = fetch_spot_yfinance(SPOT_TICKER)
        except Exception:
            spot = option_spot

        g_dte, g_ks, iv_g = build_iv_grid(df_today, max_dte=cfg.max_dte)
        x_ks = [float(v) for v in g_ks[0, :]]
        y_dte = [float(v) for v in g_dte[:, 0]]
        iv_data = [[float(v) for v in row] for row in iv_g]
        iv_g_smooth = smooth_iv_grid_quadratic(df_today, g_dte, g_ks, iv_g)
        local_vol_available = g_dte.shape[0] >= 2
        if local_vol_available:
            lv = dupire_local_vol(option_spot, g_dte, g_ks, iv_g_smooth, r=cfg.risk_free_rate)
        else:
            lv = np.full_like(iv_g_smooth, np.nan)
            warnings_list.append("Local vol unavailable: live chain contains only one expiry.")
        lv_data = [[float(v) for v in row] for row in lv]
        sv_data = [[float(v) for v in row] for row in iv_g_smooth]

        today_f = study.features.get(today_d) or {}

        vix_ctx: dict = {}
        try:
            vix_ctx = fetch_vix_context(cfg.lookback_days)
        except Exception as exc:
            warnings_list.append(f"VIX context skipped: {exc}")
            vix_ctx = {"vix": np.nan, "vrp": np.nan, "rv_22": np.nan}

        aux_ctx: dict = {}
        try:
            aux_ctx = fetch_aux_vol_context()
        except Exception as exc:
            warnings_list.append(f"Aux indices skipped: {exc}")
            aux_ctx = {"vvix": None, "skew_index": None, "cor3m": None, "meta": {}}

        # Fetch anchor IV histories to get previous day's prices
        try:
            anchor_ctx = fetch_anchor_iv_histories(df_today, cfg, query_time_period=1)
        except Exception as exc:
            warnings_list.append(f"Anchor IV history skipped: {exc}")
            anchor_ctx = {"anchors": {}, "series": pd.DataFrame(), "changes": {}, "meta": {"skipped": True}}

        # If anchor history is available, use it to calculate the previous day's IV grid
        # and detect anomalies based on the daily change
        if anchor_ctx.get("series") is not None and not anchor_ctx["series"].empty:
            # Reconstruct yesterday's IV grid using the anchor changes
            # This is a simplified approximation since we don't have the full surface history in live mode
            iv_g_prev = iv_g.copy()
            for name, meta in anchor_ctx.get("anchors", {}).items():
                chg = anchor_ctx.get("changes", {}).get(name, 0.0)
                if np.isfinite(chg):
                    # Find the closest grid point to this anchor
                    dte_idx = np.abs(g_dte[:, 0] - meta["dte"]).argmin()
                    ks_idx = np.abs(g_ks[0, :] - (meta["strike"] / spot)).argmin()
                    iv_g_prev[dte_idx, ks_idx] -= chg / 100.0  # Convert vol pts to decimal
            
            # Smooth the reconstructed previous grid
            iv_g_prev_smooth = smooth_iv_grid_quadratic(df_today, g_dte, g_ks, iv_g_prev)
            
            # Use the change in smoothed grids for anomaly detection
            iv_g_change = iv_g_smooth - iv_g_prev_smooth
            
            anomalies = detect_surface_anomalies(
                g_dte,
                g_ks,
                iv_g_change,  # Use the daily change instead of raw IV
                iv_g_smooth,
                lv if local_vol_available else None,
                today_f,
                cfg,
            )
        else:
            anomalies = detect_surface_anomalies(
                g_dte,
                g_ks,
                iv_g,
                iv_g_smooth,
                lv if local_vol_available else None,
                today_f,
                cfg,
            )
        hump = detect_term_hump(today_f, cfg)

        vix = float(vix_ctx.get("vix", np.nan))
        aiv = float(today_f.get("atm_iv_30d", np.nan))
        psk = float(today_f.get("skew_25d", np.nan))
        tsl = float(today_f.get("term_slope", np.nan))
        bfly = float(today_f.get("butterfly_25d", np.nan))
        vrp_val = vix_ctx.get("vrp")
        if vrp_val is None or not np.isfinite(float(vrp_val)):
            vrp_val = vix - aiv if np.isfinite(vix) and np.isfinite(aiv) else np.nan
        else:
            vrp_val = float(vrp_val)

        hmm_regime_str = "unknown"
        hmm_prob_today = 50.0
        hmm_prob_tmr = 50.0
        hmm_signal_bool = False
        move_rows: list[dict[str, str]] = []
        hmm_history_dates: list[str] = []
        hmm_history_prices: list[float] = []
        hmm_history_opens: list[float] = []
        hmm_history_highs: list[float] = []
        hmm_history_lows: list[float] = []
        hmm_history_volumes: list[float] = []
        hmm_history_rv: list[float] = []
        hmm_history_iv: list[float] = []
        hmm_history_regimes: list[str] = []
        hmm_history_samples: list[str] = []
        hmm_today_date: str | None = None
        hmm_iv_label = "VIX"
        hmm_iv_22d_ago: float | None = None

        try:
            hmm_model = HMMVolatilityRegime.from_market(HMM_NAME, signal_mode=True)
            hmm_model.download_data()
            hmm_model.update_latest_prices()
            hmm_model.fit_once()
            hmm_sig = hmm_model.today_signal

            hmm_iv_label = hmm_model.vol_ticker.removeprefix("^")

            if hmm_sig is not None:
                hmm_regime_str = str(hmm_sig.get("hmm", "unknown"))
                hmm_prob_today = round(float(hmm_sig.get("prob_low_vol", 0.5)) * 100, 1)
                hmm_prob_tmr = round(float(hmm_sig.get("prob_low_vol_tmr", 0.5)) * 100, 1)
                hmm_signal_bool = bool(hmm_sig.get("trade_signal", False))

                move_table = hmm_model.format_movement_table()
                for idx in move_table.index:
                    move_rows.append(
                        {
                            "horizon": str(idx),
                            "implied": str(move_table.loc[idx, "Implied"]),
                            "historical": str(move_table.loc[idx, "Historical"]),
                            "spot_implied": str(move_table.loc[idx, "Spot ± implied"]),
                        }
                    )

            feats = hmm_model.features
            regs = hmm_model.regimes
            if feats is not None and regs is not None:
                aligned = feats.join(regs, how="inner").dropna(subset=["SPX", "RV_22", "IV", "hmm"])
                history_df = aligned.tail(44)
                ohlc_df = hmm_model._download_ohlc(
                    hmm_model.underly_ticker,
                    start=history_df.index[0],
                    end=history_df.index[-1],
                ).reindex(history_df.index)

                hmm_history_dates = [d.strftime("%Y-%m-%d") for d in history_df.index]
                hmm_history_prices = [float(v) for v in history_df["SPX"]]
                hmm_history_opens = [float(v) for v in ohlc_df["Open"].fillna(history_df["SPX"])]
                hmm_history_highs = [float(v) for v in ohlc_df["High"].fillna(history_df["SPX"])]
                hmm_history_lows = [float(v) for v in ohlc_df["Low"].fillna(history_df["SPX"])]
                hmm_history_volumes = [
                    float(v)
                    for v in ohlc_df.get("Volume", pd.Series(0, index=ohlc_df.index)).fillna(0)
                ]
                hmm_history_rv = [float(v) for v in history_df["RV_22"]]
                hmm_history_iv = [float(v) for v in history_df["IV"]]
                hmm_history_regimes = [str(v) for v in history_df["hmm"]]
                sample_col = history_df["sample"] if "sample" in history_df.columns else pd.Series(
                    "in_sample", index=history_df.index
                )
                hmm_history_samples = [str(v) for v in sample_col]

                if hmm_model.today_date is not None:
                    hmm_today_date = pd.Timestamp(hmm_model.today_date).strftime("%Y-%m-%d")
                    if hmm_model.today_date in hmm_model.features.index:
                        lag_iv = hmm_model.features["IV"].shift(22).loc[hmm_model.today_date]
                        if pd.notna(lag_iv):
                            hmm_iv_22d_ago = float(lag_iv)
        except Exception as hmm_exc:
            msg = f"HMM skipped: {hmm_exc}"
            warnings_list.append(msg)
            print(f"[pipeline] SPX HMM: {hmm_exc}", flush=True)

        kind_counts: dict[str, int] = {}
        for a in anomalies:
            kind_counts[a["kind"]] = kind_counts.get(a["kind"], 0) + 1

        return {
            "exists": True,
            "warnings": warnings_list,
            "date": str(today_d),
            "spot": spot,
            "aiv": aiv,
            "vix": vix,
            "psk": psk,
            "tsl": tsl,
            "bfly": bfly,
            "vrp": vrp_val,
            "term_hump": hump,
            "aux": {
                "vvix": _aux_level(aux_ctx, "vvix"),
                "vvix_pctl": (aux_ctx.get("vvix") or {}).get("pctl_63d"),
                "skew_index": _aux_level(aux_ctx, "skew_index"),
                "skew_pctl": (aux_ctx.get("skew_index") or {}).get("pctl_63d"),
                "cor3m": _aux_level(aux_ctx, "cor3m"),
                "cor3m_pctl": (aux_ctx.get("cor3m") or {}).get("pctl_63d"),
                "cor3m_ticker": aux_ctx.get("cor3m_ticker"),
                "raw": aux_ctx,
            },
            "anomalies": anomalies,
            "anomaly_counts": kind_counts,
            "surface_x": x_ks,
            "surface_y": y_dte,
            "surface_z": iv_data,
            "surface_sv": sv_data,
            "surface_w": lv_data,
            "local_vol_available": local_vol_available,
            "hmm_regime": hmm_regime_str,
            "hmm_prob_today": hmm_prob_today,
            "hmm_prob_tmr": hmm_prob_tmr,
            "hmm_signal": hmm_signal_bool,
            "hmm_move_table": move_rows,
            "hmm_dates": hmm_history_dates,
            "hmm_prices": hmm_history_prices,
            "hmm_opens": hmm_history_opens,
            "hmm_highs": hmm_history_highs,
            "hmm_lows": hmm_history_lows,
            "hmm_volumes": hmm_history_volumes,
            "hmm_rv": hmm_history_rv,
            "hmm_iv": hmm_history_iv,
            "hmm_regimes": hmm_history_regimes,
            "hmm_samples": hmm_history_samples,
            "hmm_today_date": hmm_today_date,
            "hmm_iv_label": hmm_iv_label,
            "hmm_iv_22d_ago": hmm_iv_22d_ago,
        }


class IndexDataStore:
    """Background loader with in-memory cache (SPX only)."""

    def __init__(self, engine: QuantEngine):
        self.engine = engine
        self._cache: dict[str, dict] = {}
        self._state: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition()

    def _ready_payload(self) -> dict:
        out = dict(self._cache[INDEX_NAME])
        out["status"] = "ready"
        return out

    def _loading_payload(self) -> dict:
        with self._cond:
            st = self._state.get(INDEX_NAME, {})
        return {
            "exists": False,
            "status": "loading",
            "index": INDEX_NAME,
            "started_at": st.get("started_at"),
            "elapsed_sec": round(time.time() - float(st.get("started_mono", time.time())), 1),
        }

    def _kickoff(self) -> None:
        with self._cond:
            if INDEX_NAME in self._cache:
                return
            st = self._state.get(INDEX_NAME)
            if st and st.get("status") in ("loading", "error"):
                return

        if not self._lock.acquire(blocking=False):
            return

        with self._cond:
            self._state[INDEX_NAME] = {
                "status": "loading",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "started_mono": time.monotonic(),
            }
            self._cond.notify_all()

        threading.Thread(target=self._run_worker, name="load-SPX", daemon=True).start()

    def get(self, index_name: str = INDEX_NAME, *, async_mode: bool = False, timeout: float = 600.0) -> dict:
        index_name = index_name.upper()
        if index_name != INDEX_NAME:
            raise ValueError(f"Unknown index: {index_name}")

        self._kickoff()

        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                if INDEX_NAME in self._cache:
                    return self._ready_payload()
                st = self._state.get(INDEX_NAME)
                if st and st.get("status") == "error":
                    return {
                        "exists": False,
                        "status": "error",
                        "index": INDEX_NAME,
                        "error": st.get("error", "pipeline failed"),
                    }
                if async_mode:
                    return self._loading_payload()

            if time.monotonic() >= deadline:
                return {
                    "exists": False,
                    "status": "error",
                    "index": INDEX_NAME,
                    "error": f"Timed out after {int(timeout)}s",
                }
            with self._cond:
                self._cond.wait(timeout=1.0)

    def invalidate(self, index_name: str = INDEX_NAME) -> None:
        index_name = index_name.upper()
        if index_name != INDEX_NAME:
            return
        with self._cond:
            self._cache.pop(INDEX_NAME, None)
            self._state.pop(INDEX_NAME, None)
            self._cond.notify_all()

    def _run_worker(self) -> None:
        try:
            payload = self.engine.process_index(INDEX_NAME)
            with self._cond:
                if payload.get("exists"):
                    self._cache[INDEX_NAME] = payload
                    self._state[INDEX_NAME] = {"status": "ready"}
                else:
                    self._state[INDEX_NAME] = {
                        "status": "error",
                        "error": payload.get("error", "pipeline failed"),
                    }
                self._cond.notify_all()
        finally:
            self._lock.release()


engine = QuantEngine()
store = IndexDataStore(engine)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPX Surface Anomaly Hub</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0e1118;color:#d0d4dc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;height:100vh;display:flex;flex-direction:column}
.header{padding:14px 24px;border-bottom:1px solid #1f2330;display:flex;align-items:center;background:#131722;gap:12px;z-index:10;flex-wrap:wrap}
.header h1{font-size:18px;font-weight:600;color:#e8eaed;letter-spacing:.3px}
.badge{background:#1c2030;border-radius:6px;padding:4px 12px;font-size:13px;color:#9aa0a8;font-weight:500}
.main-layout{display:flex;flex:1;overflow:hidden;padding:16px;gap:16px;height:calc(100vh - 54px)}
.panel{background:#131722;border-radius:8px;border:1px solid #1f2330;overflow:hidden;display:flex;flex-direction:column}
.ptitle{font-size:11px;font-weight:600;padding:10px 16px;border-bottom:1px solid #1f2330;color:#8f96a3;text-transform:uppercase;letter-spacing:.8px}
.surface-panel{flex:1.4;display:flex;flex-direction:column;min-height:400px}
.left-col{flex:1.2;display:flex;flex-direction:column;gap:16px;height:100%;overflow-y:auto;padding-right:4px}
.price-regime-panel{flex:1;display:flex;flex-direction:column;min-height:520px}
.sidebar{flex:0.8;display:flex;flex-direction:column;gap:16px;overflow-y:auto;padding-right:4px}
.controls{display:flex;gap:8px;padding:8px 16px;border-bottom:1px solid #1f2330;align-items:center}
.controls label{font-size:11px;color:#8f96a3;font-weight:500}
.rg{display:flex;gap:2px;background:#1a1f2c;border-radius:6px;padding:2px}
.rb{padding:4px 12px;border-radius:4px;font-size:11px;cursor:pointer;border:none;background:transparent;color:#8f96a3;font-weight:600;transition:all .15s}
.rb.active{background:#2a6cff;color:#fff}
.dashboard-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:16px}
.metric-card{background:#181d2e;border-radius:6px;border:1px solid #232a3f;padding:10px;display:flex;flex-direction:column;justify-content:center}
.metric-card .label{font-size:9px;color:#8f96a3;text-transform:uppercase;font-weight:600;margin-bottom:3px}
.metric-card .val{font-size:16px;font-weight:700;color:#fff}
.hmm-indicator{border-radius:6px;padding:10px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;transition:all .3s;grid-column:1 / -1}
.hmm-indicator.active{background:rgba(0,204,102,.1);border:1.5px solid #00cc66}
.hmm-indicator.inactive{background:rgba(231,76,60,.1);border:1.5px solid #e74c3c}
.summary-content{padding:12px 16px;font-size:12px;line-height:1.5;overflow-y:auto;flex:1}
.anomaly-row{margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid #1c2030}
.anomaly-row:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
.anomaly-row .kind{color:#e8eaed;font-weight:600;font-size:12px;margin-bottom:4px}
.anomaly-row .meta{color:#2a6cff;font-size:11px;font-family:monospace;margin-bottom:4px}
.anomaly-row .detail{color:#9aa3b2;font-size:11.5px;line-height:1.55}
.kind-tag{display:inline-block;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;padding:2px 6px;border-radius:3px;margin-right:6px}
.kind-smooth_residual{background:#3d2a1a;color:#e67e22}
.kind-local_spike{background:#2a1a3d;color:#bb86fc}
.kind-lv_invalid{background:#3d1a1a;color:#e74c3c}
.kind-lv_explosion{background:#3d1a2a;color:#ff6b81}
.kind-term_hump{background:#1a2a3d;color:#3498db}
.kind-extreme_skew{background:#1a3d2a;color:#2ecc71}
.kind-extreme_butterfly{background:#3d3d1a;color:#f1c40f}
.condor-table{width:100%;border-collapse:collapse;font-size:11.5px;text-align:left;color:#b0b6c2}
.condor-table th{background:#1a1f2c;color:#8f96a3;font-weight:600;padding:8px 12px;border-bottom:1.5px solid #1f2330;text-transform:uppercase;font-size:10px;letter-spacing:.5px}
.condor-table td{padding:8px 12px;border-bottom:1px solid #1c2030;font-family:monospace}
.condor-panel{flex:0 0 auto!important;min-height:0!important}
.condor-panel .condor-scroll{overflow-x:auto;overflow-y:hidden}
.refresh-btn{background:#2a6cff;border:none;color:#fff;padding:6px 14px;font-size:12px;font-weight:600;border-radius:6px;cursor:pointer;margin-left:auto;transition:opacity .15s}
.refresh-btn:hover{opacity:.9}
.refresh-btn:disabled{opacity:.45;cursor:not-allowed}
.plot-wrap{position:relative;flex:1;min-height:260px}
.plot-target{width:100%;height:100%;min-height:260px;background:transparent!important}
.plot-target .svg-container{background:transparent!important}
.plot-target .main-svg{background:transparent!important}
.loading-overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px;text-align:center;color:#8f96a3;background:rgba(19,23,34,.92);z-index:5}
.loading-overlay.hidden{display:none}
.loading-overlay .title{color:#e8eaed;font-size:15px;font-weight:600;margin-bottom:8px}
.loading-overlay .hint{font-size:12px;line-height:1.6;max-width:380px}
.loading-overlay .elapsed{font-size:11px;color:#2a6cff;margin-top:12px;font-variant-numeric:tabular-nums}
.error-box{padding:40px;text-align:center;color:#e74c3c;font-weight:600}
.aux-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:12px 16px}
</style>
</head>
<body>
<div class="header">
  <h1>SPX Surface Anomaly Hub</h1>
  <button class="refresh-btn" id="refreshBtn" onclick="refreshCurrent()">Refresh</button>
  <span id="statusBadge" class="badge">Ready</span>
  <span id="warnBadge" class="badge" style="display:none;background:#3d2a1a;color:#e67e22;max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
  <span id="dateBadge" class="badge">--</span>
  <span id="spotBadge" class="badge">--</span>
</div>

<div class="main-layout">
  <div class="left-col">
    <div class="panel surface-panel">
      <div class="ptitle" id="surfaceTitle">Vol Surface - SPX</div>
      <div class="controls" id="surfaceModeControls">
        <label>Mode:</label>
        <div class="rg">
          <button class="rb active" id="btnRawIV" onclick="setMode('iv')">Raw IV</button>
          <button class="rb" id="btnSmoothIV" onclick="setMode('sv')">Smooth IV</button>
          <button class="rb" id="btnLV" onclick="setMode('lv')">Arb-free Local Vol</button>
        </div>
        <span style="font-size:10px;color:#8f96a3;margin-left:8px" id="surfaceModeHint"></span>
      </div>
      <div class="plot-wrap" style="background:transparent">
        <div id="surfacePlot" class="plot-target" style="background:transparent"></div>
        <div class="loading-overlay hidden" id="surfaceLoader">
          <div class="title">Loading SPX…</div>
          <div class="hint">Fetching options via Futu OpenD (127.0.0.1:11111).<br>First load may take 1–3 minutes.</div>
          <div class="elapsed" id="loadElapsed">0s</div>
        </div>
      </div>
    </div>

    <div class="panel" style="flex:0 0 auto">
      <div class="ptitle">Aux Vol Context (VVIX / SKEW / 3M Corr)</div>
      <div class="aux-strip" id="auxStrip">
        <div class="metric-card"><span class="label">VVIX</span><span class="val" id="vvixVal">--</span></div>
        <div class="metric-card"><span class="label">CBOE SKEW</span><span class="val" id="skewIdxVal">--</span></div>
        <div class="metric-card"><span class="label">3M Corr</span><span class="val" id="cor3mVal">--</span></div>
      </div>
    </div>

    <div class="panel price-regime-panel">
      <div class="ptitle" id="regimeTitle">HMM Signal Window - SPX</div>
      <div class="plot-wrap" style="min-height:360px">
        <div id="regimePlot" class="plot-target" style="min-height:360px"></div>
      </div>
    </div>
  </div>

  <div class="sidebar">
    <div class="panel" style="flex-shrink:0">
      <div class="ptitle">Regime & Levels</div>
      <div class="dashboard-grid">
        <div class="hmm-indicator inactive" id="hmmModeCard">
          <div style="font-size:9px;text-transform:uppercase;font-weight:600;opacity:.8">HMM Trading Signal</div>
          <div style="font-size:16px;font-weight:700;margin:2px 0" id="hmmSignalVal">--</div>
          <div style="font-size:9px;opacity:.9" id="hmmProbVal">--</div>
        </div>
        <div class="metric-card">
          <span class="label">ATM IV 30d</span>
          <span class="val" id="aivVal">--</span>
        </div>
        <div class="metric-card">
          <span class="label">25Δ Put Skew</span>
          <span class="val" id="pskVal">--</span>
        </div>
        <div class="metric-card">
          <span class="label">VRP (VIX − RV)</span>
          <span class="val" id="vraVal">--</span>
        </div>
        <div class="metric-card">
          <span class="label">Term Slope</span>
          <span class="val" id="tslVal">--</span>
        </div>
      </div>
    </div>

    <div class="panel condor-panel">
      <div class="ptitle">1-Sigma Vol Move Targets</div>
      <div class="condor-scroll">
        <table class="condor-table">
          <thead>
            <tr><th>Horizon</th><th>Implied</th><th>Historical</th><th>Spot Range</th></tr>
          </thead>
          <tbody id="condorTableBody">
            <tr><td colspan="4" style="text-align:center;color:#8f96a3">No data</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel" style="flex:1;min-height:220px">
      <div class="ptitle">Surface Anomaly List <span id="anomalyCountBadge" style="color:#2a6cff;font-weight:700"></span></div>
      <div class="summary-content" id="anomalyBox"></div>
    </div>

    <div class="panel" style="flex:0 0 auto;min-height:240px">
      <div class="ptitle">Volatility Historical Shading (RV vs IV)</div>
      <div class="plot-wrap" style="min-height:220px">
        <div id="volPlot" class="plot-target" style="min-height:220px"></div>
      </div>
    </div>
  </div>
</div>

<script>
const IDX = 'SPX';
let currentMode = 'iv';
let cache = null;
let loadJob = null;
let loadTimer = null;

function $(id){ return document.getElementById(id); }

function setStatus(text, color){
  const el = $('statusBadge');
  el.textContent = text;
  el.style.color = color || '#9aa0a8';
}

function showLoader(show){
  const loader = $('surfaceLoader');
  if (!loader) return;
  loader.classList.toggle('hidden', !show);
  if (show) {
    loader.querySelector('.title').textContent = 'Loading SPX…';
    const el = $('loadElapsed');
    const t0 = Date.now();
    if (loadTimer) clearInterval(loadTimer);
    loadTimer = setInterval(() => {
      el.textContent = Math.round((Date.now() - t0) / 1000) + 's';
    }, 500);
  } else if (loadTimer) {
    clearInterval(loadTimer);
    loadTimer = null;
  }
}

function showBlankState(){
  $('dateBadge').textContent = '--';
  $('spotBadge').textContent = 'SPX: --';
  $('surfaceTitle').textContent = 'Vol Surface - SPX';
  $('regimeTitle').textContent = 'HMM Signal Window - SPX';
  if (typeof Plotly !== 'undefined') {
    ['surfacePlot','regimePlot','volPlot'].forEach(id => {
      const el = $(id);
      if (el) Plotly.purge(el);
    });
  }
  ['vvixVal','skewIdxVal','cor3mVal','aivVal','pskVal','vraVal','tslVal'].forEach(id => {
    $(id).textContent = '--';
  });
  $('hmmSignalVal').textContent = '--';
  $('hmmProbVal').textContent = '--';
  $('hmmModeCard').className = 'hmm-indicator inactive';
  $('anomalyBox').innerHTML = '';
  $('anomalyCountBadge').textContent = '';
  $('condorTableBody').innerHTML = '<tr><td colspan="4" style="text-align:center;color:#8f96a3">--</td></tr>';
  $('warnBadge').style.display = 'none';
  showLoader(false);
}

function setRefreshEnabled(){
  $('refreshBtn').disabled = !!loadJob;
}

async function fetchIndex(force){
  if (force) {
    await fetch('/api/index/SPX/refresh', { method: 'POST', cache: 'no-store' }).catch(() => {});
    cache = null;
  }
  const res = await fetch('/api/index/SPX?async=1&_=' + Date.now(), { cache: 'no-store' });
  const data = await res.json();
  cache = data;
  return data;
}

async function pollIndex(force){
  if (loadJob) return loadJob;
  loadJob = (async () => {
    setRefreshEnabled();
    setStatus('Loading SPX…', '#2a6cff');
    showLoader(true);
    try {
      let data = await fetchIndex(force);
      while (data.status === 'loading') {
        await new Promise(r => setTimeout(r, 2000));
        data = await fetchIndex(false);
        if (data.status === 'error') break;
      }
      cache = data;
      renderAll();
      if (data.exists) setStatus('Ready · ' + new Date().toLocaleTimeString(), '#27ae60');
      else setStatus('Error', '#e74c3c');
      return data;
    } catch (err) {
      cache = { exists: false, status: 'error', error: err.message || String(err) };
      renderAll();
      setStatus('Error', '#e74c3c');
      return cache;
    } finally {
      loadJob = null;
      setRefreshEnabled();
      showLoader(false);
    }
  })();
  return loadJob;
}

function buildHighVolShapes(dates, regimes) {
  const shapes = [];
  let start = null;
  for (let i = 0; i < dates.length; i++) {
    const high = regimes[i] === 'high_vol';
    if (high && start === null) start = dates[i];
    else if (!high && start !== null) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: start, x1: dates[i - 1], y0: 0, y1: 1,
        fillcolor: 'rgba(255,0,0,0.25)', line: { width: 0 }, layer: 'below',
      });
      start = null;
    }
  }
  if (start !== null) {
    shapes.push({
      type: 'rect', xref: 'x', yref: 'paper',
      x0: start, x1: dates[dates.length - 1], y0: 0, y1: 1,
      fillcolor: 'rgba(255,0,0,0.25)', line: { width: 0 }, layer: 'below',
    });
  }
  return shapes;
}

function anomalyMarkers(data, mode) {
  const rows = (data.anomalies || []).filter(a => {
    if (a.ks == null || a.dte == null) return false;
    if (mode === 'lv') return a.surface === 'lv' || a.surface === 'structure';
    return a.surface === 'iv' || a.surface === 'structure';
  });
  if (!rows.length) return null;
  
  // To align the markers with the surface, we need to map the K/S and DTE back to the grid indices
  // and use the actual Z value from the surface at that point.
  let zData;
  if (mode === 'iv') {
    zData = data.surface_z;
  } else if (mode === 'sv') {
    zData = data.surface_sv || data.surface_z;
  } else {
    zData = data.surface_w;
  }

  const zVals = rows.map(a => {
    // Find the closest grid point
    let minDteIdx = 0;
    let minDteDiff = Infinity;
    for (let i = 0; i < data.surface_y.length; i++) {
        const diff = Math.abs(data.surface_y[i] - a.dte);
        if (diff < minDteDiff) {
            minDteDiff = diff;
            minDteIdx = i;
        }
    }

    let minKsIdx = 0;
    let minKsDiff = Infinity;
    for (let j = 0; j < data.surface_x.length; j++) {
        const diff = Math.abs(data.surface_x[j] - a.ks);
        if (diff < minKsDiff) {
            minKsDiff = diff;
            minKsIdx = j;
        }
    }
    
    // Use the actual surface value at this point, plus a small offset so it floats just above the surface
    const surfaceVal = zData[minDteIdx][minKsIdx];
    return surfaceVal + 0.5; 
  });

  return {
    type: 'scatter3d',
    mode: 'markers',
    x: rows.map(a => a.ks),
    y: rows.map(a => a.dte),
    z: zVals,
    marker: { size: 5, color: '#ff4d4f', symbol: 'diamond', line: { width: 1, color: '#fff' } },
    text: rows.map(a => a.kind + ': ' + a.detail),
    hovertemplate: '%{text}<br>K/S: %{x:.2f}<br>DTE: %{y:.0f}<extra></extra>',
    name: 'Anomalies',
  };
}

function renderSurfacePanel(data) {
  let z, title, colorscale;
  if (currentMode === 'iv') {
    z = data.surface_z; title = 'Raw Implied Vol (%)'; colorscale = 'Viridis';
  } else if (currentMode === 'sv') {
    z = data.surface_sv || data.surface_z; title = 'Smooth Implied Vol (%)'; colorscale = 'Cividis';
  } else {
    if (!data.local_vol_available) {
      $('surfacePlot').innerHTML = '<div class="error-box">Local volatility requires at least two live expiries.</div>';
      return;
    }
    z = data.surface_w; title = 'Local Vol (%)'; colorscale = 'Magma';
  }
  const zFlat = z.flat().filter(v => Number.isFinite(v));
  const zMax = zFlat.length ? Math.max.apply(null, zFlat) : 100;
  const rng = currentMode === 'lv' ? [0, zMax] : undefined;
  const singleExpiry = data.surface_y.length === 1;
  const traces = [];
  if (singleExpiry) {
    traces.push({
      type: 'scatter3d',
      mode: 'lines+markers',
      x: data.surface_x,
      y: data.surface_x.map(() => data.surface_y[0]),
      z: z[0],
      line: { color: '#2a6cff', width: 6 },
      marker: { color: z[0], colorscale: colorscale, size: 4, colorbar: { title: title } },
      hovertemplate: 'K/S: %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>',
    });
  } else {
    traces.push({
      type: 'surface',
      x: data.surface_x, y: data.surface_y, z: z,
      colorscale: colorscale,
      hovertemplate: 'K/S: %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>',
      colorbar: { title: title, titleside: 'right', x: 0.92, len: 0.7, bgcolor: 'rgba(0,0,0,0)', tickfont: { color: '#8f96a3' }, titlefont: { color: '#8f96a3' } },
      contours: { z: { show: true, usecolormap: true, highlightcolor: 'lime', project: { z: true } } },
      cmin: rng ? rng[0] : undefined, cmax: rng ? rng[1] : undefined,
    });
  }
  const marks = anomalyMarkers(data, currentMode);
  if (marks) traces.push(marks);

  Plotly.react('surfacePlot', traces, {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    scene: {
      xaxis: { title: { text: 'Moneyness (K/S)', font: { color: '#8f96a3' } }, gridcolor: '#222a3d', tickfont: { color: '#8f96a3' }, backgroundcolor: 'rgba(0,0,0,0)' },
      yaxis: { title: { text: 'DTE', font: { color: '#8f96a3' } }, gridcolor: '#222a3d', tickfont: { color: '#8f96a3' }, backgroundcolor: 'rgba(0,0,0,0)' },
      zaxis: { title: { text: title, font: { color: '#8f96a3' } }, gridcolor: '#222a3d', tickfont: { color: '#8f96a3' }, backgroundcolor: 'rgba(0,0,0,0)' },
      camera: { eye: { x: -1.5, y: -1.5, z: 0.8 } },
      aspectmode: 'manual', aspectratio: { x: 1.0, y: 1.2, z: 0.6 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    hoverlabel: { bgcolor: '#1c2030', font: { size: 12 } },
    uirevision: 'surface-SPX-' + currentMode,
  }, { displayModeBar: false, responsive: true });
}

function renderRegimePanel(data) {
  if (!data.hmm_dates || !data.hmm_dates.length) return;
  const dates = data.hmm_dates;
  const shapes = buildHighVolShapes(dates, data.hmm_regimes);
  if (data.hmm_today_date) {
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: data.hmm_today_date, x1: data.hmm_today_date, y0: 0, y1: 1,
      line: { color: '#2ca02c', width: 1.5, dash: 'dash' },
    });
  }

  const pToday = Number(data.hmm_prob_today).toFixed(1);
  const pTmr = Number(data.hmm_prob_tmr).toFixed(1);

  Plotly.react('regimePlot', [
    {
      x: dates, open: data.hmm_opens, high: data.hmm_highs, low: data.hmm_lows, close: data.hmm_prices,
      type: 'candlestick', name: 'SPX up/down', showlegend: true,
      increasing: { line: { color: '#2ca02c' }, fillcolor: '#2ca02c' },
      decreasing: { line: { color: '#d62728' }, fillcolor: '#d62728' },
    },
    {
      x: dates, y: data.hmm_prices, type: 'scatter', mode: 'lines', name: 'close',
      line: { color: '#d0d4dc', width: 1.1 }, opacity: 0.9,
    },
  ], {
    height: 380,
    margin: { l: 58, r: 16, t: 62, b: 40 },
    paper_bgcolor: '#131722', plot_bgcolor: '#131722',
    title: {
      text: '<b>SPX</b><br><sup style="color:#8f96a3">P(low vol today) = ' + pToday + '%  |  P(low vol tmr) = ' + pTmr + '%</sup>',
      font: { color: '#e8eaed', size: 16 }, x: 0.5, xanchor: 'center',
    },
    shapes: shapes,
    showlegend: true,
    legend: { orientation: 'h', y: 1.08, x: 0, font: { size: 9, color: '#8f96a3' }, bgcolor: 'rgba(0,0,0,0)' },
    hovermode: 'x unified',
    xaxis: { gridcolor: '#1f2330', type: 'date', tickfont: { color: '#8f96a3', size: 9 } },
    yaxis: { title: { text: 'SPX', font: { color: '#8f96a3', size: 11 } }, gridcolor: '#1f2330', tickfont: { color: '#8f96a3' } },
  }, { displayModeBar: false, responsive: true });
}

function renderVolPanel(data) {
  if (!data.hmm_dates || !data.hmm_dates.length) return;
  const dates = data.hmm_dates;
  const ivLabel = data.hmm_iv_label || 'VIX';
  const shapes = buildHighVolShapes(dates, data.hmm_regimes);

  const traces = [
    {
      x: dates, y: data.hmm_rv, type: 'scatter', mode: 'lines', name: '22d RV %',
      line: { color: '#1f77b4', width: 1.2 },
    },
    {
      x: dates, y: data.hmm_iv, type: 'scatter', mode: 'lines', name: ivLabel + ' IV',
      line: { color: '#ff7f0e', width: 1.2 },
    },
  ];
  if (Number.isFinite(data.hmm_iv_22d_ago)) {
    traces.push({
      x: [dates[0], dates[dates.length - 1]],
      y: [data.hmm_iv_22d_ago, data.hmm_iv_22d_ago],
      type: 'scatter', mode: 'lines', name: 'IV 22d ago',
      line: { color: '#888888', width: 1.2, dash: 'dot' },
    });
  }

  Plotly.react('volPlot', traces, {
    height: 240,
    margin: { l: 48, r: 12, t: 8, b: 32 },
    paper_bgcolor: '#131722', plot_bgcolor: '#131722',
    shapes: shapes,
    showlegend: true,
    legend: { orientation: 'h', y: 1.18, x: 0, font: { size: 9, color: '#8f96a3' }, bgcolor: 'rgba(0,0,0,0)' },
    hovermode: 'x unified',
    xaxis: { gridcolor: '#1f2330', type: 'date', tickfont: { color: '#8f96a3', size: 9 } },
    yaxis: { title: { text: 'Annualized vol %', font: { color: '#8f96a3', size: 10 } }, gridcolor: '#1f2330', tickfont: { color: '#8f96a3', size: 9 } },
  }, { displayModeBar: false, responsive: true });
}

function fmtLevel(v, digits) {
  if (v == null || !Number.isFinite(Number(v))) return '--';
  return Number(v).toFixed(digits);
}

function renderAnomalies(data) {
  const box = $('anomalyBox');
  const rows = data.anomalies || [];
  $('anomalyCountBadge').textContent = rows.length ? '(' + rows.length + ')' : '';
  if (!rows.length) {
    box.innerHTML = '<div style="color:#8f96a3;font-size:12px">No surface anomalies flagged on this session.</div>';
    return;
  }
  box.innerHTML = '';
  rows.forEach(a => {
    const div = document.createElement('div');
    div.className = 'anomaly-row';
    const loc = (a.ks != null && a.dte != null)
      ? `K/S ${Number(a.ks).toFixed(2)} · DTE ${Number(a.dte).toFixed(0)}d`
      : 'structure-level';
    const val = a.value != null ? ` · val ${Number(a.value).toFixed(1)}` : '';
    div.innerHTML =
      `<div class="kind"><span class="kind-tag kind-${a.kind}">${a.kind}</span>${a.surface}</div>` +
      `<div class="meta">${loc}${val} · score ${Number(a.score).toFixed(1)}</div>` +
      `<div class="detail">${a.detail || ''}</div>`;
    box.appendChild(div);
  });
}

function renderAll(){
  const data = cache;
  if (!data || !data.exists) {
    showBlankState();
    if (data?.status === 'error') {
      $('surfacePlot').innerHTML = '<div class="error-box">SPX: ' + (data.error || 'failed') + '</div>';
    } else if (data?.status === 'loading') {
      showLoader(true);
    }
    return;
  }

  const warnBadge = $('warnBadge');
  if (data.warnings && data.warnings.length) {
    warnBadge.style.display = 'inline-block';
    warnBadge.textContent = '⚠ ' + data.warnings.join(' | ');
    warnBadge.title = data.warnings.join('\n');
  } else {
    warnBadge.style.display = 'none';
    warnBadge.title = '';
  }

  $('dateBadge').textContent = 'Date: ' + data.date;
  $('spotBadge').textContent = 'SPX: ' + Number(data.spot).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  $('surfaceTitle').textContent = 'Vol Surface - SPX';
  renderSurfacePanel(data);

  const aux = data.aux || {};
  $('vvixVal').textContent = fmtLevel(aux.vvix, 1) + (aux.vvix_pctl != null ? ` (${Number(aux.vvix_pctl).toFixed(0)}%ile)` : '');
  $('skewIdxVal').textContent = fmtLevel(aux.skew_index, 1) + (aux.skew_pctl != null ? ` (${Number(aux.skew_pctl).toFixed(0)}%ile)` : '');
  $('cor3mVal').textContent = fmtLevel(aux.cor3m, 1) + (aux.cor3m_pctl != null ? ` (${Number(aux.cor3m_pctl).toFixed(0)}%ile)` : '');

  $('aivVal').textContent = fmtLevel(data.aiv, 1) + '%';
  $('pskVal').textContent = fmtLevel(data.psk, 1) + ' pts';
  $('vraVal').textContent = fmtLevel(data.vrp, 1) + ' pts';
  $('tslVal').textContent = fmtLevel(data.tsl, 1) + ' pts';

  const hmmCard = $('hmmModeCard');
  if (data.hmm_signal) {
    hmmCard.className = 'hmm-indicator active';
    $('hmmSignalVal').textContent = 'BUY / LONG';
    $('hmmSignalVal').style.color = '#00cc66';
  } else {
    hmmCard.className = 'hmm-indicator inactive';
    $('hmmSignalVal').textContent = 'NEUTRAL / CASH';
    $('hmmSignalVal').style.color = '#e74c3c';
  }
  $('hmmProbVal').textContent = 'P(Calm today): ' + data.hmm_prob_today + '% | P(Calm tmr): ' + data.hmm_prob_tmr + '%';

  const tbody = $('condorTableBody');
  tbody.innerHTML = '';
  if (data.hmm_move_table && data.hmm_move_table.length) {
    data.hmm_move_table.slice(0, 4).forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td style="font-weight:bold;color:#fff">${row.horizon}</td>
        <td style="color:#00cc66">${row.implied}</td>
        <td style="color:#8f96a3">${row.historical}</td>
        <td style="color:#2a6cff;font-weight:bold">${row.spot_implied}</td>`;
      tbody.appendChild(tr);
    });
  } else {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#8f96a3">No move matrix</td></tr>';
  }

  renderAnomalies(data);
  renderRegimePanel(data);
  renderVolPanel(data);
}

async function refreshCurrent(){
  showBlankState();
  await pollIndex(true);
}

function setMode(mode){
  currentMode = mode;
  $('btnRawIV').className = 'rb' + (mode === 'iv' ? ' active' : '');
  $('btnSmoothIV').className = 'rb' + (mode === 'sv' ? ' active' : '');
  $('btnLV').className = 'rb' + (mode === 'lv' ? ' active' : '');
  if (cache?.exists) renderSurfacePanel(cache);
}

(function boot(){
  showBlankState();
  pollIndex(false);
})();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(_sanitize_for_json(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._send_html(HTML)
            return

        if path.startswith("/api/index/"):
            index_name = path.split("/")[-1].upper()
            if index_name != INDEX_NAME:
                self.send_error(404)
                return
            try:
                qs = urllib.parse.parse_qs(parsed.query)
                async_mode = qs.get("async", ["0"])[0] in ("1", "true", "yes")
                payload = store.get(INDEX_NAME, async_mode=async_mode)
                code = 200 if payload.get("exists") else (202 if async_mode else 200)
                self._send_json(code, payload)
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"exists": False, "status": "error", "error": str(exc)})
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.endswith("/refresh") and path.startswith("/api/index/"):
            parts = path.split("/")
            if len(parts) >= 4:
                index_name = parts[3].upper()
                if index_name == INDEX_NAME:
                    store.invalidate(INDEX_NAME)
                    self._send_json(200, {"ok": True, "index": INDEX_NAME})
                    return

        self.send_error(404)

    def log_message(self, _format, *_args) -> None:
        return


def _warm() -> None:
    time.sleep(0.5)
    store.get(INDEX_NAME, async_mode=True)


if __name__ == "__main__":
    print(f"\n  >>> SPX Surface Anomaly Hub  http://127.0.0.1:{PORT}  <<<\n")
    print("  Preloading SPX. Futu OpenD: 127.0.0.1:11111\n")

    threading.Thread(target=_warm, name="warm-spx", daemon=True).start()

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        server = ThreadedHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        print(f"  ERROR: port {PORT} in use — stop the old app first.\n  {exc}\n")
        raise SystemExit(1) from exc

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.\n")
