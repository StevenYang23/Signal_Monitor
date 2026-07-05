"""Index Quant Signal Hub — lightweight HTTP dashboard."""

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
    build_structure_metrics_insights,
    classify_sentiment,
    compare_features,
    deepseek_enhance_structure_insights,
    detect_term_hump,
    dupire_local_vol,
    smooth_iv_grid_quadratic,
    fetch_anchor_iv_histories,
    fetch_spot_yfinance,
    fetch_vix_context,
)
from surface_sentiment import SurfaceDeltaPCA, SurfacePCAConfig  # noqa: E402
from volatility_regime import HMMVolatilityRegime  # noqa: E402

PORT = int(os.environ.get("PORT", 8050))
USE_DEEPSEEK = os.environ.get("USE_DEEPSEEK", "1").lower() not in ("0", "false", "no", "off")
INDEX_NAMES = ("SPX", "NDX", "DJI")
OPTION_MAP = {"SPX": "US..SPX", "NDX": "US..NDX", "DJI": "US.DIA"}
SPOT_MAP = {"SPX": "^SPX", "NDX": "^NDX", "DJI": "^DJI"}
HMM_MAP = {"SPX": "SPX", "NDX": "NSDQ", "DJI": "DJI"}


def _futu_reachable(host: str = "127.0.0.1", port: int = 11111, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _prepare_study(study: VolSurfaceStudy, cfg: VolSurfaceConfig, *, futu_up: bool) -> None:
    if futu_up:
        study.ensure_history(min_sessions=2, fetch_live=True, demo_backfill=True)
        return
    print(
        f"[pipeline] Futu OpenD offline ({cfg.host}:{cfg.port}) — using cache/demo surfaces",
        flush=True,
    )
    study.ensure_history(min_sessions=2, fetch_live=False, demo_backfill=True)


def _analyze_study(study: VolSurfaceStudy, *, futu_up: bool) -> dict:
    dates = sorted(study.features.keys())
    today_d = dates[-1]
    today = study.features[today_d]
    hist = [study.features[d] for d in dates[:-1]]
    changes = compare_features(today, hist[-5:] if len(hist) >= 5 else hist)
    anomalies = detect_term_hump(today, study.cfg)
    vix_ctx = fetch_vix_context(study.cfg.lookback_days)
    if futu_up:
        anchor_ctx = fetch_anchor_iv_histories(study.surfaces[today_d], study.cfg, query_time_period=1)
    else:
        anchor_ctx = {"anchors": {}, "series": pd.DataFrame(), "changes": {}, "meta": {"skipped": True}}
    sentiment = classify_sentiment(today, changes, anomalies, vix_ctx, anchor_ctx)
    return {
        "today": today,
        "changes": changes,
        "anomalies": anomalies,
        "vix_context": vix_ctx,
        "anchor_context": anchor_ctx,
        "sentiment": sentiment,
        "dates": dates,
    }


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


class QuantEngine:
    def process_index(self, short_name: str) -> dict:
        if short_name not in OPTION_MAP:
            raise ValueError(f"Unknown index: {short_name}")
        print(f"[pipeline] {short_name} start", flush=True)
        try:
            payload = self._build_payload(short_name)
            print(f"[pipeline] {short_name} OK", flush=True)
            return payload
        except Exception as exc:
            print(f"[pipeline] {short_name} failed: {exc}", flush=True)
            traceback.print_exc()
            return {"exists": False, "error": str(exc)}

    def _build_payload(self, short_name: str) -> dict:
        option_ticker = OPTION_MAP[short_name]
        warnings_list: list[str] = []

        cfg = VolSurfaceConfig(underlying=option_ticker, max_dte=60, lookback_days=15)
        futu_up = _futu_reachable(cfg.host, cfg.port)
        study = VolSurfaceStudy(cfg)
        _prepare_study(study, cfg, futu_up=futu_up)

        dates = sorted(study.surfaces.keys())
        today_d = dates[-1]
        df_today = study.surfaces[today_d]
        option_spot = float(df_today["spot"].iloc[0])
        try:
            spot, _ = fetch_spot_yfinance(SPOT_MAP[short_name])
        except Exception:
            spot = option_spot

        g_dte, g_ks, iv_g = build_iv_grid(df_today, max_dte=cfg.max_dte)

        x_ks = [float(v) for v in g_ks[0, :]]
        y_dte = [float(v) for v in g_dte[:, 0]]
        iv_data = [[float(v) for v in row] for row in iv_g]
        iv_g_smooth = smooth_iv_grid_quadratic(df_today, g_dte, g_ks, iv_g)
        lv = dupire_local_vol(option_spot, g_dte, g_ks, iv_g_smooth, r=cfg.risk_free_rate)
        lv_data = [[float(v) for v in row] for row in lv]

        sv_data = [[float(v) for v in row] for row in iv_g_smooth]
        result = _analyze_study(study, futu_up=futu_up)
        today_f = result["today"]

        pca_score_history = None
        today_scores = [0.0, 0.0, 0.0, 0.0]
        sentiment = {"joint_flags": [], "z_scores": []}
        if len(study.surfaces) >= 2:
            try:
                pca = SurfaceDeltaPCA(SurfacePCAConfig(n_components=4, baseline_window=21))
                pca_result = pca.fit(surfaces=study.surfaces)
                pca_score_history = pca_result["score_history"]
                hist_scores = pca_result["score_history"]
                today_scores = hist_scores[-1].tolist()
                sentiment = pca.sentiment_from_scores(hist_scores[-1], hist_scores)
            except ValueError as pca_exc:
                msg = f"PCA skipped: {pca_exc}"
                warnings_list.append(msg)
                print(f"[pipeline] {short_name} PCA: {pca_exc}", flush=True)

        vix = float(result.get("vix_context", {}).get("vix", 15.0))
        aiv = float(today_f.get("atm_iv_30d", np.nan))
        psk = float(today_f.get("skew_25d", np.nan))
        tsl = float(today_f.get("term_slope", np.nan))
        bfly = float(today_f.get("butterfly_25d", np.nan))

        if result["changes"] is not None and not result["changes"].empty and "delta_5d" in result["changes"].columns:
            chg = result["changes"]

            def d5(name):
                row = chg.loc[chg["feature"] == name]
                if row.empty:
                    return 0.0
                val = row["delta_5d"].iloc[0]
                return float(val) if np.isfinite(val) else 0.0

            d_skew = d5("skew_25d")
            d_bfly = d5("butterfly_25d")
        else:
            d_skew = 0.0
            d_bfly = 0.0

        vrp_val = result.get("vix_context", {}).get("vrp", vix - aiv)
        anchor_ctx = result.get("anchor_context", {})
        anchor_changes = anchor_ctx.get("changes", {})
        skew_proxy_chg = anchor_changes.get("skew_proxy", 0.0)

        ps_score = np.clip(-psk * 4.0, -35, 35)
        ts_score = np.clip(tsl * 5.0, -25, 25)
        vix_score = np.clip((20.0 - vix) * 2.0, -20, 20)
        vrp_score = np.clip((6.0 - vrp_val) * 2.5, -15, 15)
        anchor_score = np.clip(-skew_proxy_chg * 5.0, -10, 10)
        sentiment_score = float(np.clip(ps_score + ts_score + vix_score + vrp_score + anchor_score, -100, 100))

        structure_metrics = build_structure_metrics_insights(
            today=today_f,
            vix_ctx=result.get("vix_context"),
            anchor_ctx=anchor_ctx,
            today_scores=today_scores,
            pca_sentiment=sentiment,
            pca_score_history=pca_score_history,
            changes=result.get("changes"),
        )
        if USE_DEEPSEEK:
            structure_metrics = deepseek_enhance_structure_insights(
                structure_metrics,
                context={
                    "index": short_name,
                    "date": str(today_d),
                    "spot": spot,
                    "term_slope": tsl,
                    "vrp": vrp_val,
                },
            )

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
            hmm_model = HMMVolatilityRegime.from_market(HMM_MAP[short_name], signal_mode=True)
            hmm_model.download_data()
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
            print(f"[pipeline] {short_name} HMM: {hmm_exc}", flush=True)

        return {
            "exists": True,
            "warnings": warnings_list,
            "date": str(today_d),
            "spot": spot,
            "aiv": aiv,
            "vix": vix,
            "psk": psk,
            "d_skew": d_skew,
            "tsl": tsl,
            "bfly": bfly,
            "d_bfly": d_bfly,
            "vrp": vrp_val,
            "pc1": today_scores[0],
            "pc2": today_scores[1],
            "score": round(sentiment_score, 1),
            "structure_metrics": structure_metrics,
            "surface_x": x_ks,
            "surface_y": y_dte,
            "surface_z": iv_data,
            "surface_sv": sv_data,
            "surface_w": lv_data,
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
    """Background loader with in-memory cache and per-index workers."""

    def __init__(self, engine: QuantEngine):
        self.engine = engine
        self._cache: dict[str, dict] = {}
        self._state: dict[str, dict] = {}
        self._locks: dict[str, threading.Lock] = {name: threading.Lock() for name in INDEX_NAMES}
        self._cond = threading.Condition()

    def _ready_payload(self, index_name: str) -> dict:
        out = dict(self._cache[index_name])
        out["status"] = "ready"
        return out

    def _loading_payload(self, index_name: str) -> dict:
        with self._cond:
            st = self._state.get(index_name, {})
        return {
            "exists": False,
            "status": "loading",
            "index": index_name,
            "started_at": st.get("started_at"),
            "elapsed_sec": round(time.time() - float(st.get("started_mono", time.time())), 1),
        }

    def _kickoff(self, index_name: str) -> None:
        index_name = index_name.upper()
        with self._cond:
            if index_name in self._cache:
                return
            st = self._state.get(index_name)
            if st and st.get("status") in ("loading", "error"):
                return

        if not self._locks[index_name].acquire(blocking=False):
            return

        with self._cond:
            self._state[index_name] = {
                "status": "loading",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "started_mono": time.monotonic(),
            }
            self._cond.notify_all()

        threading.Thread(
            target=self._run_worker,
            args=(index_name,),
            name=f"load-{index_name}",
            daemon=True,
        ).start()

    def get(self, index_name: str, *, async_mode: bool = False, timeout: float = 600.0) -> dict:
        index_name = index_name.upper()
        if index_name not in INDEX_NAMES:
            raise ValueError(f"Unknown index: {index_name}")

        self._kickoff(index_name)

        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                if index_name in self._cache:
                    return self._ready_payload(index_name)
                st = self._state.get(index_name)
                if st and st.get("status") == "error":
                    return {
                        "exists": False,
                        "status": "error",
                        "index": index_name,
                        "error": st.get("error", "pipeline failed"),
                    }
                if async_mode:
                    return self._loading_payload(index_name)

            if time.monotonic() >= deadline:
                return {
                    "exists": False,
                    "status": "error",
                    "index": index_name,
                    "error": f"Timed out after {int(timeout)}s",
                }
            with self._cond:
                self._cond.wait(timeout=1.0)

    def invalidate(self, index_name: str) -> None:
        index_name = index_name.upper()
        with self._cond:
            self._cache.pop(index_name, None)
            self._state.pop(index_name, None)
            self._cond.notify_all()

    def _run_worker(self, index_name: str) -> None:
        try:
            payload = self.engine.process_index(index_name)
            with self._cond:
                if payload.get("exists"):
                    self._cache[index_name] = payload
                    self._state[index_name] = {"status": "ready"}
                else:
                    self._state[index_name] = {
                        "status": "error",
                        "error": payload.get("error", "pipeline failed"),
                    }
                self._cond.notify_all()
        finally:
            self._locks[index_name].release()


engine = QuantEngine()
store = IndexDataStore(engine)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Index Quant Signal Hub v3.2</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0e1118;color:#d0d4dc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;height:100vh;display:flex;flex-direction:column}
.header{padding:14px 24px;border-bottom:1px solid #1f2330;display:flex;align-items:center;background:#131722;gap:12px;z-index:10;flex-wrap:wrap}
.header h1{font-size:18px;font-weight:600;color:#e8eaed;letter-spacing:.3px}
.header-tabs{display:flex;background:#1a1f2c;border-radius:6px;padding:2px;margin-left:8px}
.tab-btn{background:transparent;border:none;color:#8f96a3;padding:6px 16px;font-size:12px;font-weight:600;border-radius:4px;cursor:pointer;transition:all .15s}
.tab-btn.active{background:#2a6cff;color:#fff}
.tab-btn.loading{opacity:.55}
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
.dashboard-grid{display:grid;grid-template-columns:1fr;gap:10px;padding:16px}
.metric-card{background:#181d2e;border-radius:6px;border:1px solid #232a3f;padding:10px;display:flex;flex-direction:column;justify-content:center}
.metric-card .label{font-size:9px;color:#8f96a3;text-transform:uppercase;font-weight:600;margin-bottom:3px}
.metric-card .val{font-size:16px;font-weight:700;color:#fff}
.hmm-indicator{border-radius:6px;padding:10px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;transition:all .3s}
.hmm-indicator.active{background:rgba(0,204,102,.1);border:1.5px solid #00cc66}
.hmm-indicator.inactive{background:rgba(231,76,60,.1);border:1.5px solid #e74c3c}
.summary-content{padding:12px 16px;font-size:12px;line-height:1.5;overflow-y:auto;flex:1}
.metric-block{margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #1c2030}
.metric-block:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
.metric-block .metric-line{color:#e8eaed;font-weight:600;font-size:12px;margin-bottom:6px;line-height:1.4}
.metric-block .insight-line{color:#9aa3b2;font-size:11.5px;line-height:1.55}
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
.blank-hint{color:#8f96a3;font-size:12px;text-align:center;padding:24px}
</style>
</head>
<body>
<div class="header">
  <h1>Index Quant Signal Hub</h1>
  <div class="header-tabs">
    <button class="tab-btn active" data-idx="SPX" onclick="switchIndex('SPX')">SPX</button>
    <button class="tab-btn" data-idx="NDX" onclick="switchIndex('NDX')">NDX</button>
    <button class="tab-btn" data-idx="DJI" onclick="switchIndex('DJI')">DJI</button>
  </div>
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

    <div class="panel" style="flex:0 0 auto;padding:12px 16px">
      <div class="ptitle" style="padding:0 0 10px;border-bottom:none">Quant Sentiment Compass</div>
      <div id="horizontalCompassContainer" style="min-height:52px;margin-top:4px"></div>
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
      <div class="ptitle">Regime & Sentiment</div>
      <div class="dashboard-grid">
        <div class="hmm-indicator inactive" id="hmmModeCard">
          <div style="font-size:9px;text-transform:uppercase;font-weight:600;opacity:.8">HMM Trading Signal</div>
          <div style="font-size:16px;font-weight:700;margin:2px 0" id="hmmSignalVal">--</div>
          <div style="font-size:9px;opacity:.9" id="hmmProbVal">--</div>
        </div>
        <div class="metric-card">
          <span class="label">Volatility Risk Premium (VRP)</span>
          <span class="val" id="vraVal">--</span>
        </div>
        <div class="metric-card">
          <span class="label">Term Structure Roll Spread</span>
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

    <div class="panel" style="flex:1;min-height:200px">
      <div class="ptitle">Quantitative Structure Metrics</div>
      <div class="summary-content" id="bulletsBox"></div>
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
const ALL = ['SPX','NDX','DJI'];
let currentIdx = 'SPX';
let currentMode = 'iv';
let cache = {};
let loadJobs = {};

function $(id){ return document.getElementById(id); }

function setStatus(text, color){
  const el = $('statusBadge');
  el.textContent = text;
  el.style.color = color || '#9aa0a8';
}

function showLoader(show, idx){
  const loader = $('surfaceLoader');
  if (!loader) return;
  loader.classList.toggle('hidden', !show);
  if (show) {
    loader.querySelector('.title').textContent = 'Loading ' + (idx || currentIdx) + '…';
  }
}

function showBlankState(){
  $('dateBadge').textContent = '--';
  $('spotBadge').textContent = currentIdx + ': --';
  $('surfaceTitle').textContent = 'Vol Surface - ' + currentIdx;
  $('regimeTitle').textContent = 'HMM Signal Window - ' + currentIdx;
  if (typeof Plotly !== 'undefined') {
    ['surfacePlot','regimePlot','volPlot'].forEach(id => {
      const el = $(id);
      if (el) Plotly.purge(el);
    });
  }
  $('horizontalCompassContainer').innerHTML = '';
  $('vraVal').textContent = '--';
  $('tslVal').textContent = '--';
  $('hmmSignalVal').textContent = '--';
  $('hmmProbVal').textContent = '--';
  $('hmmModeCard').className = 'hmm-indicator inactive';
  $('bulletsBox').innerHTML = '';
  $('condorTableBody').innerHTML = '<tr><td colspan="4" style="text-align:center;color:#8f96a3">--</td></tr>';
  $('warnBadge').style.display = 'none';
  showLoader(false);
}

function updateStatusBadge(){
  const parts = ALL.map(i => {
    const d = cache[i];
    if (d?.exists) return i + ' ✓';
    if (d?.status === 'error') return i + ' ✗';
    if (d?.status === 'loading' || loadJobs[i]) return i + ' …';
    return i + ' …';
  });
  const allReady = ALL.every(i => cache[i]?.exists);
  setStatus(
    allReady ? ('All ready · ' + new Date().toLocaleTimeString()) : parts.join(' | '),
    allReady ? '#27ae60' : '#2a6cff'
  );
}

function setTabLoading(idx, loading){
  document.querySelectorAll('.tab-btn').forEach(btn => {
    if (btn.dataset.idx === idx) btn.classList.toggle('loading', loading);
  });
}

function setRefreshEnabled(){
  $('refreshBtn').disabled = !!loadJobs[currentIdx];
}

async function fetchIndex(idx, force){
  if (force) {
    await fetch('/api/index/' + idx + '/refresh', { method: 'POST', cache: 'no-store' }).catch(() => {});
    delete cache[idx];
  }
  const res = await fetch('/api/index/' + idx + '?async=1&_=' + Date.now(), { cache: 'no-store' });
  const data = await res.json();
  cache[idx] = data;
  return data;
}

async function pollIndexInBackground(idx, force){
  if (loadJobs[idx]) return loadJobs[idx];
  loadJobs[idx] = (async () => {
    setTabLoading(idx, true);
    if (idx === currentIdx) setRefreshEnabled();
    updateStatusBadge();
    try {
      let data = await fetchIndex(idx, force);
      while (data.status === 'loading') {
        await new Promise(r => setTimeout(r, 2000));
        data = await fetchIndex(idx, false);
        if (data.status === 'error') break;
      }
      cache[idx] = data;
      if (idx === currentIdx) renderAll();
      updateStatusBadge();
      return data;
    } catch (err) {
      cache[idx] = { exists: false, status: 'error', error: err.message || String(err) };
      if (idx === currentIdx) renderAll();
      updateStatusBadge();
      return cache[idx];
    } finally {
      delete loadJobs[idx];
      setTabLoading(idx, false);
      if (idx === currentIdx) setRefreshEnabled();
    }
  })();
  return loadJobs[idx];
}

async function preloadAllIndices(force){
  await Promise.all(ALL.map(idx => pollIndexInBackground(idx, force)));
}

function getScoreLabel(score){
  if (score > 50) return 'Extremely Bullish';
  if (score > 15) return 'Slightly Bullish';
  if (score > -15) return 'Neutral';
  if (score > -50) return 'Slightly Bearish';
  return 'Extremely Bearish';
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

function renderSurfacePanel(data, idx) {
  let z, title, colorscale;
  if (currentMode === 'iv') {
    z = data.surface_z; title = 'Raw Implied Vol (%)'; colorscale = 'Viridis';
  } else if (currentMode === 'sv') {
    z = data.surface_sv || data.surface_z; title = 'Smooth Implied Vol (%)'; colorscale = 'Cividis';
  } else {
    z = data.surface_w; title = 'Local Vol (%)'; colorscale = 'Magma';
  }
  const zFlat = z.flat();
  const zMax = zFlat.length ? Math.max.apply(null, zFlat) : 100;
  const rng = currentMode === 'lv' ? [0, zMax] : undefined;

  Plotly.react('surfacePlot', [{
    type: 'surface',
    x: data.surface_x, y: data.surface_y, z: z,
    colorscale: colorscale,
    hovertemplate: 'K/S: %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>',
    colorbar: { title: title, titleside: 'right', x: 0.92, len: 0.7, bgcolor: 'rgba(0,0,0,0)', tickfont: { color: '#8f96a3' }, titlefont: { color: '#8f96a3' } },
    contours: { z: { show: true, usecolormap: true, highlightcolor: 'lime', project: { z: true } } },
    cmin: rng ? rng[0] : undefined, cmax: rng ? rng[1] : undefined,
  }], {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      xaxis: { title: { text: 'Moneyness (K/S)', font: { color: '#8f96a3' } }, gridcolor: '#222a3d', tickfont: { color: '#8f96a3' }, backgroundcolor: 'rgba(0,0,0,0)' },
      yaxis: { title: { text: 'DTE', font: { color: '#8f96a3' } }, gridcolor: '#222a3d', tickfont: { color: '#8f96a3' }, backgroundcolor: 'rgba(0,0,0,0)' },
      zaxis: { title: { text: title, font: { color: '#8f96a3' } }, gridcolor: '#222a3d', tickfont: { color: '#8f96a3' }, backgroundcolor: 'rgba(0,0,0,0)' },
      camera: { eye: { x: -1.5, y: -1.5, z: 0.8 } },
      aspectmode: 'manual', aspectratio: { x: 1.0, y: 1.2, z: 0.6 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    hoverlabel: { bgcolor: '#1c2030', font: { size: 12 } },
    uirevision: 'surface-' + idx + '-' + currentMode,
  }, { displayModeBar: false, responsive: true });
}

function renderRegimePanel(data, idx) {
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
      type: 'candlestick', name: idx + ' up/down', showlegend: true,
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
      text: '<b>' + idx + '</b><br><sup style="color:#8f96a3">P(low vol today) = ' + pToday + '%  |  P(low vol tmr) = ' + pTmr + '%</sup>',
      font: { color: '#e8eaed', size: 16 }, x: 0.5, xanchor: 'center',
    },
    shapes: shapes,
    showlegend: true,
    legend: { orientation: 'h', y: 1.08, x: 0, font: { size: 9, color: '#8f96a3' }, bgcolor: 'rgba(0,0,0,0)' },
    hovermode: 'x unified',
    xaxis: { gridcolor: '#1f2330', type: 'date', tickfont: { color: '#8f96a3', size: 9 } },
    yaxis: { title: { text: idx, font: { color: '#8f96a3', size: 11 } }, gridcolor: '#1f2330', tickfont: { color: '#8f96a3' } },
  }, { displayModeBar: false, responsive: true });

  document.getElementById('regimeTitle').textContent = 'HMM Signal Window - ' + idx;
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

function renderAll(){
  const data = cache[currentIdx];
  if (!data || !data.exists) {
    showBlankState();
    if (data?.status === 'error') {
      $('surfacePlot').innerHTML = '<div class="error-box">' + currentIdx + ': ' + (data.error || 'failed') + '</div>';
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
  $('spotBadge').textContent = currentIdx + ': ' + Number(data.spot).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  $('surfaceTitle').textContent = 'Vol Surface - ' + currentIdx;
  renderSurfacePanel(data, currentIdx);

  const score = data.score;
  const compassPct = ((score + 100) / 200) * 100;
  let scoreColor = '#f1c40f';
  if (score > 50) scoreColor = '#27ae60';
  else if (score > 15) scoreColor = '#2ecc71';
  else if (score < -50) scoreColor = '#e74c3c';
  else if (score < -15) scoreColor = '#e67e22';

  $('horizontalCompassContainer').innerHTML = `
    <div style="position:relative;width:100%;padding-top:20px">
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#8f96a3;margin-bottom:6px;font-weight:500">
        <span>Extremely Bearish (-100)</span><span>Neutral (0)</span><span>Extremely Bullish (100)</span>
      </div>
      <div style="position:relative;height:16px;background:linear-gradient(90deg,#e74c3c 0%,#e67e22 25%,#f1c40f 50%,#2ecc71 75%,#27ae60 100%);border-radius:8px">
        <div style="position:absolute;top:-3px;left:${compassPct}%;width:6px;height:22px;background:#fff;border-radius:3px;transform:translateX(-50%)"></div>
        <div style="position:absolute;top:-34px;left:${compassPct}%;background:#2a6cff;color:#fff;font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;transform:translateX(-50%);white-space:nowrap">
          Score: ${score >= 0 ? '+' : ''}${score}/100 (<span style="color:${scoreColor}">${getScoreLabel(score)}</span>)
        </div>
      </div>
    </div>`;

  $('vraVal').textContent = Number(data.vrp).toFixed(1) + ' pts';
  $('tslVal').textContent = Number(data.tsl).toFixed(1) + ' vol pts';

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

  const bulletsBox = $('bulletsBox');
  bulletsBox.innerHTML = '';
  (data.structure_metrics || []).forEach(block => {
    const div = document.createElement('div');
    div.className = 'metric-block';
    div.innerHTML = `<div class="metric-line">${block.metric}</div><div class="insight-line">${block.insight}</div>`;
    bulletsBox.appendChild(div);
  });
  if (!data.structure_metrics || !data.structure_metrics.length) {
    bulletsBox.innerHTML = '<div style="color:#8f96a3;font-size:12px">No structure metrics.</div>';
  }

  renderRegimePanel(data, currentIdx);
  renderVolPanel(data);
}

function switchIndex(idx){
  currentIdx = idx;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.idx === idx);
  });
  setRefreshEnabled();
  renderAll();
  if (!cache[idx]?.exists && !loadJobs[idx]) {
    pollIndexInBackground(idx, false);
  }
}

async function refreshCurrent(){
  showBlankState();
  await pollIndexInBackground(currentIdx, true);
}

function setMode(mode){
  currentMode = mode;
  $('btnRawIV').className = 'rb' + (mode === 'iv' ? ' active' : '');
  $('btnSmoothIV').className = 'rb' + (mode === 'sv' ? ' active' : '');
  $('btnLV').className = 'rb' + (mode === 'lv' ? ' active' : '');
  renderAll();
}

(function boot(){
  showBlankState();
  updateStatusBadge();
  ALL.forEach(idx => pollIndexInBackground(idx, false));
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
            if index_name not in INDEX_NAMES:
                self.send_error(404)
                return
            try:
                qs = urllib.parse.parse_qs(parsed.query)
                async_mode = qs.get("async", ["0"])[0] in ("1", "true", "yes")
                payload = store.get(index_name, async_mode=async_mode)
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
                if index_name in INDEX_NAMES:
                    store.invalidate(index_name)
                    self._send_json(200, {"ok": True, "index": index_name})
                    return

        self.send_error(404)

    def log_message(self, _format, *_args) -> None:
        return


def _warm_all() -> None:
    time.sleep(0.5)
    for name in INDEX_NAMES:
        store.get(name, async_mode=True)


if __name__ == "__main__":
    print(f"\n  >>> Index Quant Hub  http://127.0.0.1:{PORT}  <<<\n")
    print(
        "  Preloading SPX + NDX + DJI in parallel. Futu OpenD: 127.0.0.1:11111\n"
        f"  DeepSeek insights: {'ON' if USE_DEEPSEEK else 'OFF'}  (set USE_DEEPSEEK=0 to disable)\n"
    )

    threading.Thread(target=_warm_all, name="warm-all", daemon=True).start()

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        server = ThreadedHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        print(f"  ERROR: port {PORT} in use — stop the old app.py first.\n  {exc}\n")
        raise SystemExit(1) from exc

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
