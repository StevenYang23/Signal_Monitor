import os, sys, glob, warnings, json, http.server, urllib.parse, traceback, time
from pathlib import Path
from datetime import date
warnings.filterwarnings("ignore")

# Configure temporary configurations for matplotlib in headless deployment environments
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__) or ".", ".matplotlib_temp")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__) or ".")
from vol_surface import VolSurfaceConfig, VolSurfaceStudy, build_iv_grid, dupire_local_vol, fetch_spot_yfinance
from surface_sentiment import SurfaceDeltaPCA, SurfacePCAConfig
from volatility_regime import HMMVolatilityRegime

PORT = int(os.environ.get("PORT", 8050))

# ----------------- Quant Pipeline Handler -----------------
class QuantEngine:
    def __init__(self):
        # Memory caching disabled to ensure fresh reads / live updates on every refresh
        pass

    def run_pipeline(self) -> dict:
        # Run pipeline over SPX, NDX (IXIC tab), DJI
        # NDX/DJI IV from liquid ETF options; index spot kept for display
        option_map = {
            "SPX": "US..SPX",
            "IXIC": "US.QQQ",
            "DJI": "US.DIA",
        }
        index_spot_map = {
            "SPX": "^SPX",
            "IXIC": "^NDX",
            "DJI": "^DJI",
        }
        hmm_map = {
            "SPX": "SPX",
            "IXIC": "NSDQ",
            "DJI": "DJI"
        }

        indices_data = {}
        for short_name, option_ticker in option_map.items():
            try:
                # 1. Volatility Surfaces
                cfg = VolSurfaceConfig(underlying=option_ticker, max_dte=60, lookback_days=15)
                study = VolSurfaceStudy(cfg)
                # Load real data directly from disk; demo synthetic generations are strictly disabled
                study.load_history(use_demo_if_empty=False)
                
                dates = sorted(study.surfaces.keys())
                today_d = dates[-1]
                df_today = study.surfaces[today_d]
                option_spot = float(df_today["spot"].iloc[0])
                try:
                    spot, _ = fetch_spot_yfinance(index_spot_map[short_name])
                except Exception:
                    spot = option_spot

                # Interpolate 3D Moneyness grids
                g_dte, g_ks, iv_g = build_iv_grid(df_today, max_dte=cfg.max_dte)
                lv = dupire_local_vol(option_spot, g_dte, g_ks, iv_g, r=cfg.risk_free_rate)

                x_ks = [float(val) for val in g_ks[0, :]]
                y_dte = [float(val) for val in g_dte[:, 0]]
                iv_data = [[float(v) for v in row] for row in iv_g]
                lv_data = [[float(v) for v in row] for row in lv]

                # Calculate smooth IV (SVI-style quadratic regression)
                sub_today = df_today[np.isfinite(df_today["iv"]) & np.isfinite(df_today["dte"])].copy()
                if "ks_ratio" not in sub_today.columns:
                    sub_today["ks_ratio"] = sub_today["option_strike_price"] / sub_today["spot"]
                grouped = sub_today.groupby(["dte", "ks_ratio"])["iv"].mean().reset_index()

                iv_g_smooth = np.zeros_like(iv_g)
                for i, dte in enumerate(y_dte):
                    slice_df = grouped[np.abs(grouped["dte"] - dte) <= 4]
                    if len(slice_df) >= 3:
                        x_pts = np.log(slice_df["ks_ratio"].to_numpy())
                        y_pts = slice_df["iv"].to_numpy()
                        coeffs = np.polyfit(x_pts, y_pts, 2)
                        # Keep quadratic term positive for smile shape sanity
                        if coeffs[0] < 0:
                            coeffs = np.polyfit(x_pts, y_pts, 1)
                            coeffs = np.array([0.0] + list(coeffs))
                        for j, ks in enumerate(x_ks):
                            val = np.polyval(coeffs, np.log(ks))
                            iv_g_smooth[i, j] = np.clip(val, 5.0, 150.0)
                    else:
                        iv_g_smooth[i, :] = iv_g[i, :]

                sv_data = [[float(v) for v in row] for row in iv_g_smooth]

                # Run PCA Decomposition
                result = study.analyze()
                today_f = result["today"]
                pca = SurfaceDeltaPCA(SurfacePCAConfig(n_components=4, baseline_window=21))
                pca_result = pca.fit(surfaces=study.surfaces)
                evr = pca_result["explained_variance_ratio"]
                hist_scores = pca_result["score_history"]
                today_scores = hist_scores[-1].tolist()
                sentiment = pca.sentiment_from_scores(hist_scores[-1], hist_scores)

                vix = float(result.get("vix_context", {}).get("vix", 15.0))
                aiv = float(today_f.get("atm_iv_30d", np.nan))
                psk = float(today_f.get("skew_25d", np.nan))
                tsl = float(today_f.get("term_slope", np.nan))
                bfly = float(today_f.get("butterfly_25d", np.nan))

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

                # Compute consolidated Compass Metric score (-100 to 100)
                ps_score = np.clip(-psk * 4.0, -35, 35)
                ts_score = np.clip(tsl * 5.0, -25, 25)
                vix_score = np.clip((20.0 - vix) * 2.0, -20, 20)
                vrp_score = np.clip((6.0 - vrp_val) * 2.5, -15, 15)
                anchor_score = np.clip(-skew_proxy_chg * 5.0, -10, 10)
                sentiment_score = float(np.clip(ps_score + ts_score + vix_score + vrp_score + anchor_score, -100, 100))

                # 2. HMM Volatility Regime 
                hmm_ticker_name = hmm_map[short_name]
                hmm_model = HMMVolatilityRegime.from_market(hmm_ticker_name, signal_mode=True)
                hmm_model.download_data()
                hmm_model.fit_once()
                hmm_sig = hmm_model.today_signal

                hmm_regime_str = "unknown"
                hmm_prob_today = 50.0
                hmm_prob_tmr = 50.0
                hmm_signal_bool = False
                move_rows = []
                hmm_history_dates = []
                hmm_history_prices = []
                hmm_history_rv = []
                hmm_history_iv = []
                hmm_history_regimes = []

                if hmm_sig is not None:
                    hmm_regime_str = str(hmm_sig.get("hmm", "unknown"))
                    hmm_prob_today = round(float(hmm_sig.get("prob_low_vol", 0.5)) * 100, 1)
                    hmm_prob_tmr = round(float(hmm_sig.get("prob_low_vol_tmr", 0.5)) * 100, 1)
                    hmm_signal_bool = bool(hmm_sig.get("trade_signal", False))

                    # Extract Move Summary Boundaries Table for constructing Iron Condors
                    move_table = hmm_model.format_movement_table()
                    for idx in move_table.index:
                        move_rows.append({
                            "horizon": str(idx),
                            "implied": str(move_table.loc[idx, "Implied"]),
                            "historical": str(move_table.loc[idx, "Historical"]),
                            "spot_implied": str(move_table.loc[idx, "Spot ± implied"])
                        })

                    # Extract HMM history (last 44 business days)
                    feats = hmm_model.features
                    regs = hmm_model.regimes
                    if feats is not None and regs is not None:
                        aligned = feats.join(regs, how="inner").dropna(subset=["SPX", "RV_22", "VIX", "hmm"])
                        history_df = aligned.tail(44)
                        
                        # Fetch Open, High, Low and Volume information
                        ohlc_df = hmm_model._download_ohlc(
                            hmm_model.underly_ticker,
                            start=history_df.index[0],
                            end=history_df.index[-1]
                        ).reindex(history_df.index)
                        
                        hmm_history_dates = [d.strftime("%Y-%m-%d") for d in history_df.index]
                        hmm_history_prices = [float(v) for v in history_df["SPX"]]
                        hmm_history_opens = [float(v) for v in ohlc_df["Open"].fillna(history_df["SPX"])]
                        hmm_history_highs = [float(v) for v in ohlc_df["High"].fillna(history_df["SPX"])]
                        hmm_history_lows = [float(v) for v in ohlc_df["Low"].fillna(history_df["SPX"])]
                        hmm_history_volumes = [float(v) for v in ohlc_df.get("Volume", pd.Series(0, index=ohlc_df.index)).fillna(0)]
                        
                        hmm_history_rv = [float(v) for v in history_df["RV_22"]]
                        hmm_history_iv = [float(v) for v in history_df["VIX"]]
                        hmm_history_regimes = [str(v) for v in history_df["hmm"]]

                indices_data[short_name] = {
                    "exists": True,
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
                    "joint_signal": sentiment["joint_flags"][0].upper().replace("_", " ") if sentiment["joint_flags"] else "NEUTRAL",
                    "score": round(sentiment_score, 1),
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
                    "hmm_regimes": hmm_history_regimes
                }
            except Exception as e:
                print(f"Error processing index {short_name}: {e}")
                traceback.print_exc()
                indices_data[short_name] = {
                    "exists": False,
                    "error": str(e)
                }

        return indices_data


engine = QuantEngine()

# ----------------- Dashboard HTML / JS -----------------
HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Index Quant Signal Hub</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0e1118;color:#d0d4dc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;height:100vh;display:flex;flex-direction:column}
.header{padding:14px 24px;border-bottom:1px solid #1f2330;display:flex;align-items:center;background:#131722;gap:12px;z-index:10}
.header h1{font-size:18px;font-weight:600;color:#e8eaed;letter-spacing:.3px}
.header-tabs{display:flex;background:#1a1f2c;border-radius:6px;padding:2px;margin-left:24px}
.tab-btn{background:transparent;border:none;color:#8f96a3;padding:6px 16px;font-size:12px;font-weight:600;border-radius:4px;cursor:pointer;transition:all .15s}
.tab-btn.active{background:#2a6cff;color:#fff}
.badge{background:#1c2030;border-radius:6px;padding:4px 12px;font-size:13px;color:#9aa0a8;font-weight:500}
.main-layout{display:flex;flex:1;overflow:hidden;padding:16px;gap:16px;height:calc(100vh - 54px)}
.panel{background:#131722;border-radius:8px;border:1px solid #1f2330;overflow:hidden;display:flex;flex-direction:column}
.ptitle{font-size:11px;font-weight:600;padding:10px 16px;border-bottom:1px solid #1f2330;color:#8f96a3;text-transform:uppercase;letter-spacing:.8px}
.surface-panel{flex:1.4;display:flex;flex-direction:column;min-height: 400px;}
.left-col{flex:1.2;display:flex;flex-direction:column;gap:16px;height:100%;overflow-y:auto;padding-right:4px;}
.price-regime-panel{flex:1.0;display:flex;flex-direction:column;min-height: 380px;}
.sidebar{flex:0.8;display:flex;flex-direction:column;gap:16px;overflow-y:auto;padding-right:4px}
.controls{display:flex;gap:8px;padding:8px 16px;border-bottom:1px solid #1f2330;align-items:center}
.controls label{font-size:11px;color:#8f96a3;font-weight:500}
.rg{display:flex;gap:2px;background:#1a1f2c;border-radius:6px;padding:2px}
.rb{padding:4px 12px;border-radius:4px;font-size:11px;cursor:pointer;border:none;background:transparent;color:#8f96a3;font-weight:600;transition:all .15s}
.rb.active{background:#2a6cff;color:#fff}
.dashboard-grid{display:grid;grid-template-columns:1fr 1.1fr;gap:12px;padding:16px;flex:1}
.metric-card{background:#181d2e;border-radius:6px;border:1px solid #232a3f;padding:10px;display:flex;flex-direction:column;justify-content:center}
.metric-card .label{font-size:9px;color:#8f96a3;text-transform:uppercase;font-weight:600;margin-bottom:3px}
.metric-card .val{font-size:16px;font-weight:700;color:#ffffff}
.gauge-wrap{position:relative;text-align:center;padding:8px;flex:1;display:flex;align-items:center;justify-content:center;flex-direction:column}
.gauge-svg{width:100%;max-width:240px;height:120px}
.gv{font-size:22px;font-weight:700;margin-top:2px}
.gl{font-size:12px;font-weight:600}
.hmm-indicator{border-radius:6px;padding:10px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;transition:all .3s}
.hmm-indicator.active{background:rgba(0,204,102,0.1);border:1.5px solid #00cc66}
.hmm-indicator.inactive{background:rgba(231,76,60,0.1);border:1.5px solid #e74c3c}
.summary-content{padding:12px 16px;font-size:12px;line-height:1.5;overflow-y:auto;flex:1}
.summary-content li{margin-bottom:6px;color:#b0b6c2;list-style:none;display:flex;align-items:flex-start}
.summary-content li::before{content:'•';color:#2a6cff;font-weight:bold;font-size:14px;margin-right:8px;line-height:1}

/* Styled Table for Iron Condor and moves */
.condor-table{width:100%;border-collapse:collapse;font-size:11.5px;text-align:left;color:#b0b6c2}
.condor-table th{background:#1a1f2c;color:#8f96a3;font-weight:600;padding:8px 12px;border-bottom:1.5px solid #1f2330;text-transform:uppercase;font-size:10px;letter-spacing:.5px}
.condor-table td{padding:8px 12px;border-bottom:1px solid #1c2030;font-family:monospace}
.condor-table tr:hover{background:rgba(42,108,255,0.05)}
</style>
</head>
<body>
<div class="header">
  <h1>Index Quant Signal Hub</h1>
  <div class="header-tabs">
    <button class="tab-btn active" onclick="switchIndex('SPX')">SPX (S&P 500)</button>
    <button class="tab-btn" onclick="switchIndex('IXIC')">IXIC (NASDAQ)</button>
    <button class="tab-btn" onclick="switchIndex('DJI')">DJI (Dow Jones)</button>
  </div>
  <span style="flex:1"></span>
  <span id="dateBadge" class="badge">--</span>
  <span id="spotBadge" class="badge">--</span>
</div>

<div class="main-layout">
  <div class="left-col">
    <div class="panel surface-panel">
      <div class="ptitle" id="surfaceTitle">Vol Surface - SPX</div>
      <div class="controls">
        <label>Mode:</label>
        <div class="rg">
          <button class="rb active" id="btnRawIV" onclick="setMode('iv')">Raw IV</button>
          <button class="rb" id="btnSmoothIV" onclick="setMode('sv')">Smooth IV</button>
          <button class="rb" id="btnLV" onclick="setMode('lv')">Local Vol</button>
        </div>
      </div>
      <div id="surfaceContainer" style="flex:1"></div>
    </div>

    <!-- Horizontal Sentiment Thermometer Bar placed directly under the Surface -->
    <div class="panel" style="flex: 0 0 auto; padding: 12px 16px;">
      <div class="ptitle" style="padding: 0 0 10px 0; border-bottom: none;">Quant Sentiment Compass (Horizontal Thermometer)</div>
      <div id="horizontalCompassContainer" style="position: relative; min-height: 52px; margin-top: 4px;"></div>
    </div>
    
    <div class="panel price-regime-panel">
      <div class="ptitle" id="regimeTitle">Price & Volatility Regime Analysis - SPX</div>
      <div id="regimeContainer" style="flex:1; min-height: 250px;"></div>
    </div>
  </div>

  <div class="sidebar">
    <div class="panel" style="flex-shrink: 0;">
      <div class="ptitle">Quant Regime & Sentiment Compass</div>
      <div class="dashboard-grid" style="grid-template-columns: 1fr; gap: 10px; padding: 16px;">
        <div style="display:flex; flex-direction:column; gap:10px">
          <div class="hmm-indicator" id="hmmModeCard">
            <div style="font-size:9px;text-transform:uppercase;font-weight:600;opacity:0.8">HMM Trading Signal</div>
            <div style="font-size:16px;font-weight:700;margin:2px 0" id="hmmSignalVal">--</div>
            <div style="font-size:9px;opacity:0.9" id="hmmProbVal">--</div>
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
    </div>

    <!-- Short Iron Condor Selector Panel -->
    <div class="panel" style="flex: 1; min-height: 200px;">
      <div class="ptitle">1-Sigma Vol Move Targets (Short Iron Condor Reference)</div>
      <div style="overflow-x: auto; flex: 1;">
        <table class="condor-table">
          <thead>
            <tr>
              <th>Horizon</th>
              <th>Implied Move (IV)</th>
              <th>Historical Move</th>
              <th>Spot Range (Implied ±1σ)</th>
            </tr>
          </thead>
          <tbody id="condorTableBody">
            <tr><td colspan="4" style="text-align:center;color:#8f96a3">No data loaded</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Vol Shading Analysis RV vs IV (Third subplot, positioned below structure metrics / move targets) -->
    <div class="panel price-regime-panel" style="flex: 1; min-height: 200px;">
      <div class="ptitle">Volatility Historical Shading (RV vs IV)</div>
      <div id="volHeatmapContainer" style="flex: 1; min-height: 200px;"></div>
    </div>

    <div class="panel" style="flex: 1; min-height: 200px;">
      <div class="ptitle">Quantitative Structure Metrics</div>
      <div class="summary-content" id="bulletsBox"></div>
    </div>
  </div>
</div>

<script>
let currentIdx = 'SPX';
let currentMode = 'iv';
let dbData = {};

async function fetchServerData() {
  try {
    const res = await fetch('/api/signals');
    dbData = await res.json();
    renderAll();
  } catch (e) {
    console.error("Error drawing Quant Hub layouts:", e);
  }
}

function renderAll() {
  const data = dbData[currentIdx];
  if (!data || !data.exists) {
    document.getElementById('surfaceContainer').innerHTML = `<div style="padding:40px;text-align:center;color:#e74c3c;font-weight:600">Quant pipeline generation failed. Double check caches or fetch live.</div>`;
    return;
  }
  
  // Date and Spot indicators
  document.getElementById('dateBadge').textContent = 'Date: ' + data.date;
  document.getElementById('spotBadge').textContent = currentIdx + ': ' + data.spot.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
  document.getElementById('surfaceTitle').textContent = "Vol Surface - " + currentIdx;

  // 1. 3D Surface
  let z;
  let title;
  let colorscale;
  if (currentMode === 'iv') {
    z = data.surface_z;
    title = 'Raw Implied Vol (%)';
    colorscale = 'Viridis';
  } else if (currentMode === 'sv') {
    z = data.surface_sv || data.surface_z;
    title = 'Smooth Implied Vol (%)';
    colorscale = 'Cividis';
  } else {
    z = data.surface_w;
    title = 'Local Vol (%)';
    colorscale = 'Magma';
  }
  const rng = currentMode === 'lv' ? [0, Math.max(...z.flat())] : undefined;
  
  const plotlyData = [{
    type: 'surface',
    x: data.surface_x,
    y: data.surface_y,
    z: z,
    colorscale: colorscale,
    hovertemplate: 'K/S (Moneyness): %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>',
    colorbar: {title: title, titleside: 'right', x: 0.88, len: 0.7},
    contours: {z: {show: true, usecolormap: true, highlightcolor: 'lime', project: {z: true}}},
    cmin: rng && rng[0], cmax: rng && rng[1],
  }];

  const layout = {
    margin: {l:0, r:0, t:0, b:0},
    paper_bgcolor: '#131722',
    plot_bgcolor: '#131722',
    scene: {
      xaxis: {title:'Moneyness (K/S)', gridcolor:'#222a3d', zerolinecolor:'#222a3d', tickfont:{color:'#8f96a3'}},
      yaxis: {title:'DTE', gridcolor:'#222a3d', zerolinecolor:'#222a3d', tickfont:{color:'#8f96a3'}},
      zaxis: {title: title, gridcolor:'#222a3d', zerolinecolor:'#222a3d', tickfont:{color:'#8f96a3'}},
      camera: {eye: {x:-1.5, y:-1.5, z:0.8}},
      aspectmode: 'manual',
      aspectratio: {x:1.0, y:1.2, z:0.6}
    },
    hoverlabel: {bgcolor:'#1c2030', font:{size:12}},
    uirevision: 'surface',
  };
  Plotly.react('surfaceContainer', plotlyData, layout, {displayModeBar: false, responsive: true});

  // 2. Speedometer Gauge (Disabled & replaced by horizontal thermometer under Surface)
  const score = data.score;

  // 7. Render Horizontal Sentiment Thermometer Bar under the 3D Surface
  const compassPct = ((score + 100) / 200) * 100;
  let scoreLabelColor = "#f1c40f";
  if (score > 50) scoreLabelColor = "#27ae60";
  else if (score > 15) scoreLabelColor = "#2ecc71";
  else if (score < -50) scoreLabelColor = "#e74c3c";
  else if (score < -15) scoreLabelColor = "#e67e22";

  document.getElementById('horizontalCompassContainer').innerHTML = `
    <div style="position:relative; width:100%; padding-top:20px;">
      <div class="sentiment-thermometer-label-container" style="display: flex; justify-content: space-between; font-size: 10px; color: #8f96a3; margin-bottom: 6px; font-weight: 500;">
        <span>Extremely Bearish (-100)</span>
        <span>Neutral (0)</span>
        <span>Extremely Bullish (100)</span>
      </div>
      <div class="sentiment-bar-track" style="position: relative; height: 16px; background: linear-gradient(90deg, #e74c3c 0%, #e67e22 25%, #f1c40f 50%, #2ecc71 75%, #27ae60 100%); border-radius: 8px; overflow: visible;">
        <!-- Pointer marker -->
        <div id="sentimentPointer" style="position: absolute; top: -3px; left: ${compassPct}%; width: 6px; height: 22px; background: #ffffff; border-radius: 3px; box-shadow: 0 0 10px rgba(255,255,255,0.8), 0 0 4px rgba(0,0,0,0.5); transform: translateX(-50%); transition: left 0.3s ease;"></div>
        <!-- Tooltip box sliding with pointer -->
        <div id="sentimentValueMarker" style="position: absolute; top: -34px; left: ${compassPct}%; background: #2a6cff; color: #ffffff; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; transform: translateX(-50%); transition: left 0.3s ease; white-space: nowrap; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
          Score: <span id="sentimentValueSpan">${score >= 0 ? '+' : ''}${score}</span>/100 (<span style="color: ${scoreLabelColor}">${getScoreLabel(score)}</span>)
        </div>
      </div>
    </div>
  `;

  // 3. Side Quant Cards
  // VRP
  document.getElementById('vraVal').textContent = data.vrp.toFixed(1) + ' pts';
  // Spread
  document.getElementById('tslVal').textContent = data.tsl.toFixed(1) + ' vol pts';

  // 4. HMM Mode Alert
  const hmmCard = document.getElementById('hmmModeCard');
  const signalVal = document.getElementById('hmmSignalVal');
  const probVal = document.getElementById('hmmProbVal');
  if (data.hmm_signal) {
    hmmCard.className = "hmm-indicator active";
    signalVal.textContent = "BUY / LONG";
    signalVal.style.color = "#00cc66";
  } else {
    hmmCard.className = "hmm-indicator inactive";
    signalVal.textContent = "NEUTRAL / CASH";
    signalVal.style.color = "#e74c3c";
  }
  probVal.textContent = 'P(Calm today): ' + data.hmm_prob_today + '% | P(Calm tmr): ' + data.hmm_prob_tmr + '%';

  // 5. Render Iron Condor Moves Reference Table 
  const tableBody = document.getElementById('condorTableBody');
  tableBody.innerHTML = '';
  if (data.hmm_move_table && data.hmm_move_table.length > 0) {
    data.hmm_move_table.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:bold;color:#fff">` + row.horizon + `</td>
        <td style="color:#00cc66">` + row.implied + `</td>
        <td style="color:#8f96a3">` + row.historical + `</td>
        <td style="color:#2a6cff;font-weight:bold">` + row.spot_implied + `</td>
      `;
      tableBody.appendChild(tr);
    });
  } else {
    tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#8f96a3">No move matrix compiled</td></tr>`;
  }

  // 6. Structure Bullets
  const bulletsBox = document.getElementById('bulletsBox');
  bulletsBox.innerHTML = '';
  const ul = document.createElement('ul');
  
  const bullets = [
    'Implied Vol (ATM 30d): ' + data.aiv.toFixed(1) + '% | VIX Base level: ' + data.vix.toFixed(1) + '%',
    'Skew Steepness Premium: 25d option slope is ' + data.psk.toFixed(1) + ' vol pts (5d Change: ' + data.d_skew.toFixed(1) + ' pts)',
    'Wings Fly Curvature: 25d butterfly represents ' + data.bfly.toFixed(1) + ' vol pts (5d Change: ' + data.d_bfly.toFixed(1) + ' pts)',
    'PCA Delta Systemic Shocks: PC1 (Shift) = ' + data.pc1.toFixed(2) + ' | PC2 (Skew Tilt) = ' + data.pc2.toFixed(2) + '',
    'Structural Joint Flag: PCA classified as [' + data.joint_signal + ']',
    'Market Volatility Regime: HMM classified present status as [' + data.hmm_regime.toUpperCase() + ']'
  ];

  bullets.forEach(b => {
    const li = document.createElement('li');
    li.textContent = b;
    ul.appendChild(li);
  });
  bulletsBox.appendChild(ul);

  // 7. Render Price & Volatility Regime Analysis Plot (Plotly Multi-axis with Shading)
  if (data.hmm_dates && data.hmm_dates.length > 0) {
    document.getElementById('regimeTitle').textContent = "Price & Volatility Regime Analysis - " + currentIdx;
    
    // Draw HMM Vol Shading blocks as layout shapes
    let shapes = [];
    let stateStart = null;
    for (let i = 0; i < data.hmm_regimes.length; i++) {
      if (data.hmm_regimes[i] === 'high_vol') {
        if (stateStart === null) {
          stateStart = data.hmm_dates[i];
        }
      } else {
        if (stateStart !== null) {
          shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: stateStart,
            x1: data.hmm_dates[i - 1],
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(231,76,60,0.12)',
            line: { width: 0 }
          });
          stateStart = null;
        }
      }
    }
    if (stateStart !== null) {
      shapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: stateStart,
        x1: data.hmm_dates[data.hmm_dates.length - 1],
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(231,76,60,0.12)',
        line: { width: 0 }
      });
    }

    const regimePlotData = [
      {
        x: data.hmm_dates,
        open: data.hmm_opens,
        high: data.hmm_highs,
        low: data.hmm_lows,
        close: data.hmm_prices,
        name: currentIdx + ' Price',
        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y1',
        increasing: { line: { color: '#2ecc71' } },
        decreasing: { line: { color: '#e74c3c' } }
      },
      {
        x: data.hmm_dates,
        y: data.hmm_volumes,
        name: 'Volume',
        type: 'bar',
        xaxis: 'x',
        yaxis: 'y2',
        marker: { color: 'rgba(42, 108, 255, 0.45)' }
      }
    ];

    const regimeLayout = {
      margin: { l: 50, r: 50, t: 25, b: 40 },
      paper_bgcolor: '#131722',
      plot_bgcolor: '#131722',
      showlegend: false,
      xaxis: {
        gridcolor: '#1f2330',
        tickfont: { color: '#8f96a3', size: 10 },
        type: 'date',
        rangeslider: { visible: false }
      },
      yaxis: {
        title: 'Index price',
        titlefont: { color: '#2a6cff', size: 11 },
        tickfont: { color: '#8f96a3', size: 10 },
        gridcolor: '#1f2330',
        domain: [0.3, 1]
      },
      yaxis2: {
        title: 'Volume',
        titlefont: { color: '#8f96a3', size: 11 },
        tickfont: { color: '#8f96a3', size: 9 },
        gridcolor: '#1f2330',
        domain: [0, 0.25],
        anchor: 'x'
      },
      shapes: shapes,
      hovermode: 'x'
    };

    Plotly.react('regimeContainer', regimePlotData, regimeLayout, { displayModeBar: false, responsive: true });
  }

  // 8. Render Volatility Shading Plot (RV vs IV)
  if (data.hmm_dates && data.hmm_dates.length > 0) {
    const volPlotData = [
      {
        x: data.hmm_dates,
        y: data.hmm_rv,
        name: '22d Realized Vol %',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f1c40f', width: 1.5 }
      },
      {
        x: data.hmm_dates,
        y: data.hmm_iv,
        name: 'ATM Implied Vol %',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ff7f0e', width: 1.5, dash: 'dot' }
      }
    ];

    const volLayout = {
      margin: { l: 40, r: 20, t: 25, b: 40 },
      paper_bgcolor: '#131722',
      plot_bgcolor: '#131722',
      showlegend: true,
      legend: {
        orientation: 'h',
        x: 0,
        y: 1.15,
        font: { color: '#8f96a3', size: 10 }
      },
      xaxis: {
        gridcolor: '#1f2330',
        tickfont: { color: '#8f96a3', size: 10 },
        type: 'date'
      },
      yaxis: {
        title: 'Annualized Vol %',
        titlefont: { color: '#8f96a3', size: 11 },
        tickfont: { color: '#8f96a3', size: 10 },
        gridcolor: '#1f2330'
      },
      hovermode: 'x'
    };

    Plotly.react('volHeatmapContainer', volPlotData, volLayout, { displayModeBar: false, responsive: true });
  }
}

// Helpers
function getScoreLabel(score) {
  if (score > 50) return "Extremely Bullish";
  if (score > 15) return "Slightly Bullish";
  if (score > -15) return "Neutral";
  if (score > -50) return "Slightly Bearish";
  return "Extremely Bearish";
}

function switchIndex(idx) {
  currentIdx = idx;
  const btns = document.querySelectorAll('.tab-btn');
  btns.forEach(b => {
    if (b.textContent.includes(idx)) b.classList.add('active');
    else b.classList.remove('active');
  });
  renderAll();
}

function setMode(mode) {
  currentMode = mode;
  document.getElementById('btnRawIV').className = 'rb' + (mode === 'iv' ? ' active' : '');
  document.getElementById('btnSmoothIV').className = 'rb' + (mode === 'sv' ? ' active' : '');
  document.getElementById('btnLV').className = 'rb' + (mode === 'lv' ? ' active' : '');
  renderAll();
}

// Fetch in loop
fetchServerData();
setInterval(fetchServerData, 300000); // 5 min interval update
</script>
</body>
</html>
"""

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif parsed.path == "/api/signals":
            try:
                data_dict = engine.run_pipeline()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data_dict).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Silence annoying routing logs
        pass

if __name__ == "__main__":
    print(f"\n  >>> Index Quant Hub Running at http://127.0.0.1:{PORT}  <<<\n")
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
