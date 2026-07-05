
let currentIdx = 'SPX';
let currentMode = 'iv';
let dbData = {};
let isFetching = false;
let loadTimer = null;
let loadStartedAt = 0;
const ALL_INDICES = ['SPX', 'IXIC', 'DJI'];

function startLoadTimer() {
  loadStartedAt = Date.now();
  const el = document.getElementById('loadElapsed');
  if (loadTimer) clearInterval(loadTimer);
  loadTimer = setInterval(() => {
    if (el) el.textContent = Math.floor((Date.now() - loadStartedAt) / 1000) + 's elapsed';
  }, 1000);
}

function stopLoadTimer() {
  if (loadTimer) { clearInterval(loadTimer); loadTimer = null; }
}

function showLoadingOverlay(idx) {
  const box = document.getElementById('surfaceContainer');
  box.innerHTML = `
    <div class="loading-overlay">
      <div class="title">Loading ${idx}…</div>
      <div class="hint">Live Futu fetch in progress for all indices.<br>Please wait — do not refresh the page.</div>
      <div class="elapsed" id="loadElapsed">0s elapsed</div>
    </div>`;
  startLoadTimer();
}

function setRefreshState(loading) {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = loading;
  btn.textContent = loading ? 'Loading…' : 'Refresh All';
}

async function fetchServerData() {
  if (isFetching) return;
  isFetching = true;
  setRefreshState(true);
  dbData = {};
  showLoadingOverlay('SPX (1/3)');
  const loadedBadge = document.getElementById('loadedBadge');
  loadedBadge.style.display = 'inline-block';
  loadedBadge.textContent = 'Fetching live data…';
  let anyOk = false;
  try {
    for (let i = 0; i < ALL_INDICES.length; i++) {
      const idx = ALL_INDICES[i];
      loadedBadge.textContent = `Fetching ${idx} (${i + 1}/${ALL_INDICES.length})…`;
      showLoadingOverlay(`${idx} (${i + 1}/${ALL_INDICES.length})`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 600000);
      const res = await fetch('/api/index/' + idx + '?_=' + Date.now(), { cache: 'no-store', signal: controller.signal });
      clearTimeout(timeoutId);
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        dbData[idx] = { exists: false, error: errBody.error || ('HTTP ' + res.status) };
      } else {
        dbData[idx] = await res.json();
        if (dbData[idx].exists) anyOk = true;
      }
      if (idx === currentIdx || (currentIdx === 'SPX' && dbData.SPX)) {
        try { renderAll(); } catch (renderErr) {
          console.error('renderAll failed:', renderErr);
        }
      }
    }
    if (anyOk) {
      loadedBadge.textContent = 'Loaded: ' + new Date().toLocaleTimeString();
    } else {
      loadedBadge.textContent = 'Load failed';
      throw new Error(dbData[currentIdx]?.error || 'All indices failed');
    }
  } catch (e) {
    console.error("Error drawing Quant Hub layouts:", e);
    stopLoadTimer();
    const msg = e.name === 'AbortError'
      ? 'Request timed out. Check Futu OpenD and try Refresh All.'
      : ('Failed to load: ' + e.message);
    document.getElementById('surfaceContainer').innerHTML =
      `<div class="loading-overlay"><div class="title" style="color:#e74c3c">${msg}</div></div>`;
    loadedBadge.textContent = 'Load failed';
  } finally {
    stopLoadTimer();
    isFetching = false;
    setRefreshState(false);
  }
}

function refreshAll() {
  fetchServerData();
}

function renderAll() {
  stopLoadTimer();
  try {
  const data = dbData[currentIdx];
  if (!data) {
    showLoadingOverlay(currentIdx);
    return;
  }
  if (!data.exists) {
    document.getElementById('surfaceContainer').innerHTML =
      `<div style="padding:40px;text-align:center;color:#e74c3c;font-weight:600">${currentIdx}: ${data.error || 'pipeline failed'}</div>`;
    return;
  }
  
  // Date and Spot indicators
  const warnBadge = document.getElementById('warnBadge');
  if (data.warnings && data.warnings.length > 0) {
    warnBadge.style.display = 'inline-block';
    warnBadge.textContent = '⚠ ' + data.warnings.join(' | ');
    warnBadge.title = data.warnings.join('
');
  } else {
    warnBadge.style.display = 'none';
    warnBadge.title = '';
  }
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
    data.hmm_move_table.slice(0, 4).forEach(row => {
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

  // 6. Structure metrics with insights
  const bulletsBox = document.getElementById('bulletsBox');
  bulletsBox.innerHTML = '';
  const blocks = data.structure_metrics || [];
  if (blocks.length === 0) {
    bulletsBox.innerHTML = '<div style="color:#8f96a3;font-size:12px">No structure metrics available.</div>';
  } else {
    blocks.forEach(block => {
      const div = document.createElement('div');
      div.className = 'metric-block';
      div.innerHTML = `
        <div class="metric-line">${block.metric}</div>
        <div class="insight-line">${block.insight}</div>
      `;
      bulletsBox.appendChild(div);
    });
  }

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
    const iv22Idx = Math.max(0, data.hmm_iv.length - 1 - 22);
    const iv22Ago = data.hmm_iv[iv22Idx];
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
    if (Number.isFinite(iv22Ago)) {
      volPlotData.push({
        x: [data.hmm_dates[0], data.hmm_dates[data.hmm_dates.length - 1]],
        y: [iv22Ago, iv22Ago],
        name: 'IV 22d ago',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#e74c3c', width: 1.5, dash: 'dash' },
        hovertemplate: 'IV 22d ago: %{y:.1f}%<extra></extra>'
      });
    }

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
  } catch (err) {
    console.error('renderAll error:', err);
    document.getElementById('surfaceContainer').innerHTML =
      `<div class="loading-overlay"><div class="title" style="color:#e74c3c">Render error: ${err.message}</div></div>`;
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
  const btns = document.querySelectorAll('.header-tabs .tab-btn');
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

// Fresh fetch on page open; tab switches reuse in-memory dbData until Refresh All
fetchServerData();
