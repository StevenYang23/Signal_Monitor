import os, sys, glob, warnings, json, http.server, urllib.parse, traceback
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__) or ".", ".matplotlib_temp")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__) or ".")
from vol_surface import build_iv_grid_delta, dupire_local_vol_delta

DATA_DIR = Path(__file__).parent.resolve() / "research" / "data" / "vol_surface" / "US__SPX"
files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
ALL_SURFACES = {}
for f in files:
    ALL_SURFACES[pd.Timestamp(Path(f).stem).date()] = pd.read_parquet(f)
dates = sorted(ALL_SURFACES.keys())
TODAY, DF_TODAY = dates[-1], ALL_SURFACES[dates[-1]]
SPOT = float(DF_TODAY["spot"].iloc[0])
dg = np.linspace(-0.5, 0.5, 31)
eg = np.array([7, 10, 14, 21, 30, 45, 60])
GD, GDL, IV_GRID = build_iv_grid_delta(DF_TODAY, delta_grid=dg, dte_grid=eg, max_dte=60)
LV_GRID = dupire_local_vol_delta(SPOT, GD, GDL, IV_GRID, r=0.045)

# compact 1d axes
dte_1d = [float(x) for x in GD[:,0]]
del_1d = [float(x) for x in GDL[0,:]]
iv_data = [[float(v) for v in row] for row in IV_GRID]
lv_data = [[float(v) for v in row] for row in LV_GRID]

# sentiment
df30 = DF_TODAY[DF_TODAY["dte"].between(28, 32)].copy()
if len(df30) == 0: df30 = DF_TODAY.nsmallest(50, "dte")
df30["ad"] = df30["delta"].abs(); atm = df30.loc[df30["ad"].idxmin()]; aiv = float(atm["iv"])
pw = df30[df30["delta"].between(-0.35, -0.20)]
psk_v = float(pw["iv"].mean()) if len(pw) > 0 else aiv*1.1; psk = (psk_v/aiv-1)*100
cw = df30[df30["delta"].between(0.20, 0.35)]
csk_v = float(cw["iv"].mean()) if len(cw) > 0 else aiv*1.05; csk = (csk_v/aiv-1)*100
s7 = DF_TODAY[DF_TODAY["dte"].between(4,10)].copy(); s7["ad"]=s7["delta"].abs()
r7 = s7.loc[s7["ad"].idxmin()] if len(s7)>0 else atm
s60 = DF_TODAY[DF_TODAY["dte"].between(50,61)].copy(); s60["ad"]=s60["delta"].abs()
r60 = s60.loc[s60["ad"].idxmin()] if len(s60)>0 else atm; tsl = float(r60["iv"])-float(r7["iv"])
am = [s[np.isfinite(s["iv"])]["iv"].median() for _,s in ALL_SURFACES.items() if len(s[np.isfinite(s["iv"])])>0]
ivs = pd.Series(am, index=sorted(ALL_SURFACES.keys()))
rnk = (ivs<aiv).sum()/max(len(ivs)-1,1) if len(ivs)>1 else 0.5; ivc = (1-rnk)*100-50
ps = np.clip(-psk*25/max(abs(psk)+0.5,5), -25, 25)
cs = np.clip(csk*25/max(abs(csk)+0.5,3), -15, 25)
ts = np.clip(tsl*25/max(abs(tsl)+0.5,3), -25, 25)
cmp = float(np.clip(ps+cs+ivc+ts, -100, 100))
if cmp > 50: lbl, col = "Bullish", "#00cc66"
elif cmp > 15: lbl, col = "Slightly Bullish", "#88cc44"
elif cmp > -15: lbl, col = "Neutral", "#cccc44"
elif cmp > -50: lbl, col = "Slightly Bearish", "#cc8844"
else: lbl, col = "Bearish", "#cc4444"

# summary
ivf = IV_GRID.flatten(); hi, lo = float(np.nanmax(ivf)), float(np.nanmin(ivf))
hidx = int(np.nanargmax(ivf)); hd, hdel = int(GD.flatten()[hidx]), float(GDL.flatten()[hidx])
lvf = LV_GRID.flatten(); lvm = float(np.nanmean(lvf)) if np.isfinite(lvf).any() else 0; lmin = float(np.nanmin(lvf))
psk_r, tsl_r, aiv_r = round(psk,1), round(tsl,1), round(aiv,1)
psk_str = f"Put/call skew balanced ({psk_r:+.1f}%) — no extreme directional bias" if -1<=psk_r<=3 else (f"Put skew elevated ({psk_r:+.1f}%) — tail hedging demand elevated" if psk_r>3 else f"Call skew dominant ({psk_r:+.1f}%) — put premium subdued")
tsl_str = f"Term structure flat ({tsl_r:+.1f}pt) — no clear tenor dislocation" if -2<=tsl_r<=2 else (f"Term structure inverted ({tsl_r:+.1f}pt) — near-term stress" if tsl_r<-2 else f"Term structure steep ({tsl_r:+.1f}pt) — uncertainty in longer tenor")
slist = [f"SPX @ {SPOT:,.0f} | ATM IV: {aiv_r}%", psk_str, tsl_str, f"Surface sentiment: {lbl} ({round(cmp)}/100)", f"IV range: {lo:.1f}–{hi:.1f}% (peak @ DTE={hd}, delta={hdel:+.2f})", f"Local vol avg: {lvm:.1f}% (min: {lmin:.1f}%)", f"Data as of {TODAY}"]

data = json.dumps(dict(t=str(TODAY), sp=SPOT, aiv=aiv_r, cmp=round(cmp), l=lbl, c=col, psk=psk_r, csk=round(csk,1), tsl=tsl_r, x=dte_1d, y=del_1d, z=iv_data, w=lv_data, hi=round(hi,1), lo=round(lo,1), ph=hd, pd=round(hdel,2), lm=round(lvm,1), ln=round(lmin,1), s=slist), separators=(",", ":"))

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Signal Monitor</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#d0d4dc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden}
.header{padding:18px 24px;border-bottom:1px solid #2a2d35;display:flex;align-items:center;gap:12px}
.header h1{font-size:20px;font-weight:600;color:#e8eaed;letter-spacing:.3px}
.badge{background:#1e2028;border-radius:6px;padding:4px 12px;font-size:13px;color:#9aa0a8}
.main{display:flex;gap:16px;padding:16px;height:calc(100vh - 62px)}
.panel{background:#161820;border-radius:10px;border:1px solid #252830;overflow:hidden;display:flex;flex-direction:column}
.ptitle{font-size:13px;font-weight:600;padding:12px 16px;border-bottom:1px solid #252830;color:#b0b4bc;text-transform:uppercase;letter-spacing:.5px}
.surface-panel{flex:1}
.sidebar{width:360px;display:flex;flex-direction:column;gap:16px}
.controls{display:flex;gap:8px;padding:8px 16px;border-bottom:1px solid #252830;align-items:center}
.controls label{font-size:12px;color:#9aa0a8}
.rg{display:flex;gap:2px;background:#1a1d26;border-radius:6px;padding:2px}
.rb{padding:5px 14px;border-radius:5px;font-size:12px;cursor:pointer;border:none;background:transparent;color:#9aa0a8;transition:all .15s;font-family:inherit}
.rb.active{background:#2a6cff;color:#fff}
.rb:hover:not(.active){color:#d0d4dc}
.gauge-wrap{position:relative;text-align:center;padding:4px 8px}
.gauge-svg{width:100%;max-height:160px}
.gl{text-align:center;font-size:13px;font-weight:500;padding:2px 0 4px}
.gv{font-size:26px;font-weight:700;color:#e8eaed;margin-top:-2px}
.summary-content{padding:12px 16px;font-size:13px;line-height:1.7;overflow-y:auto}
.summary-content li{margin-bottom:6px;color:#b8bcc4;list-style:none}
.summary-content li::before{content:'\u2022';color:#5a6eff;font-weight:bold;margin-right:8px}
</style></head><body>
<div class="header"><h1>Surface Monitor</h1><span class="badge" id="dateBadge"></span><span style="flex:1"></span><span id="spotBadge" class="badge"></span></div>
<div class="main">
<div class="panel surface-panel"><div class="ptitle">Vol Surface</div>
<div class="controls"><label>Mode:</label>
<div class="rg"><button class="rb active" id="btnIV" onclick="setMode('iv')">Implied Vol</button><button class="rb" id="btnLV" onclick="setMode('lv')">Local Vol</button></div>
</div><div id="surfaceContainer" style="flex:1"></div></div>
<div class="sidebar">
<div class="panel" style="min-height:220px"><div class="ptitle">Market Sentiment</div><div id="gaugeContainer" class="gauge-wrap"></div></div>
<div class="panel" style="flex:1"><div class="ptitle">Summary</div><div id="summaryContainer" class="summary-content"></div></div>
</div></div>
<script>
const D = """ + data + """;
const ivZ = D.z, lvZ = D.w;
let currentMode = 'iv';

function renderSurface(mode) {
  const z = mode === 'iv' ? ivZ : lvZ;
  const title = mode === 'iv' ? 'Implied Vol (%)' : 'Local Vol (%)';
  const rng = mode === 'lv' ? [0, Math.max.apply(null, z.flat())] : undefined;
  const data = [{
    type: 'surface', x: D.x, y: D.y, z: z,
    colorscale: 'RdYlBu_r',
    hovertemplate: 'DTE: %{x}<br>Delta: %{y:.2f}<br>Vol: %{z:.1f}%<extra></extra>',
    colorbar: {title: 'Vol %', titleside: 'right', x: 0.88},
    contours: {z: {show: true, usecolormap: true, highlightcolor: 'lime', project: {z: true}}},
    cmin: rng && rng[0], cmax: rng && rng[1],
  }];
  const layout = {
    margin: {l:0, r:0, t:0, b:0},
    paper_bgcolor: '#161820', plot_bgcolor: '#161820',
    scene: {
      xaxis: {title:'DTE', gridcolor:'#2a2d35', zerolinecolor:'#3a3d45', tickfont:{color:'#9aa0a8'}},
      yaxis: {title:'Delta', gridcolor:'#2a2d35', zerolinecolor:'#3a3d45', tickfont:{color:'#9aa0a8'}},
      zaxis: {title:'Vol (%)', gridcolor:'#2a2d35', zerolinecolor:'#3a3d45', tickfont:{color:'#9aa0a8'}},
      camera: {eye: {x:-1.8, y:-1.5, z:0.7}},
      aspectmode: 'auto',
    },
    hoverlabel: {bgcolor:'#1e2028', font:{size:12}},
    uirevision: 'surface',
  };
  const cfg = {displayModeBar: false, responsive: true, scrollZoom: true};
  Plotly.react('surfaceContainer', data, layout, cfg);
}

function renderGauge() {
  const angle = (D.cmp + 100) / 200 * 180;
  const rad = angle * Math.PI / 180;
  const r = 70, cx = 100, cy = 90;
  const nx = cx + r * Math.sin(rad), ny = cy - r * Math.cos(rad);
  const bearArc = 'M ' + (cx-r) + ',' + cy + ' A ' + r + ',' + r + ' 0 0,0 ' + cx + ',' + (cy-r);
  const bullArc = 'M ' + cx + ',' + (cy-r) + ' A ' + r + ',' + r + ' 0 0,0 ' + (cx+r) + ',' + cy;
  let ticksHtml = '';
  [-100,-50,0,50,100].forEach(function(v) {
    const a = (v + 100) / 200 * 180 * Math.PI / 180;
    const tx = cx + (r + 6) * Math.sin(a), ty = cy - (r + 6) * Math.cos(a);
    ticksHtml += '<span style="position:absolute;font-size:10px;color:#7a7e86;left:' + (tx+85) + 'px;top:' + (ty-2) + 'px;transform:translate(-50%,-50%)">' + v + '</span>';
  });
  document.getElementById('gaugeContainer').innerHTML =
    '<div style="position:relative;text-align:center">' +
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120" class="gauge-svg">' +
    '<path d="' + bearArc + '" stroke="#cc4444" stroke-width="10" fill="none" stroke-linecap="round" opacity="0.6"/>' +
    '<path d="' + bullArc + '" stroke="#00cc66" stroke-width="10" fill="none" stroke-linecap="round" opacity="0.6"/>' +
    '<path d="M ' + (cx-r-2) + ',' + cy + ' A ' + (r+2) + ',' + (r+2) + ' 0 0,0 ' + (cx+r+2) + ',' + cy + '" stroke="#252830" stroke-width="1" fill="none"/>' +
    '<circle cx="' + cx + '" cy="' + cy + '" r="' + (r-8) + '" fill="#11131c" opacity="0.5"/>' +
    '<line x1="' + cx + '" y1="' + cy + '" x2="' + nx + '" y2="' + ny + '" stroke="#e8eaed" stroke-width="2.5" stroke-linecap="round"/>' +
    '<circle cx="' + cx + '" cy="' + cy + '" r="4" fill="#e8eaed"/>' +
    '</svg>' + ticksHtml +
    '<span style="position:absolute;left:18px;bottom:4px;font-size:11px;color:#cc4444;font-weight:500">Bear</span>' +
    '<span style="position:absolute;right:18px;bottom:4px;font-size:11px;color:#00cc66;font-weight:500">Bull</span>' +
    '<div class="gl" style="color:' + D.c + '">' + D.l + '</div>' +
    '<div class="gv">' + (D.cmp >= 0 ? '+' : '') + D.cmp + ' / 100</div>' +
    '</div>';
}

function renderSummary() {
  const ul = document.createElement('ul');
  D.s.forEach(function(b) {
    const li = document.createElement('li');
    li.textContent = b;
    ul.appendChild(li);
  });
  document.getElementById('summaryContainer').appendChild(ul);
}

function setMode(m) {
  currentMode = m;
  document.getElementById('btnIV').className = 'rb' + (m === 'iv' ? ' active' : '');
  document.getElementById('btnLV').className = 'rb' + (m === 'lv' ? ' active' : '');
  renderSurface(m);
}

// init
document.getElementById('dateBadge').textContent = D.t;
document.getElementById('spotBadge').textContent = 'SPX ' + D.sp.toLocaleString();
renderSurface('iv');
renderGauge();
renderSummary();
</script></body></html>"""

PORT = int(os.environ.get("PORT", 8050))
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {args[0]} {args[1]} {args[2]}")

print(f"\n  >>> Dashboard running at http://127.0.0.1:{PORT}  <<<\n")
server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()