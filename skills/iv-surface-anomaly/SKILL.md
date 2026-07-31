---
name: iv-surface-anomaly
description: >-
  Find anomalous nodes on a single-session SPX IV and Dupire local-vol surface;
  cross-check with VVIX, SKEW, and 3M implied correlation. Use when analyzing
  app_simp dashboard/API anomaly lists or notebook surface grids. Not for
  day-over-day PCA sentiment.
---

# IV / Local-Vol Surface Anomaly Hunt

Find anomalous points on a **single-session** SPX implied-vol and Dupire local-vol surface. Cross-check with auxiliary indices (VVIX, CBOE SKEW, 3M implied correlation). Write a structured anomaly analysis from dashboard or notebook output.

## When to use

Call this skill when the user wants to:
- Spot weird nodes / spikes / holes on today's SPX IV or local-vol surface
- Interpret `app_simp.py` anomaly list / API payload
- Diagnose butterfly violations, Dupire explosions, term humps, or extreme skew/butterfly **levels**
- Relate surface anomalies to VVIX / SKEW / 3M corr context

Do **NOT** use for:
- Day-over-day surface-delta PCA or sentiment scoring → use `surface-sentiment-analysis` instead
- Multi-index (NDX/DJI) workflows — this skill is **SPX-only**
- Trade recommendations or price targets

## Scope constraints

| In scope | Out of scope |
|----------|--------------|
| Today's live K/S × DTE surface | Cross-session surface deltas |
| Raw IV, smooth IV, Dupire local vol | PCA PC1–PC3 sentiment |
| Structure levels (skew, butterfly, term hump) | Compass fear/greed score |
| Aux indices: VVIX, SKEW, 3M corr | NDX / DJI surfaces |

## Primary inputs

### A. Dashboard / API (`app_simp.py`)

```
GET /api/index/SPX
```

Key fields:

```json
{
  "date": "2026-07-30",
  "spot": 6400.0,
  "aiv": 16.2,
  "psk": 6.1,
  "tsl": 1.4,
  "bfly": 1.8,
  "vrp": 3.2,
  "vix": 15.5,
  "aux": {
    "vvix": 88.0,
    "vvix_pctl": 42.0,
    "skew_index": 142.0,
    "skew_pctl": 71.0,
    "cor3m": 28.5,
    "cor3m_pctl": 55.0
  },
  "anomalies": [
    {
      "kind": "smooth_residual",
      "surface": "iv",
      "ks": 0.92,
      "dte": 14.0,
      "value": 24.5,
      "baseline": 19.1,
      "score": 3.4,
      "detail": "Raw IV residual +5.4 vol pts vs smooth smile (z=+3.4)"
    }
  ],
  "anomaly_counts": {"smooth_residual": 2, "lv_invalid": 1},
  "surface_x": ["... K/S axis ..."],
  "surface_y": ["... DTE axis ..."],
  "surface_z": ["... raw IV grid ..."],
  "surface_sv": ["... smooth IV ..."],
  "surface_w": ["... local vol ..."]
}
```

`anomalies[]` is the primary object to narrate. Surface grids are for locating / confirming nodes.

### B. Notebook / library call

```python
from vol_surface import (
    VolSurfaceConfig,
    VolSurfaceStudy,
    build_iv_grid,
    smooth_iv_grid_quadratic,
    dupire_local_vol,
    detect_surface_anomalies,
    fetch_aux_vol_context,
    fetch_vix_context,
)

cfg = VolSurfaceConfig(underlying="US..SPX", max_dte=60)
study = VolSurfaceStudy(cfg)
study.fetch_live(save=False)

d = sorted(study.surfaces.keys())[-1]
df = study.surfaces[d]
spot = float(df["spot"].iloc[0])
g_dte, g_ks, iv = build_iv_grid(df, max_dte=cfg.max_dte)
iv_s = smooth_iv_grid_quadratic(df, g_dte, g_ks, iv)
lv = dupire_local_vol(spot, g_dte, g_ks, iv_s, r=cfg.risk_free_rate)

anomalies = detect_surface_anomalies(
    g_dte, g_ks, iv, iv_s, lv, study.features[d], cfg
)
aux = fetch_aux_vol_context()
vix = fetch_vix_context()
```

## Anomaly kinds (detector taxonomy)

Implemented in `vol_surface.detect_surface_anomalies` — **single session only**.

### 1. `smooth_residual` (surface=`iv`)

Raw IV minus per-DTE quadratic-smooth smile. Flagged when MAD-z ≥ ~3 **and** |residual| ≥ ~2 vol pts.

- Large positive residual: quote/liquidity spike or genuine wing demand not captured by smooth smile
- Large negative residual: stale/cheap node or interpolation artifact

### 2. `local_spike` (surface=`iv`)

Node vs mean of 4-neighbors (adjacent DTE / K/S). Flagged when MAD-z ≥ ~3 **and** |dev| ≥ ~2.5 vol pts.

- Isolated diamond on an otherwise smooth ridge → check OI/volume and nearby strikes
- Cluster along one DTE → expiry-specific event / pin risk

### 3. `lv_invalid` (surface=`lv`)

Interior Dupire local vol is NaN after sanitize → butterfly violation (`∂²C/∂K² ≤ 0`) or bad denom.

- Often data noise at sparse wings; more serious if clustered near ATM mid-tenor

### 4. `lv_explosion` (surface=`lv`)

Local vol ≫ implied vol (ratio ≥ ~2×) or absolute LV ≥ ~120%.

- Numerical instability from thin smile curvature **or** genuinely peaked local vol (event density)

### 5. `term_hump` (surface=`structure`)

ATM term structure local maximum in the event-hump DTE window (`detect_term_hump`).

- Classic near-dated event pricing (FOMC, CPI, earnings basket)

### 6. `extreme_skew` / `extreme_butterfly` (surface=`structure`)

Level thresholds on `skew_25d` / `butterfly_25d` (defaults ~8 / ~4 vol pts) — **levels**, not 5d changes.

## Auxiliary index playbook

Use these to decide whether anomalies are **idiosyncratic surface noise** or **macro-consistent**.

| Index | Ticker (best-effort) | High reading means | If anomalies agree |
|-------|----------------------|--------------------|--------------------|
| VVIX | `^VVIX` | Vol-of-vol elevated; vol surface unstable | Expect more IV spikes / LV noise |
| CBOE SKEW | `^SKEW` | Fat left-tail priced in SPX | Expect put-wing residuals / steep `psk` |
| 3M corr | `^COR3M` | Constituents move together | Broad risk-off; anomalies near ATM/level more plausible |

Percentiles in `aux.*_pctl` are ~63-session ranks when available.

**Consistency rules:**
- High VVIX + many `local_spike` / `lv_*` → treat as regime noise; demand clusters, not single-node stories
- High SKEW index + `extreme_skew` / put-side `smooth_residual` → coherent crash premium
- Low corr + localized wing spike → name-specific / liquidity artifact more likely than index stress
- Term hump + calm VVIX → calendar event, not vol-of-vol blowup

## Narrative template

Write analysis in this order:

1. **Snapshot** — date, spot, ATM IV, VIX, VRP, skew, term slope, butterfly
2. **Aux context** — VVIX / SKEW / 3M corr levels + percentiles; one sentence on regime
3. **Anomaly inventory** — count by `kind`; list top 3–5 by `|score|` with K/S and DTE
4. **Spatial pattern** — isolated nodes vs ridge along DTE vs wing band vs ATM term hump
5. **IV vs LV** — do LV holes/explosions coincide with IV residuals? (yes → shape problem; only LV → Dupire sensitivity)
6. **Macro consistency** — anomalies vs VVIX/SKEW/corr (coherent vs idiosyncratic)
7. **Caveats** — sparse OI wings, single-expiry LV unavailable, 0DTE gamma, Yahoo aux ticker gaps

### Example narrative

"SPX session 2026-07-30: ATM 30d IV 16.2%, VIX 15.5, VRP +3.2. VVIX at 42nd %ile (quiet vol-of-vol); CBOE SKEW 71st %ile. Detector flags 6 anomalies: 2× `smooth_residual` on the 14d 0.90–0.92 put wing (+5 vol pts vs smooth), 1× `term_hump` near 14d, 2× interior `lv_invalid`, 1× `extreme_skew` (psk=9.1). Pattern is a front-end put-wing ridge plus a mild ATM term hump — consistent with elevated SKEW and a near-dated event, not a VVIX-driven chaos regime. LV NaNs sit next to the same wing — treat as butterfly-fragile quotes rather than a separate local-vol story."

## What NOT to do

- Do NOT invent day-over-day surface moves; this stack is live-only
- Do NOT revive sentiment / compass scores for this workflow
- Do NOT treat every `lv_invalid` as tradable arb — often grid/sanitize artifacts
- Do NOT ignore aux percentiles when claiming "extreme" macro stress
- Do NOT expand to NDX/DJI unless the user explicitly changes scope

## Codebase references

- `app_simp.py` — SPX-only dashboard; anomaly list + aux strip; `/api/index/SPX`
- `vol_surface.py` — `detect_surface_anomalies`, `fetch_aux_vol_context`, `build_iv_grid`, `smooth_iv_grid_quadratic`, `dupire_local_vol`, `detect_term_hump`
- `skills/surface-sentiment-analysis/` — separate skill for PCA surface-**delta** sentiment (multi-day cache)
