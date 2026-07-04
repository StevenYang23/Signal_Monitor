# Surface Sentiment Analysis

Analyze equity index option implied volatility (IV) surfaces to read market sentiment, detect positioning shifts, and identify tail-risk events. Uses PCA decomposition of delta-space IV surface changes to separate noise from structurally meaningful signals.

## When to use

Call this skill when the user's goal involves understanding market sentiment from options market data: volatility surface dynamics, put/call skew changes, term structure shifts, or risk-premium repricing.

Do NOT use for predicting short-term price direction, recommending option trades, or when the user is working with a different asset class that lacks a liquid options market.

## Data sources

The analysis runs on daily cached IV surfaces stored as parquet files under `research/data/vol_surface/`. Each file contains the option chain for one session after cleaning (moneyness filter, OI filter, IV bounds).

### Primary input: `SurfaceDeltaPCA.build_report()` output

```
Surface PCA sentiment  |  date: 2026-07-03

Explained variance:  PC1=78.3%, PC2=9.2%, PC3=2.8%, total=90.3% (PC1-3)
Fitted on 63 sessions

Per-PC signals:
  PC1: z=+1.32, intensity=mild_positive, pctile=91st
  PC2: z=+2.05, intensity=strong_positive, pctile=98th
  PC3: z=-0.41, intensity=neutral, pctile=34th

Joint flags: risk_off_surge

Surface features:
  atm_iv_30d: 18.45
  skew_25d: 6.32
  term_slope: +1.20
  vix: 17.80
```

### Download and run

```python
from data_fetcher import RateLimiter, daily_pipeline
from surface_sentiment import SurfaceDeltaPCA

result = daily_pipeline(RateLimiter(600))
study = result["study"]

pca = SurfaceDeltaPCA()
sent = pca.sentiment(study.surfaces)
report = pca.build_report(sent, study.features[list(study.features.keys())[-1]])

# report is the primary input for analysis
print(report)
```

### Secondary inputs (for deeper analysis)

- **Compass Gauge speedometer**: A dual-panel representation displaying aggregate sentiment as a Fear/Greed-style speedometer index (ranging from Extremely Bearish to Extremely Bullish) alongside structured market summary bullets.
- **Local vol heatmap / Moneyness surface**: `study.plot_local_vol_heatmap()` or gridding over K/S (moneyness) and DTE coordinates to show where local vol spikes occur.
- **PCA loading heatmaps**: `pca.plot_loadings(result)` shows what each PC looks like as a surface.
- **Term structure**: `study.plot_term_structure()` shows ATM IV across expiries over time.

## Methodology — PCA on IV surface changes

### Why surface deltas, not levels

IV surface levels are non-stationary (vol clusters, long-term drift). PCA on levels is dominated by the current vol regime, not sentiment changes. PCA on **deltas** (today's surface minus a rolling baseline) isolates shocks to the surface - this is where sentiment lives.

The pipeline:
1. **Grid construction**: For each session, `build_iv_grid_delta()` builds a (DTE x delta) mesh covering 7 DTE values x 21 delta values = 147 grid nodes. Only OTM options are used (delta < 0 = puts, delta > 0 = calls) with a small ATM buffer for smooth interpolation.
2. **Feature vector**: Each session's 147 IV values are flattened into a 147-dimensional vector.
3. **Delta computation**: For each session, delta = surface_vector[t] - rolling_21d_average(excluding latest 5 sessions). This isolates recent changes from baseline.
4. **Standardization**: Each of the 147 grid nodes is z-scored across time, so a 1-standard-deviation move at each node is comparable.
5. **PCA**: 4-component PCA on the standardized delta matrix.

### What each PC captures

The loadings (reshaped back to DTE x delta) have stable, interpretable shapes:

**PC1 — Level shift (typically 75-85% of variance)**
- Nearly uniform loading across all delta values
- Positive loading everywhere: a parallel shift up in IV across all strikes and expiries
- Z-score interpretation:
  - z > +2.0: Broad risk-off repricing. Hedging demand surging across the entire surface. Usually accompanied by spot decline.
  - z < -2.0: Broad vol compression. Complacency or absorption of risk. Often precedes sustained rallies.
  - z within +/- 1.0: Normal noise; no regime signal from level alone.

**PC2 — Skew/tilt (typically 6-10% of variance)**  
- Negative loading on left wing (delta < 0, puts), positive on right wing (delta > 0, calls)
- Zero-crossing near delta = 0 (ATM)
- Z-score interpretation:
  - z > +2.0: Put skew widening sharply. Crash-concern premium being added. OTM puts getting expensive relative to OTM calls. Bearish signal for near-term equity returns (Xing, Zhang & Zhao 2010).
  - z < -2.0: Put skew compressing. Calls relatively richer vs puts. Can reflect short-squeeze dynamics, FOMO, or genuine optimism. Cremers & Weinbaum (2010) show this predicts positive equity returns.
  - z within +/- 1.0: Normal skew fluctuations.

**PC3 — Smile/curvature (typically 2-4% of variance)**
- Symmetric U-shape: positive loading on both wings, negative near ATM (or vice versa)
- Captures butterfly/curvature changes
- Z-score interpretation:
  - z > +2.0: Both wings getting relatively expensive vs ATM. Uncertainty about tail outcomes in both directions. Divergent risk scenarios (e.g., binary event, earnings, macro data).
  - z < -2.0: Wings compressing relative to ATM. Market comfortable that tails are well-understood. Low uncertainty mode.
  - When PC2 > +2.0 AND PC3 > +2.0 simultaneously: genuine tail-risk event, not just one-sided hedging.

**PC4+ (typically < 2% each)**
- Higher-order patterns that are often period-specific.
- Can capture term-structure-specific patterns (front-end vs back-end vol).
- Treat with caution: often noise or overfitting to the training window.

### Joint signal interpretation

Individual PC z-scores are informative; their joint behavior is where real insight lives.

| Pattern | PC1 | PC2 | PC3 | Interpretation |
|---------|-----|-----|-----|----------------|
| Risk-off surge | >+1.5 | >+1.5 | any | Classic protection-buying. SPX puts + broad vol up. Stress mode. |
| Risk-on compression | <-1.5 | <-1.5 | any | Vol collapsing, skew flattening. Complacency or calm. Can precede reversals. |
| Tail uncertainty | >+1.5 | any | >+1.5 | Both wings expensive. Binary outcome pricing (election, data, earnings). |
| Pure put skew | <+1.0 | >+2.0 | <+1.0 | Put demand without vol shock. Targeted hedging or tactical positioning. |
| Pure call skew | <+1.0 | <-2.0 | <+1.0 | Call demand without vol collapse. Can reflect positive gamma positioning. |
| Butterfly squeeze | <+1.0 | <+1.0 | >+2.0 | Strange: both wings up but level and skew flat. Look for liquidity artifacts or market-maker positioning. |

### Cross-referencing with surface features

Combine PC signals with traditional surface features for richer context:

- **atm_iv_30d > 25**: Absolute vol is elevated regardless of PC signals
- **skew_25d > 8**: Put skew is structurally steep
- **term_slope > +2**: Front-end vol elevated (near-term stress)
- **term_slope < -2**: Inverted term structure (immediate concern)
- **vix > vix_5d_ago + 3**: VIX surging independently — macro-driven stress
- **vrp (vix - rv_22) > 5**: VRP very wide — options expensive relative to realized. Potential sell signal for premium.

## Narrative building

A good sentiment analysis follows this structure:

1. **Current state**: What is the vol level? Flat, elevated, suppressed relative to recent range?
2. **Direction of change**: What moved today? Level, skew, curvature, or a combination?
3. **PCA signal**: Which PCs are firing? How strong? What's the joint pattern?
4. **Attribution**: Is this macro-driven (PC1 dominant) or positioning-driven (PC2 dominant)? Event-specific (PC3)?
5. **Scenario**: Is the signal consistent with the spot price action? Volume confirmation?
6. **Caveats**: What could make this signal wrong? Liquidity regime? Seasonality? Expiry dynamics?

### Example narratives

**Example A — Risk-off surge**

"PC1 and PC2 both firing at the 98th percentile simultaneously. This is a classic risk-off surge: broad IV repricing + put skew widening. The joint pattern suggests institutional hedging rather than speculative put buying. Term structure shows front-end stress with term_slope at +3.1. VIX up 4.2 pts from 5d ago. This is a high-conviction bearish sentiment reading, consistent with the spot break below the 50-day moving average."

**Example B — Pure put skew**

"PC2 at the 95th percentile, PC1 at the 60th percentile. Level unchanged but put skew has widened sharply. This suggests targeted put buying rather than broad risk-off. Likely large block hedges rolling forward (SPX puts expiring, being replaced). Without PC1 confirmation, this is a moderate concern signal. Watch tomorrow: if PC1 joins, it becomes structural."

**Example C — Butterfly squeeze**

"PC3 at the 97th percentile, PC1 and PC2 neutral. This is unusual: both OTM puts and OTM calls getting richer while ATM pricing is flat. Consistent with event pricing (binary outcome). Check the calendar: economic data release or FOMC within the next 3 sessions."

**Example D — Complacency**

"PC1 at the 4th percentile, PC2 at the 8th percentile. Vol compression at extreme levels. VIX at the 15th percentile of its 6-month range. Skew flat. This pattern in isolation is a contrarian warning signal — option market pricing no risk premium, which historically precedes vol spikes. Seek spot price confirmation: extended rally on low volume would strengthen the contrarian call."

## What NOT to do

- Do NOT use the PCA sentiment report as a standalone trade signal. It is a descriptive framework for understanding market positioning.
- Do NOT interpret PC1 alone in isolation for directional bets. PC1 + PC2 jointly is more reliable.
- Do NOT attribute a single cause to a joint signal without checking spot price action, volume, and calendar context.
- Do NOT over-interpret PC4+. These components capture under 2% of variance and are often training-window-specific.
- Do NOT use the PCA report during expiry weeks without noting that gamma dynamics distort IV readings, especially near 0DTE.
- Do NOT report the PCA z-scores as if they were p-values. A z-score of +3.0 in a 63-session window is extreme but not impossible due to randomness.

## References

### Academic

- Cont & Fonseca (2002) "Dynamics of Implied Volatility Surfaces". *Quantitative Finance*. PCA decomposition of S&P 500 IV surfaces.
- Cremers & Weinbaum (2010) "Deviations from Put-Call Parity and Stock Return Predictability". *Journal of Financial Economics*.
- Xing, Zhang & Zhao (2010) "What Does the Individual Option Volatility Smirk Tell Us About Future Equity Returns?" *Journal of Financial and Quantitative Analysis*.
- Bollen & Whaley (2004) "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?" *Journal of Finance*.
- Gatheral (2006) "The Volatility Surface: A Practitioner's Guide". Wiley.
- Fengler (2012) "Option Data and Modeling: Some Empirical Regularities". *Handbook of Computational Finance*.

### Codebase references

- `vol_surface.py` — `build_iv_grid()` (K/S for visualization), `build_iv_grid_delta()` (delta for PCA)
- `surface_sentiment.py` — `SurfaceDeltaPCA` class with extract_features, compute_deltas, fit, sentiment
- `data_fetcher.py` — `RateLimiter(600)`, `SafeFetchSession`, `daily_pipeline()` for safe API calls
