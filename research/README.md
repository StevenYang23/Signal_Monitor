# Volatility Regime Methodology

This research studies whether major US equity indices can be separated into two practical market regimes:

- **Low-vol / slow-bull regime:** lower realized and implied volatility, positive trend, smaller drawdowns, and fewer daily direction changes.
- **High-vol / choppy regime:** higher volatility, larger drawdowns, weaker or unstable trend, and more frequent up/down price swings.

The notebook `vol_regime_study.ipynb` is a self-contained walkthrough aligned with `regime_study_v2.ipynb`: download data, build `lagged_VRP`, run a daily walk-forward HMM, backtest a long-only rule, and plot results. A reusable class implementation lives in `volatility_regime.py` at the repo root for production-style workflows (`SPX_signal.ipynb`) and optional feature-set ablation.

## Notebook Structure

`vol_regime_study.ipynb` has four code cells:

1. **Download data** — pull daily closes from Yahoo Finance for SPX, DJI, and NSDQ with their vol indices.
2. **Feature engineering** — build `lagged_VRP` only (plus price/vol columns for plots).
3. **Training** — daily walk-forward HMM; fit on prior 2 years excluding today; score today OOS.
4. **Display** — three-panel charts (price, IV, cumulative log-return strategy vs buy-and-hold).

## Data

Daily market data comes from Yahoo Finance (`yfinance`).

| Label | Underlying | Implied vol |
|-------|------------|-------------|
| SPX   | `^SPX`     | `^VIX`      |
| DJI   | `^DJI`     | `^VXD`      |
| NSDQ  | `^NDX`     | `^VXN`      |

**Inner join on dates:** underlying and vol series are merged with `join="inner"` so holiday mismatches (e.g. VIX row without SPX) do not corrupt rolling features.

Key parameters:

- `DATA_PERIOD`: `"max"`
- `LIVE_START`: backtest start in the notebook (e.g. `2015-01-01`; `volatility_regime.train()` default is `2000-01-01`)
- `LIVE_END`: backtest end (`None` = latest available in notebook)
- `TRADING_DAYS`: `252`

## Feature Engineering

The HMM uses a single input, **`lagged_VRP`**, matching `regime_study_v2.ipynb`:

```text
IV           = VOL                         # implied vol, annualized %
RV_22        = rolling_std(log_return, 22) * sqrt(252) * 100
lagged_VRP   = IV.shift(22) - RV_22
```

`IV` and `RV_22` share the same percent units (e.g. VIX ≈ 17 lines up with RV_22 ≈ 15).

## Regime Methods

Two-state Gaussian HMM (`hmmlearn`), daily walk-forward:

| Parameter | Default |
|-----------|---------|
| `train_window` | `504` (2 years) |
| `refit_step` | `1` (daily refit) |
| `threshold_today` / `threshold_tomorrow` | `0.5` |
| `hmm_n_iter` | `500` |
| `hmm_tol` | `1e-1` |
| `transmat_prior` | `1.0` |

**Training (each day):**

- Fit on `iloc[loc - 504 : loc]` — prior 2 years, **today excluded**.
- Score today OOS with `predict_proba` on `iloc[loc - 503 : loc + 1]`; take the last row.

**State labeling:** lower HMM mean on `lagged_VRP` → low-vol (calm) state.

**Trade signal:**

```text
signal = (prob_low_vol >= 0.5) & (prob_low_vol_tmr >= 0.5)
```

`prob_low_vol_tmr` uses the fitted transition matrix (one-step-ahead low-vol probability).

## Today's signal (live vs backtest)

Two workflows share the **same per-day scoring math** but differ in how often the model is refit:

| | `vol_regime_study.ipynb` | `volatility_regime.py` (`SPX_signal`) |
|---|--------------------------|----------------------------------------|
| Purpose | Historical backtest | Live monitor |
| Refit | Every trading day | Once per `display()` / `fit_once()` |
| Data fetch | Full history (`max`) | ~2.5y (`630d` in `signal_mode`) |
| Today's row | Last day of walk-forward loop | `fit_once(exclude_today=True)` |

**Per-day scoring (identical in both):**

1. Fit HMM on `lagged_VRP` for the prior **504 days**, ending yesterday.
2. Run `predict_proba` on the **504-day window ending today**; take the last row.
3. Map states: lower Gaussian mean → low-vol.
4. `signal = 1` iff `P(low today) ≥ 0.5` **and** `P(low tomorrow) ≥ 0.5`.

`fit_once()` produces the same probabilities as the notebook's last backtest day when run on the same prices (verified on SPX).

**Interpretation:** at today's close, the signal answers whether to be long **tomorrow**, using only information available through today. PnL in the notebook applies `signal[t]` to `return[t → t+1]`.

**`display()`** updates only **today's** SPX/VIX quotes on repeat calls (initial call downloads ~2.5y history), then runs `fit_once()`.

## Strategy PnL

Long-only diagnostic backtest:

- Long when `signal = 1`; flat otherwise.
- PnL uses **next-day log return**: `log(price_{t+1} / price_t)` applied to the signal at `t`.
- Charts show cumulative log return; no transaction costs.

## Display

Per market, three panels:

1. Underlying price with high-vol periods (`signal = 0`) shaded red.
2. IV and 22-day realized vol (both annualized %).
3. Cumulative log return: regime strategy vs buy-and-hold.

Signals are saved under `research/Saved_Signal/regime_{market}.csv`.

## Relation to `volatility_regime.py`

`HMMVolatilityRegime` implements the same methodology as `vol_regime_study.ipynb`:

- Inner-join price/vol download; `lagged_VRP` formula above.
- Defaults: `train_window=504`, `refit_step=1`, thresholds `0.5`, `live_start="2000-01-01"`.
- `train()` — full walk-forward backtest over `live_start`…`live_end`.
- `fit_once(exclude_today=True)` — fit on prior 504 days, score latest bar OOS.
- `display(display_period=22)` — update **today's** SPX/VIX only (full download on first call), then `fit_once` and plot.
- `from_market("SPX", signal_mode=True)` — auto-sets `data_period="630d"` (~2.5y fetch for 2y fit + warm-up).
- `add_strategy_pnl()`, `return_summary()`, `plot()`, `evaluate_feature_sets()` for extended experiments.

Use the notebook for multi-market research plots. Use `volatility_regime.py` for daily signal generation and class-based workflows.
