# SPX Volatility Regime Methodology

This research studies whether SPX can be separated into two practical market regimes:

- **Low-vol / slow-bull regime:** lower realized and implied volatility, positive trend, smaller drawdowns, and fewer daily direction changes.
- **High-vol / choppy regime:** higher volatility, larger drawdowns, weaker or unstable trend, and more frequent up/down price swings.

The main implementation is in `spx_study.ipynb`.

## Data

The notebook downloads daily market data from Yahoo Finance with `yfinance`.

Core series:

- `^GSPC`: S&P 500 index level.
- `^VIX`: 30-day implied volatility.
- `^VIX9D`: short-dated implied volatility.
- `^VVIX`: volatility of VIX.
- `^SKEW`: CBOE SKEW index.
- `^SDEX`: dispersion-related index, when available from Yahoo.

The study uses up to 20 years of available data, subject to each ticker's Yahoo Finance history.

## Feature Engineering

Daily SPX log returns are computed first. Realized volatility is then annualized into percent units so it can be compared directly with VIX:

```text
RV = rolling_std(log_return) * sqrt(252) * 100
```

Key features:

- `RV_21`: 21-trading-day annualized realized volatility.
- `RV_63`: 63-trading-day annualized realized volatility.
- `VRP_21`: volatility risk premium, computed as `VIX - RV_21`.
- `lagged_VRP_21`: lagged VIX minus current realized volatility.
- `trend_63` and `trend_126`: medium-term SPX trend.
- `ma_gap_50_200`: distance between 50-day and 200-day moving averages.
- `drawdown_252`: SPX drawdown from the rolling 252-day peak.
- `sign_flip_21`: 21-day frequency of daily return sign changes, used as a choppiness proxy.
- `vix9d_vix`, `vvix_vix`, and `skew_sdex`: volatility structure and risk-sentiment ratios where data is available.

Forward 5-day and 22-day SPX returns are also computed for regime evaluation.

## Regime Methods

The notebook compares three two-state regime classifiers.

### 1. Threshold Regime

The threshold method creates a composite volatility score from:

- 21-day realized volatility percentile.
- VIX percentile.
- 21-day sign-flip percentile.

It uses hysteresis to reduce noisy regime flipping:

- Switch from low-vol to high-vol when the score rises above the high threshold.
- Switch back to low-vol only when the score falls below the lower threshold.

This method is the most interpretable baseline.

### 2. Clustering Regime

The clustering method standardizes volatility, trend, drawdown, and choppiness features, then fits a two-component Gaussian Mixture model.

Because cluster labels are arbitrary, the notebook maps clusters into `low_vol` and `high_vol` by scoring each cluster on:

- Higher realized volatility.
- Higher sign-flip frequency.
- Lower trend.

The cluster with the higher combined stress/choppiness score is labeled `high_vol`.

### 3. HMM Regime

The HMM method fits a two-state Gaussian Hidden Markov Model on standardized regime features.

To avoid the instability seen in earlier experiments, the notebook uses:

- A longer rolling training window.
- Periodic refits instead of daily one-step refits.
- Multiple random seeds.
- Diagonal covariance and minimum covariance regularization.
- Regime labels based on realized volatility and choppiness rather than raw HMM state IDs.

The HMM is treated as a benchmark, not as the default answer. If it is unstable or has poor coverage, the conclusion should favor threshold or clustering.

## Evaluation

Each method is evaluated by comparing the behavior of `low_vol` and `high_vol` states.

Metrics include:

- Number and share of observations.
- Average regime duration.
- Transitions per year.
- Annualized return.
- Annualized volatility.
- Maximum drawdown.
- Daily hit rate.
- Average realized volatility and VIX.
- Average trend and drawdown.
- Average sign-flip frequency.
- Mean forward 5-day and 22-day returns.

The notebook also computes regime spreads such as:

- High-vol minus low-vol realized volatility.
- High-vol minus low-vol VIX.
- High-vol minus low-vol sign-flip frequency.
- Low-vol minus high-vol trend.
- Low-vol minus high-vol forward returns.

These spreads are used to judge whether a method actually captures the intended slow-bull versus choppy-volatility distinction.

## Recommendation Logic

The final method is selected with a practical score that rewards:

- Clear realized-volatility separation.
- Clear choppiness separation.
- Better low-vol trend relative to high-vol trend.
- Fewer noisy transitions.
- Adequate data coverage.

The preferred framework should be stable, interpretable, and useful out of sample. In practice:

- Use the recommended method as the primary regime filter.
- Use the other methods as confirmation checks.
- Treat agreement across threshold, clustering, and HMM as a stronger high-vol or low-vol signal.
- Treat disagreement as a reason to size decisions more conservatively.
