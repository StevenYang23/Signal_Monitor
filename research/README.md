# SPX Volatility Regime Methodology

This research studies whether SPX can be separated into two practical market regimes:

- **Low-vol / slow-bull regime:** lower realized and implied volatility, positive trend, smaller drawdowns, and fewer daily direction changes.
- **High-vol / choppy regime:** higher volatility, larger drawdowns, weaker or unstable trend, and more frequent up/down price swings.

The reusable implementation is in `volatility_regime.py`. The notebook `spx_study.ipynb` is a thin four-cell example that calls the class.

## Data

The notebook downloads daily market data from Yahoo Finance with `yfinance`.

Default series:

- `^GSPC`: S&P 500 index level.
- `^VIX`: 30-day implied volatility.

These can be changed through the class parameters:

- `underly_ticker`: underlying price ticker, default `^GSPC`.
- `vol_ticker`: implied volatility ticker, default `^VIX`.
- `train_window`: rolling HMM training window, default `756` trading days.
- `refit_step`: HMM refit interval, default `22` trading days.

The notebook downloads historical data first, then `train()` builds the required HMM features internally before fitting the walk-forward model.

## Feature Engineering

Daily SPX log returns are computed first. Realized volatility is then annualized into percent units so it can be compared directly with VIX:

```text
RV = rolling_std(log_return) * sqrt(252) * 100
```

The notebook only computes features needed by the HMM. It does not compute PnL, forward returns, or return-based evaluation metrics.

## Model Input Features

The active regime classifier is the HMM. It uses this standardized feature set:

- `VIX`: 30-day implied volatility from listed SPX options. It helps because high-vol regimes usually appear first as elevated option-implied uncertainty, not only as realized price movement. This follows the Cboe VIX interpretation as a forward-looking volatility measure.
- `RV_22`: one-month realized volatility using 22 trading days. It helps measure the volatility the market has actually experienced recently, which is central to separating calm regimes from turbulent regimes.
- `RV_63`: three-month realized volatility. It helps smooth noisy short-term volatility and gives the HMM a slower volatility context, so the model is less dependent on one brief spike.
- `lagged_VRP`: prior-month implied volatility minus current one-month realized volatility. It helps compare what option markets recently implied against what volatility actually became.
- `trend_63`: 63-trading-day SPX price trend. It helps distinguish a low-vol slow-bull regime from a low-vol but weak or deteriorating market. Slow-bull regimes should generally have positive medium-term trend.
- `drawdown_252`: distance from the rolling one-year high. It helps because high-vol regimes often coincide with deeper drawdowns and stress periods, while slow-bull regimes tend to stay closer to highs.
- `sign_flip_22`: frequency of daily return direction changes over 22 trading days. It helps capture choppiness: a market that alternates up/down frequently can feel high-risk even if the net return over the window is small.

These inputs describe implied volatility, realized volatility, volatility risk premium, trend, drawdown, and choppiness. The input set is intentionally compact to reduce instability in rolling walk-forward HMM fits.

## Regime Methods

The notebook uses a two-state Gaussian Hidden Markov Model as the regime classifier.

To avoid the instability seen in earlier experiments, the notebook uses:

- A longer rolling training window.
- 22-trading-day refits instead of daily one-step refits.
- Multiple random seeds.
- Diagonal covariance and minimum covariance regularization.
- Regime labels based on realized volatility and choppiness rather than raw HMM state IDs.

The HMM uses its transition matrix internally when decoding the most likely state path, so the classification accounts for regime persistence instead of classifying each day independently.

## Display

The notebook display is intentionally minimal. It plots SPX with HMM `high_vol` periods shaded, plus `RV_22` and `VIX` below the price chart.

Before plotting, `plot()` refreshes the latest available intraday price for both the underlying ticker and volatility ticker. This updates the final chart point during trading hours without retraining the historical HMM model. Set `display_period=None` to show the full history or use an integer such as `22` to show only the most recent rows.

This keeps the research focused on regime identification rather than strategy backtesting. Any PnL, forward-return, hit-rate, or drawdown evaluation should be handled separately if a trading rule is later added.

## Recommendation Logic

The HMM is the preferred framework because it explicitly models regime persistence through transition probabilities and keeps the regime definition focused on market state rather than PnL.

In practice:

- Use the HMM regime as the primary regime filter.
- Treat persistent `high_vol` classifications as a choppy/risk-management signal.
- Treat persistent `low_vol` classifications as the slow-bull regime candidate.
- Review the chart for excessive regime flipping; if it flips too often, increase the training window, refit interval, or simplify the input feature set.

## References

The notebook methodology is based on standard volatility-regime and market-regime detection ideas. These references are useful background for the design choices:

- QuantStart, [Market Regime Detection using Hidden Markov Models in QSTrader](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/). Practical reference for using HMMs as market regime filters.

- Nystrup, Kolm, and Lindstrom, [Regime-Switching Factor Investing with Hidden Markov Models](https://www.mdpi.com/1911-8074/13/12/311). Academic-style reference for applying HMMs to identify market regimes and adapt investment behavior.

- LSEG Developers, [Market Regime Detection using Statistical and ML Based Approaches](https://developers.lseg.com/en/article-catalog/article/market-regime-detection). Practical reference for comparing statistical and machine-learning approaches to market regime detection.

These sources motivate the feature families and model families, but the exact implementation in `spx_study.ipynb` is an empirical design for this project rather than a direct replication of any single paper.
