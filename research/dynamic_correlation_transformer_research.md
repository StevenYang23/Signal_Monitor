# Dynamic Correlation Transformer for SPX Return Prediction

## Executive Summary

The best fit for the current `research/cor_study.ipynb` workflow is a compact **Dynamic Feature-Correlation Transformer**: a transformer that treats each feature as a token, learns feature-to-feature attention, and conditions that attention on rolling correlation structure and market regime. The model should predict the next 22-trading-day SPX log return from daily windows of SPX, volatility, skew, and derived risk-premium features.

The main recommendation is not a large vanilla transformer over daily timestamps. With roughly 10 years of daily data, a timestamp-token transformer can overfit and produce hard-to-interpret attention. A feature-token transformer, inspired by iTransformer, is a better match because the research goal is explicitly to capture dynamic correlation between features.

## Current Notebook Context

The existing notebook builds a daily feature matrix from Yahoo Finance:

- Raw market features: `VIX`, `SPX`, `SKEW`, `VIX9D`, `VVIX`, `SDEX`.
- Derived features: `SPX_ret`, `RV`, `VRP`, `lagged_VRP`, `vvix_vix`, `skew_sdex`.
- Current label: `Label_ret = SPX.shift(-22) - SPX`.

The target should be changed for modeling research to:

```python
df["Label_fwd_22d_log_ret"] = np.log(df["SPX"].shift(-22) / df["SPX"])
df["Label_fwd_22d_direction"] = (df["Label_fwd_22d_log_ret"] > 0).astype(int)
```

The absolute SPX price difference is non-stationary. A forward log return is more stable across regimes and maps naturally to trading evaluation.

## Literature Findings

### iTransformer

iTransformer argues that standard time-series transformers often use each timestamp as a token, mixing all variables at the same time into one embedding. That can obscure variable-specific behavior and produce less meaningful attention maps. iTransformer instead embeds the full lookback history of each variable as a token, then applies attention across variable tokens to capture multivariate correlations.

For this project, that design is directly relevant: the target is one SPX return, and the key question is which feature relationships matter dynamically. Treating `VIX`, `RV`, `VRP`, `VVIX`, `SKEW`, and related features as tokens makes attention maps interpretable as feature-correlation structure.

Reference: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625).

### MASTER

MASTER is a market-guided stock transformer for stock prediction. It highlights two finance-specific issues:

- Correlations are dynamic, momentary, and can be cross-time rather than static.
- Feature effectiveness changes with market state.

MASTER addresses this with market-guided feature gating and alternating temporal and correlation aggregation. In the current SPX setup, there is only one target asset, so the stock-to-stock correlation module should be adapted into a feature-to-feature correlation module. The market gating idea is still highly useful: VIX, RV, VRP, and volatility ratios can condition which features receive more weight.

Reference: [MASTER: Market-Guided Stock Transformer for Stock Price Forecasting](https://arxiv.org/html/2312.15235v1).

### Stockformer and Finance-Specific Transformer Work

Stockformer frames stock prediction as multivariate-to-one forecasting and emphasizes finance-specific design choices: using transformed return targets, comparing against simple baselines, tracking direction and trading metrics, and being careful with learning rates because transformers can be unstable on small financial datasets.

For this project, the practical lesson is to keep the model small, validate chronologically, and evaluate both statistical error and trading relevance.

Reference: [Transformer Based Time-Series Forecasting For Stock](https://arxiv.org/html/2502.09625v1).

## Proposed Architecture

### Problem Definition

Given a lookback window of length `L` trading days and `F` features:

```text
X_t shape: [L, F]
y_t: 22-trading-day forward SPX log return
```

The model predicts:

```text
return_hat_t: scalar regression output
direction_hat_t: optional probability that forward return > 0
```

Recommended initial values:

- `L`: test `66`, `126`, and `252`.
- `F`: start with the notebook features after removing future labels.
- Prediction horizon: `22` trading days.
- Frequency: daily observations.

### Module 1: Window Builder

For each date `t`, build a window from dates `[t-L+1, t]`, then predict the return from `t` to `t+22`.

Important constraints:

- All feature engineering must use information available at or before `t`.
- Scaling must be fit only on the training period in each split.
- Labels near the end with unavailable `t+22` values must be dropped.

### Module 2: Feature-Token Encoder

Instead of representing each timestamp as a token, represent each feature as a token:

```text
Input window:       [batch, L, F]
Transpose:          [batch, F, L]
Feature projection: [batch, F, d_model]
```

Each feature token contains its own recent history. A small MLP or 1D convolution can project the length-`L` history into `d_model`.

Example:

```python
feature_tokens = history_projection(x.transpose(1, 2))
```

This lets attention learn relationships such as:

- `VIX` with `RV`.
- `VRP` with future SPX returns.
- `VVIX/VIX` with volatility-regime changes.
- `SKEW/SDEX` with downside-risk pricing.

### Module 3: Dynamic Correlation Prior

For each input window, compute feature correlation matrices:

```text
Pearson correlation:  [batch, F, F]
Spearman correlation: [batch, F, F]
EWMA correlation:     [batch, F, F]
```

The simplest first implementation should use Pearson or EWMA Pearson. Spearman can be an ablation because it is slower and may be noisier with small windows.

The correlation prior can enter the transformer in two ways:

1. **Attention bias**: add a learned transformation of the rolling correlation matrix to the attention logits.
2. **Correlation embedding**: encode each feature’s correlation row and add it to the feature token.

Recommended first version:

```text
attention_logits = QK^T / sqrt(d_head) + gamma * corr_bias
```

where `gamma` is a learned scalar initialized near zero. This lets the model start close to a normal transformer and learn whether the correlation prior helps.

### Module 4: Market-State Feature Gating

Build a regime vector from the latest available values in the window:

```text
m_t = [
    VIX_t,
    RV_t,
    VRP_t,
    VIX9D_t / VIX_t,
    VIX_t / VVIX_t,
    SKEW_t / SDEX_t,
    rolling_22d_SPX_ret_t,
]
```

The gating module maps `m_t` to `F` feature weights:

```text
gate = F * softmax(MLP(m_t) / temperature)
gated_tokens = feature_tokens * gate[:, :, None]
```

This adapts the model to regimes where different features matter. For example, realized volatility and volatility-risk premium may matter more during high-volatility regimes, while skew-related features may matter more around tail-risk repricing.

### Module 5: Feature-Correlation Transformer

Use a small stack of transformer encoder blocks over the feature tokens:

```text
Input:  [batch, F, d_model]
Output: [batch, F, d_model]
```

Recommended starting size:

- `d_model`: `32` or `64`.
- Layers: `1` or `2`.
- Heads: `2` or `4`.
- Dropout: `0.1` to `0.3`.
- Weight decay: small, such as `1e-4`.

The model should expose attention weights for diagnostics. The average attention matrix across validation/test windows can be compared with rolling empirical correlations.

### Module 6: Aggregation and Prediction Heads

Aggregate feature tokens into one market representation:

```text
pooled = attention_pool(transformer_output)
```

Then predict:

```text
return_hat = Linear(MLP(pooled))
direction_logit = Linear(MLP(pooled))
```

Recommended training objective:

```text
loss = huber_loss(return_hat, y_return) + lambda_dir * bce_loss(direction_logit, y_direction)
```

Use `lambda_dir = 0.1` or `0.2` initially. The regression target should remain primary; the direction head acts as an auxiliary stabilizer.

## Training and Validation Protocol

### Preprocessing

Use only past information for feature construction:

1. Download and align the raw series.
2. Compute daily log returns and rolling statistics.
3. Create ratios and volatility-risk-premium features.
4. Replace infinite values from ratios.
5. Drop rows with missing feature or label values.
6. Split chronologically.
7. Fit scalers on the training period only.
8. Generate rolling windows after scaling.

Recommended feature refinements:

- Use `log(VIX)`, `log(VVIX)`, and `log(SKEW)` or z-scored raw values.
- Use returns or changes for nonstationary price-like columns where appropriate.
- Consider replacing raw `SPX` level with `SPX_ret`, rolling return, moving-average distance, or realized trend features.

### Splits

Avoid random splits. Use one of:

- Fixed chronological split: train first 70%, validate next 15%, test final 15%.
- Expanding walk-forward split: train on early history, validate/test the next block, then roll forward.

Because the target horizon is 22 trading days, use an embargo around split boundaries if evaluating trading results. This reduces leakage from overlapping forward-return labels.

### Baselines

The transformer must beat simple baselines before it is useful:

- Zero-return forecast.
- Historical mean return forecast.
- Ridge or elastic net on latest features and rolling statistics.
- Random forest or gradient boosting if added later.
- LSTM/GRU over timestamp tokens.
- Vanilla timestamp-token transformer.
- iTransformer-style feature-token model without correlation bias.
- Full model with correlation bias and market gating.

### Metrics

Use both prediction and trading metrics:

- MAE and Huber/MSE on forward log return.
- Directional accuracy.
- Balanced accuracy if up/down classes are imbalanced.
- Information coefficient: correlation between prediction and realized forward return.
- Sign strategy return: `position = sign(prediction)`.
- Threshold strategy return: trade only when `abs(prediction)` exceeds a validation-selected threshold.
- Annualized return, volatility, Sharpe-like ratio, and max drawdown.
- Turnover and approximate transaction-cost sensitivity.

### Ablation Tests

Run these in order:

1. Feature-token model without correlation prior or gating.
2. Add rolling correlation attention bias.
3. Add market-state gating.
4. Add auxiliary direction head.
5. Compare lookbacks `66`, `126`, and `252`.
6. Compare Pearson vs EWMA correlation prior.
7. Compare raw current features vs stationarized features.

## Implementation Blueprint

### Notebook Structure

The current `research/cor_study.ipynb` can evolve into these sections:

1. Imports and configuration.
2. Data download.
3. Feature engineering.
4. Label construction.
5. Chronological split.
6. Scaling and window dataset.
7. Baseline models.
8. Dynamic Feature-Correlation Transformer.
9. Training loop.
10. Walk-forward or final holdout evaluation.
11. Attention and correlation diagnostics.

### Dataset Class

Suggested PyTorch dataset behavior:

```python
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, features, returns, lookback):
        self.features = features
        self.returns = returns
        self.lookback = lookback

    def __len__(self):
        return len(self.returns) - self.lookback + 1

    def __getitem__(self, idx):
        end = idx + self.lookback
        x = self.features[idx:end]
        y = self.returns[end - 1]
        direction = float(y > 0)
        corr = np.corrcoef(x, rowvar=False)
        return x.astype("float32"), corr.astype("float32"), np.float32(y), np.float32(direction)
```

The production version should handle constant columns in correlation calculation by replacing NaN correlations with zero and setting the diagonal to one.

### Model Skeleton

The model should be implemented as:

```text
DynamicCorrelationTransformer
    history_projection: projects each feature's lookback path
    regime_gate: maps latest regime vector to feature weights
    corr_bias_projection: maps rolling correlation matrix to attention bias
    encoder_blocks: small correlation-biased transformer stack
    pooling: attention pooling over feature tokens
    return_head: regression output
    direction_head: optional classification output
```

If PyTorch’s built-in `TransformerEncoderLayer` is used, attention bias and attention weight extraction are awkward. For this project, a custom small multi-head attention block is preferable because the correlation prior is central to the research question.

### Diagnostics

The final notebook should plot:

- Rolling realized forward returns versus predictions.
- Attention matrices averaged over validation/test windows.
- Correlation-prior matrices averaged over the same windows.
- Feature gate values across time.
- Strategy equity curve for sign and threshold strategies.

These diagnostics are important because the model’s claim is not just “better prediction”; it is “prediction helped by dynamic correlations.”

## Risks and Practical Notes

The biggest risk is sample size. Ten years of daily data gives roughly 2,500 observations before rolling-window loss. A transformer can overfit quickly. Keep the first model small and compare aggressively against linear baselines.

The second risk is label overlap. A 22-day forward return label means adjacent samples share most of their future return window. This can inflate metrics if validation is careless. Chronological splits and embargoed boundaries are important.

The third risk is interpretability drift. Attention is not automatically correlation or causality. Treat attention maps, rolling correlations, and ablations together. The model should only be described as using dynamic correlation if the correlation-prior ablation improves validation/test performance and the learned attention is stable enough to inspect.

## Recommended First Experiment

Start with:

- Lookback: `126` days.
- Target: 22-day forward log return.
- Features: stationarized versions of the current notebook columns.
- Model: feature-token transformer, `d_model=32`, `2` heads, `1` layer.
- Loss: Huber return loss plus small auxiliary direction BCE.
- Baselines: zero forecast, ridge regression, and vanilla feature-token model without correlation bias.
- Validation: chronological split with an embargo around boundaries.

Only add complexity after this first experiment beats the simple baselines on out-of-sample information coefficient and thresholded trading metrics.
