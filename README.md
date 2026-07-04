## Live Dashboard Demo / 实时看板预览

![Index Quant Signal Hub — SPX Dashboard](demo/signal_page.png)

*Index Quant Signal Hub — integrated 3D volatility surface, sentiment compass, HMM regime signals, VRP/term-structure metrics, and 1-sigma move targets for SPX, IXIC, and DJI.*

*Index Quant Signal Hub — 集成 3D 波动率曲面、情绪罗盘、HMM 机制信号、VRP/期限结构指标及 1-sigma 波动目标，覆盖 SPX、IXIC、DJI 三大指数。*

---

# Signal Monitor — Equity Volatility Regime & Option Sentiment Dashboard
# Signal Monitor — 美股波动率机制与期权情绪监测看板

An enterprise-grade research and live production dashboard designed to monitor and trade equity markets using Option Volatility Surface structure, Dupire Local Volatility, Principal Component Analysis (PCA) sentiment, and a 2-state Gaussian Hidden Markov Model (HMM) volatility regime model.

这是一个企业级的量化研究与实盘显示看板。它利用期权隐含波动率曲面（Option Volatility Surface）、Dupire局部波动率模型、主成分分析（PCA）情绪指标，以及基于两状态高斯隐马尔可夫模型（HMM）历史滚动训练的波动率机制分类模型，对美股主流指数（SPX、DJI、NDX）进行市场状态追踪与交易决策。

---

## Project Architecture / 项目内核结构

The system consists of interactive research walk-throughs in [research/](research/) and scalable production-oriented core library modules at the repository root:

系统主要由 [research/](research/) 目录下的深入探究 Notebook 及根目录下的模块化工程文件组成：

- **Volatility Surface & Grid Construction / 波动率曲面与网格构建**：
  [vol_surface.py](vol_surface.py) ingests option chain data, computes implied volatilities across a dynamic Moneyness ($K/S$) and Tenor ($DTE$) grid, and implements a Dupire Local Volatility solver to resolve risk-neutral local volatilities. It is studied in [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb).
  [vol_surface.py](vol_surface.py) 负责期权链数据的提取、隐含波动率表征，并在行权价偏离度（Moneyness, $K/S$）及到期期限（$DTE$）的双维网格上完成插值，同时配有 Dupire 局部波动率求解器。该部分的具体研究见 [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb)。

- **Principal Component Analysis (PCA) Vol Sentiment / 期权表面主成分分析与情绪量化**：
  [surface_sentiment.py](surface_sentiment.py) reduces the dimensionality of daily volatility surface changes, extracting standard variance-ratio modes (PC1 Level shift, PC2 Twist, PC3 Skew/Slope) and translating surface deformations into a single interactive Compass Bull/Bear Gauge.
  [surface_sentiment.py](surface_sentiment.py) 降维分析波动率曲面的每日动态德尔塔（Delta）移动，抓取主成分特征（如：PC1 整体平移、PC2 期限曲率或偏斜度），并结合 ATM 隐含波动率、偏斜（Skew）、期限结构斜率（Term Slope）转化为可交互的“罗盘（Compass）牛熊时速表”。

- **Markov Volatility Regime Filter / 隐马尔可夫波动率机制过滤器**：
  [volatility_regime.py](volatility_regime.py) implements rolling 2-state Gaussian Hidden Markov Models (HMM) to separate equity indexes into two practical regimes: "Low-vol / slow-bull" vs "High-vol / choppy". It is analyzed in [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb).
  [volatility_regime.py](volatility_regime.py) 采用滑动窗口（通常为 504 个交易日，即 2 年）来实时拟合两状态高斯隐马尔可夫模型，将指数划分为“低波缓牛机制”和“高波震荡机制”。具体的全样本回测见 [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb)。

- **Live Production App / 实时看板应用**：
  [app.py](app.py) consolidates core calculations into a lightweight, high-performance HTML/web server exposing terminal plots, live Futu API options fetch, synthetic fallback models, 3D surface meshes, and trade signal gauges.
  [app.py](app.py) 汇总所有量化计算模块，在本地暴露一个轻量化、高性能的交互式 Web 服务。支持实时对接富途 OpenAPI 提取最新期权，并提供备用合成曲面方案、3D 隐含与局部波交互图及罗盘时速表。

- **Robust Ingestion Engine / 数据获取引擎**：
  [data_fetcher.py](data_fetcher.py) guarantees rate-limited, crash-safe, and cached downloads from both Yahoo Finance and Futu API.
  [data_fetcher.py](data_fetcher.py) 提供健壮的数据抓取支持，实现频率限制、自动重试及本地 JSON 矩阵缓存。

---

## Methodology & Visualizations / 量化方法论与成果展示

### 1. Volatility Surfaces & 3D Local Volatility / 波动率曲面与 Dupire 局部波动率

In [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb), option chains are mapped onto continuous surfaces. By extracting the daily grid of Implied Volatility, the partial derivatives with respect to moneyness $M = K/S$ and time-to-decay $T = \text{DTE}/365$ are evaluated to construct a Dupire Local Volatility surface. This maps the instantaneous, state-dependent volatility of the index.

在 [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb) 中，期权链被转化并对应到连续的多维曲面。提取到连续隐含波动率网格后，对偏离度（Moneyness, $M = K/S$）及到期折算年（$T = DTE/365$）求偏导，代入 Dupire 局部波动率偏微分方程，以此解析出标的指数在不同价格和时间处的瞬时风险中性局部波动率状态。

#### Output Visualization / 曲面图示:
* **Implied Volatility (IV) Moneyness Surface / 隐含波动率曲面**:
  ![IV Surface](demo/IV_surface.png)
* **Resolved 3D Volatility Mesh / 3D 隐含与局部波动率曲面网格**:
  ![Vol Surface](demo/vol_surface.png)

---

### 2. Option Delta PCA & Sentiment Speedometer / 期权 Delta 主成分分析与罗盘时速表

To extract actionable sentiment from multiple cross-sections of options, [surface_sentiment.py](surface_sentiment.py) runs PCA on daily volatility changes across different expiries and strikes.
- **PC1 (Level Shift):** Captures general market panic or calmness.
- **PC2 (Twist & Skew Slope):** Captures changes in near-term hedging premiums vs. far-term expectations.

By combining the principal scores with ATM skewness, term slope, and the base VIX level, a **Compass Sentiment Speedometer** is created. This scores market anxiety from **Bearish (-100)** to **Bullish (+100)**.

为了从庞杂的各档期权变动中提取交易情绪，[surface_sentiment.py](surface_sentiment.py) 针对波动率期限和行权价偏离度的动态位移矩阵运行 PCA：
- **第一主成分 (PC1 - 水平平移):** 揭示整体市场避险情绪及恐慌程度。
- **第二主成分 (PC2 - 扭转及偏斜斜率):** 揭示远近月对冲溢价与远期宏观预期的强弱转移。

结合主成分特征得分、平价价差（ATM Skew）、到期期限结构斜率（Term Slope）以及 VIX level，计算出综合的 **罗盘情绪度量表（Compass Sentiment Speedometer）**，将市场当前情绪量化在 **极度看空 (-100)** 到 **极度看多 (+100)** 之间。

#### Sentiment Gauge Speedometer / 罗盘时速仪表盘:
![Compass Sentiment Gauge](demo/Compass.png)

---

### 3. HMM-based Volatility Regime Filtering / 隐马尔可夫机制概率分类

Standard Trend Following strategies suffer severe whipsaw losses during sudden crisis periods. In [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb), we build a 2-State Gaussian Hidden Markov Model (HMM) fitted exclusively on rolling 2-year windows (504 trading days) using a single feature: **`lagged_VRP`** (Volatility Risk Premium). 

$$RV22_t = \text{rolling std}(\log\text{ return}, 22) \times \sqrt{252} \times 100$$
$$\text{lagged\_VRP}_t = IV_{t-22} - RV22_t$$

**Walk-Forward Out-Of-Sample Scoring:**
1. Fit the Gaussian HMM on 504 historical days ending on $t-1$ (excluding today, $t$, to avoid leakages).
2. Determine state roles: the state with the lower mean of VRP is assigned as the **Calm (Low-Vol)** regime.
3. Use `predict_proba` to evaluate the OOS probabilities of today being in a low-vol regime, and tomorrow transitioning to a low-vol regime.
4. **Trading Rule:** Long the index tomorrow if and only if $P(\text{low\_vol}_t \ge 0.5)$ and $P(\text{low\_vol}_{t+1} \ge 0.5)$. Otherwise, go Flat.

在震荡走熊或危机降临期间，传统的持股或趋势跟踪策略往往面临严重的净值回撤。在 [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb) 中，本研究拟合了一个基于两状态高斯隐马尔可夫模型（HMM）的日频过滤策略。该模型仅使用一个衍生特征：**滞后波动率风险溢价（`lagged_VRP`）**：

$$RV22_t = \text{rolling std}(\text{对数日收益率}, 22) \times \sqrt{252} \times 100$$
$$\text{lagged\_VRP}_t = IV_{t-22} - RV22_t$$

**滚动跨期样本外概率打分：**
1. 选取截至 $t-1$ 日以前的滚动 504 天（约 2 年）数据训练 HMM 模型（严格剥离当天 $t$ 的数据，绝无未来偏误）。
2. 根据特征均值对隐含状态进行自动纠偏：均值较低的一方定义为 **低波平静状态**。
3. 调取 `predict_proba` 求解今日处于该平静状态的样本外后验概率 $P(\text{low\_vol}_t)$ 以及一阶转移矩阵下明日仍处于该状态的预测概率 $P(\text{low\_vol}_{t+1})$。
4. **交易规则：** 当且仅当两项低波概率判定皆 $\ge 0.5$ 时，明日做多标的（SPX, DJI, NDX）， 否则持币平仓。

#### Historical Outperformance & Regime Shading / 历史净值曲线与机制分类染色:
* **Market Regimes vs Shaded Crucial Crash Zones (Red = Crisis, High-Vol/Choppy) / 机制状态划定（红色代表高波震荡）**:
  ![SPX Regime Shading](demo/SPX_regime.png)
* **Backtest: Regime Strategy vs. Buy and Hold / 回测结果：机制策略 vs 标的持股**:
  ![SPX Regime Signals & EQ](demo/SPX_signal.png)

---

## Setup & Running Guide / 环境部署与运行指南

### Dependencies / 环境准备
Install the Python environment inside VS Code or in your shell:
在您的本地 Python 环境中安装必要依赖：

```bash
pip install numpy pandas matplotlib yfinance hmmlearn scikit-learn scipy plotly
```

If you have a Futu OpenAPI subscription and intend to pull real-time cash options chains, ensure your FutuOpenD setup is active on port `11111` or configure your settings in [vol_surface.py](vol_surface.py). Otherwise, the code will seamlessly auto-generate highly realistic simulated synthetics and cache metrics locally.

如果您配置了富途 OpenAPI 并在本地开通了 FutuOpenD (默认端口 `11111`)，本系统可直接获取美股实时多期限期权链。如未开启，系统将自动平滑切换至高度逼真的模拟合成波动面，并实现本地无缝缓存。

### 1. Run Interactive Notebook Studies / 交互式回测与模型调试
Open the notebooks inside [research/](research/) to explore the mathematics step-by-step:
在 VS Code 内部直接加载 [research/](research/) 目录下的 Jupyter Notebooks：
- **Volatility Surface & PCA Sentiment Study:** [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb)
- **HMM Walk-Forward Regime Transition Backtest:** [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb)

### 2. Run Live Production Dashboard / 启动本地交互式看板
Run the server from your terminal inside the project root:
在项目根目录下通过终端开启可视化平台：

```bash
python app.py
```
After initialization, open your browser and navigate to `http://localhost:8050` to inspect interactive 3D options surfaces, current Dupire local vol matrices, HMM state breakdowns, and consolidated global equity risk indicators.

启动后在浏览器访问 `http://localhost:8050` 便可无缝查看 3D 隐含波局部波曲面、HMM 后验概率走势、美股三大核心指数的实时罗盘雷达。
