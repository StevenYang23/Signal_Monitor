## Live Dashboard Demo / 实时看板预览

![Index Quant Signal Hub — SPX Dashboard](demo/signal_page.png)

*Index Quant Signal Hub — integrated 3D volatility surface, dealer GEX by TTM, HMM regime signals, VRP/term-structure metrics, and 1-sigma move targets for **SPX, NDX, and DJI**.*

*Index Quant Signal Hub — 集成 3D 波动率曲面、分到期 GEX、HMM 机制信号、VRP/期限结构指标及 1-sigma 波动目标，覆盖 **SPX、NDX、DJI** 三大指数。*

---

## Index Coverage & Data Mapping / 指数覆盖与数据源映射

The live dashboard tab labels match the indices below. Option surfaces are pulled from Futu OpenD; spot references use Yahoo Finance; HMM regime models use the internal market keys in [volatility_regime.py](volatility_regime.py).

看板标签与底层数据映射如下（期权链来自富途 OpenD，现货参考来自 Yahoo Finance）：

| Tab | Options (Futu) | Spot (Yahoo) | HMM market key | Notes |
|-----|------------------|--------------|----------------|-------|
| **SPX** | `US..SPX` | `^SPX` | `SPX` | S&P 500 index options |
| **NDX** | `US..NDX` | `^NDX` | `NSDQ` | Nasdaq-100 index options |
| **DJI** | `US.DIA` | `^DJI` | `DJI` | Dow Jones — IV surface uses **DIA ETF options** as proxy |

---

# Signal Monitor — Equity Volatility Regime & Option Sentiment Dashboard
# Signal Monitor — 美股波动率机制与期权情绪监测看板

An enterprise-grade research and live production dashboard designed to monitor and trade equity markets using Option Volatility Surface structure, Dupire Local Volatility, dealer Gamma Exposure (GEX), and a 2-state Gaussian Hidden Markov Model (HMM) volatility regime model.

这是一个企业级的量化研究与实盘显示看板。它利用期权隐含波动率曲面（Option Volatility Surface）、Dupire局部波动率模型、做市商 Gamma 暴露（GEX），以及基于两状态高斯隐马尔可夫模型（HMM）历史滚动训练的波动率机制分类模型，对美股主流指数（SPX、DJI、NDX）进行市场状态追踪与交易决策。

---

## Project Architecture / 项目内核结构

The system consists of interactive research walk-throughs in [research/](research/) and scalable production-oriented core library modules at the repository root:

系统主要由 [research/](research/) 目录下的深入探究 Notebook 及根目录下的模块化工程文件组成：

- **Volatility Surface & Grid Construction / 波动率曲面与网格构建**：
  [vol_surface.py](vol_surface.py) ingests option chain data, computes implied volatilities across a dynamic Moneyness ($K/S$) and Tenor ($DTE$) grid, and implements a Dupire Local Volatility solver to resolve risk-neutral local volatilities. It is studied in [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb).
  [vol_surface.py](vol_surface.py) 负责期权链数据的提取、隐含波动率表征，并在行权价偏离度（Moneyness, $K/S$）及到期期限（$DTE$）的双维网格上完成插值，同时配有 Dupire 局部波动率求解器。该部分的具体研究见 [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb)。

- **Dealer Gamma Exposure (GEX) / 做市商 Gamma 暴露**：
  [gex.py](gex.py) converts the live chain into dollar GEX per 1% spot move, sliced by TTM (0DTE / ≤7d / ≤14d / ≤30d / ≤45d / all). Call GEX is signed positive and put GEX negative under the standard dealer-short-to-public convention. The dashboard plots the per-strike profile plus net GEX, gamma flip, call wall, and put wall.
  [gex.py](gex.py) 将期权链转为每 1% 现货变动的美元 GEX，并按到期窗口切片。看板上展示逐行权价分布、净 GEX、gamma flip、call wall 与 put wall。

- **Markov Volatility Regime Filter / 隐马尔可夫波动率机制过滤器**：
  [volatility_regime.py](volatility_regime.py) implements rolling 2-state Gaussian Hidden Markov Models (HMM) to separate equity indexes into two practical regimes: "Low-vol / slow-bull" vs "High-vol / choppy". It is analyzed in [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb).
  [volatility_regime.py](volatility_regime.py) 采用滑动窗口（通常为 504 个交易日，即 2 年）来实时拟合两状态高斯隐马尔可夫模型，将指数划分为“低波缓牛机制”和“高波震荡机制”。具体的全样本回测见 [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb)。

- **Live Production App / 实时看板应用**：
  [app.py](app.py) consolidates core calculations into a lightweight HTTP dashboard at `http://127.0.0.1:8050`. On startup it **preloads SPX, NDX, and DJI in parallel** (background workers + frontend polling). Each tab shows:
  - **3D vol surface** (Raw IV / Smooth IV / Arb-free Local Vol) on a transparent dark-theme canvas
  - **Dealer GEX** (TTM radios, per-strike call/put bars, flip / walls)
  - **HMM Signal Window** (candlesticks + close line, red high-vol regime shading, today marker)
  - **Sidebar:** HMM trade signal, VRP, term-structure roll spread, 1-sigma move table, DeepSeek-enhanced structure metrics, RV vs IV chart

  [app.py](app.py) 汇总所有量化计算模块，暴露轻量 Web 看板（默认端口 `8050`）。**SPX / NDX / DJI 三个指数并行后台加载**，页面在未就绪前留白。各标签页包含：3D 波动率曲面（Raw / Smooth / Local Vol）、分到期 GEX、HMM 信号窗（K 线 + 高波红色 shading）、侧边栏机制信号、VRP、期限结构、1-sigma 目标位、DeepSeek 结构指标解读及 RV vs IV 图。

  End-to-end workflow notebook: [trading_signal.ipynb](trading_signal.ipynb)
  完整研究/信号流程见 [trading_signal.ipynb](trading_signal.ipynb)

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

### 2. Dealer Gamma Exposure (GEX) / 做市商 Gamma 暴露

[gex.py](gex.py) estimates how much underlying dealers must trade to stay delta-neutral after a 1% spot move:

$$
\mathrm{GEX} = \Gamma \times \mathrm{OI} \times M \times S^{2} \times 0.01 \times \mathrm{sign}
$$

Calls are signed positive and puts negative. Positive net GEX: dealers long gamma (buy dips / sell rips). Negative: dealers short gamma (amplify the move). The live hub slices by TTM (0DTE, ≤7d, ≤14d, ≤30d, ≤45d, all) and plots the per-strike profile with gamma flip, call wall, and put wall.

[gex.py](gex.py) 按到期窗口计算美元 GEX。正净 GEX 对应做市商多 gamma（抑制波动），负净 GEX 对应空 gamma（放大波动）。看板提供 gamma flip、call wall、put wall。

---

### 3. HMM-based Volatility Regime Filtering / 隐马尔可夫机制概率分类

Standard Trend Following strategies suffer severe whipsaw losses during sudden crisis periods. In [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb), we build a 2-State Gaussian Hidden Markov Model (HMM) fitted exclusively on rolling 2-year windows (504 trading days) using a single feature: **lagged VRP** (Volatility Risk Premium).

$$
\mathrm{RV22}_{t} = \mathrm{Std}(\log r,\, 22) \times \sqrt{252} \times 100
$$

$$
\mathrm{laggedVRP}_{t} = IV_{t-22} - \mathrm{RV22}_{t}
$$

**Walk-Forward Out-Of-Sample Scoring:**
1. Fit the Gaussian HMM on 504 historical days ending on day *t*−1 (excluding today, *t*, to avoid leakages).
2. Determine state roles: the state with the lower mean of VRP is assigned as the **Calm (Low-Vol)** regime.
3. Use `predict_proba` to evaluate the OOS probabilities of today being in a low-vol regime, and tomorrow transitioning to a low-vol regime.
4. **Trading Rule:** Long the index tomorrow if and only if **P(low vol today) ≥ 0.5** and **P(low vol tomorrow) ≥ 0.5**. Otherwise, go Flat.

在震荡走熊或危机降临期间，传统的持股或趋势跟踪策略往往面临严重的净值回撤。在 [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb) 中，本研究拟合了一个基于两状态高斯隐马尔可夫模型（HMM）的日频过滤策略。该模型仅使用一个衍生特征：**滞后波动率风险溢价（lagged VRP）**：

$$
\mathrm{RV22}_{t} = \mathrm{Std}(\text{对数日收益率},\, 22) \times \sqrt{252} \times 100
$$

$$
\mathrm{laggedVRP}_{t} = IV_{t-22} - \mathrm{RV22}_{t}
$$

**滚动跨期样本外概率打分：**
1. 选取截至 *t*−1 日以前的滚动 504 天（约 2 年）数据训练 HMM 模型（严格剥离当天 *t* 的数据，绝无未来偏误）。
2. 根据特征均值对隐含状态进行自动纠偏：均值较低的一方定义为 **低波平静状态**。
3. 调取 `predict_proba` 求解今日处于该平静状态的样本外后验概率 **P(low vol today)** 以及一阶转移矩阵下明日仍处于该状态的预测概率 **P(low vol tomorrow)**。
4. **交易规则：** 当且仅当两项低波概率判定皆 ≥ 0.5 时，明日做多标的（SPX, DJI, NDX），否则持币平仓。

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
pip install numpy pandas matplotlib yfinance hmmlearn scikit-learn scipy plotly futu-api
```

If you have a Futu OpenAPI subscription and intend to pull real-time cash options chains, ensure your FutuOpenD setup is active on port `11111` or configure your settings in [vol_surface.py](vol_surface.py). Otherwise, the code will seamlessly auto-generate highly realistic simulated synthetics and cache metrics locally.

如果您配置了富途 OpenAPI 并在本地开通了 FutuOpenD (默认端口 `11111`)，本系统可直接获取美股实时多期限期权链。如未开启，系统将自动平滑切换至高度逼真的模拟合成波动面，并实现本地无缝缓存。

### 1. Run Interactive Notebook Studies / 交互式回测与模型调试
Open the notebooks inside [research/](research/) or the root workflow notebook to explore step-by-step:
在 VS Code 内加载以下 Notebook 进行交互式研究：

- **End-to-end signal workflow (surfaces + HMM):** [trading_signal.ipynb](trading_signal.ipynb)
- **Volatility Surface Study:** [research/vol_surface_study.ipynb](research/vol_surface_study.ipynb)
- **HMM Walk-Forward Regime Transition Backtest:** [research/vol_regime_study.ipynb](research/vol_regime_study.ipynb)

### 2. Run Live Production Dashboard / 启动本地交互式看板

**Prerequisites / 前置条件**
- Futu **OpenD** running locally (`127.0.0.1:11111`) for live option chains
- Python env with dependencies below
- Optional: DeepSeek API key for AI structure-metric insights (`USE_DEEPSEEK=1`, key in `.env` as `deepseek=...` or env var `DEEPSEEK_API_KEY`)

Run the server from the project root:

在项目根目录启动服务：

```bash
python app.py
```

Windows (conda example):

```powershell
& C:\Users\Lenovo\miniconda3\envs\env2\python.exe app.py
```

Then open **`http://127.0.0.1:8050`** in your browser. Use **Ctrl+Shift+R** after code updates. Keep the terminal running (do not Ctrl+C) while using the dashboard.

浏览器访问 **`http://127.0.0.1:8050`**。代码更新后请 **Ctrl+Shift+R** 强制刷新。使用看板期间请保持终端进程运行。

**Environment variables / 环境变量**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8050` | HTTP server port |
| `USE_DEEPSEEK` | `1` | Set `0` to disable DeepSeek insight rewriting |
| `DEEPSEEK_API_KEY` | — | API key (or `deepseek=` in `.env`) |

If OpenD is offline, the pipeline falls back to cached/demo surfaces automatically.

若 Futu OpenD 未启动，系统自动使用本地缓存或 demo 合成曲面，不会无限阻塞。
