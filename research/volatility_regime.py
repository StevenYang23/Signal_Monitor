from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None


class HMMVolatilityRegime:
    """Two-state HMM volatility regime classifier."""

    def __init__(
        self,
        underly_ticker: str = "^GSPC",
        vol_ticker: str = "^VIX",
        train_window: int = 756,
        refit_step: int = 22,
        data_period: str = "20y",
        trading_days: int = 252,
    ):
        self.underly_ticker = underly_ticker
        self.vol_ticker = vol_ticker
        self.train_window = train_window
        self.refit_step = refit_step
        self.data_period = data_period
        self.trading_days = trading_days
        self.feature_columns = [
            "VIX",
            "RV_22",
            "RV_63",
            "lagged_VRP",
            "trend_63",
            "drawdown_252",
            "sign_flip_22",
        ]

    def download_data(self) -> pd.DataFrame:
        prices = pd.concat(
            {
                "SPX": self._download_close(self.underly_ticker),
                "VIX": self._download_close(self.vol_ticker),
            },
            axis=1,
        ).sort_index()
        self.prices = prices
        return prices

    def build_features(self, prices: pd.DataFrame | None = None) -> pd.DataFrame:
        if prices is None:
            prices = self.prices

        df = prices.copy()
        df["SPX_ret"] = np.log(df["SPX"] / df["SPX"].shift(1))
        df["RV_22"] = df["SPX_ret"].rolling(22).std() * np.sqrt(self.trading_days) * 100
        df["RV_63"] = df["SPX_ret"].rolling(63).std() * np.sqrt(self.trading_days) * 100
        df["lagged_VRP"] = df["VIX"].shift(22) - df["RV_22"]
        df["trend_63"] = df["SPX"].pct_change(63)
        df["drawdown_252"] = df["SPX"] / df["SPX"].rolling(252, min_periods=63).max() - 1
        df["sign_flip_22"] = (
            np.sign(df["SPX_ret"]).ne(np.sign(df["SPX_ret"]).shift(1)).rolling(22).mean()
        )

        self.features = df.dropna(subset=self.feature_columns).copy()
        return self.features

    def train(self, prices: pd.DataFrame | None = None) -> pd.Series:
        if GaussianHMM is None:
            raise ImportError("Install hmmlearn to run the HMM: pip install hmmlearn")

        features = self.build_features(prices)
        hmm_data = features[self.feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        regimes = pd.Series(index=hmm_data.index, dtype="object", name="hmm")

        for train_end in range(self.train_window, len(hmm_data), self.refit_step):
            train_start = train_end - self.train_window
            test_end = min(train_end + self.refit_step, len(hmm_data))
            train = hmm_data.iloc[train_start:train_end]
            test = hmm_data.iloc[train_end:test_end]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(train)
            x_test = scaler.transform(test)
            model = self._best_hmm(x_train)
            if model is None:
                continue

            train_states = pd.Series(model.predict(x_train), index=train.index)
            test_states = pd.Series(model.predict(x_test), index=test.index)
            train_labels = self._label_high_low(features.loc[train.index], train_states)
            high_state = train_states[train_labels.eq("high_vol")].mode().iloc[0]
            regimes.loc[test.index] = np.where(test_states.eq(high_state), "high_vol", "low_vol")

        self.regimes = regimes.dropna()
        return self.regimes

    def update_latest_prices(self, prices: pd.DataFrame | None = None) -> pd.DataFrame:
        if prices is None:
            prices = self.prices

        updated = prices.copy()
        for column, ticker in {"SPX": self.underly_ticker, "VIX": self.vol_ticker}.items():
            latest = self._latest_price(ticker)
            if latest is None:
                continue

            latest_date, latest_price = latest
            updated.loc[latest_date, column] = latest_price

        self.prices = updated.sort_index()
        return self.prices

    def plot(
        self,
        features: pd.DataFrame | None = None,
        regimes: pd.Series | None = None,
        display_period: int | None = None,
        update_latest: bool = True,
    ) -> None:
        if update_latest:
            features = self.build_features(self.update_latest_prices())
        if features is None:
            features = self.features
        if regimes is None:
            regimes = self.regimes

        plot_regimes = regimes.copy()
        missing_dates = features.index.difference(plot_regimes.index)
        if len(missing_dates) and not plot_regimes.empty:
            latest_regime = pd.Series(plot_regimes.iloc[-1], index=missing_dates, name=plot_regimes.name)
            plot_regimes = pd.concat([plot_regimes, latest_regime]).sort_index()

        plot_df = features[["SPX", "RV_22", "VIX"]].join(plot_regimes.rename("regime"), how="inner").dropna()
        if display_period is not None:
            plot_df = plot_df.tail(display_period)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

        axes[0].plot(plot_df.index, plot_df["SPX"], color="black", linewidth=1.2, label="SPX")
        high_vol = plot_df["regime"].eq("high_vol")
        blocks = high_vol.ne(high_vol.shift()).cumsum()
        for _, block in plot_df[high_vol].groupby(blocks[high_vol]):
            axes[0].axvspan(block.index[0], block.index[-1], color="red", alpha=0.16, linewidth=0)
            axes[1].axvspan(block.index[0], block.index[-1], color="red", alpha=0.16, linewidth=0)

        axes[0].set_title("SPX with HMM high-vol regime shading")
        axes[0].set_ylabel("SPX")
        axes[0].legend(loc="upper left")

        axes[1].plot(plot_df.index, plot_df["RV_22"], color="tab:blue", linewidth=1.0, label="22d realized vol")
        axes[1].plot(plot_df.index, plot_df["VIX"], color="tab:orange", linewidth=1.0, alpha=0.75, label="VIX")
        axes[1].set_ylabel("Annualized vol %")
        axes[1].legend(loc="upper left")
        plt.show()

    def _download_close(self, ticker: str) -> pd.Series:
        raw = yf.download(ticker, period=self.data_period, auto_adjust=True, progress=False)
        if raw.empty or "Close" not in raw:
            raise ValueError(f"No close data returned for {ticker}")

        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.astype(float)

    def _latest_price(self, ticker: str) -> tuple[pd.Timestamp, float] | None:
        raw = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        if raw.empty or "Close" not in raw:
            return None

        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return None

        latest_time = pd.Timestamp(close.index[-1])
        if latest_time.tzinfo is not None:
            latest_time = latest_time.tz_convert(None)

        return latest_time.normalize(), float(close.iloc[-1])

    def _best_hmm(self, x_train: np.ndarray) -> GaussianHMM | None:
        best_model = None
        best_score = -np.inf
        for seed in [7, 21, 42, 101]:
            model = GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=500,
                min_covar=1e-3,
                random_state=seed,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_train)
                score = model.score(x_train)
            except Exception:
                continue

            if np.isfinite(score) and score > best_score:
                best_model = model
                best_score = score
        return best_model

    @staticmethod
    def _label_high_low(data: pd.DataFrame, raw_labels: pd.Series) -> pd.Series:
        score = (
            data["RV_22"].groupby(raw_labels).mean().rank()
            + data["sign_flip_22"].groupby(raw_labels).mean().rank()
            - data["trend_63"].groupby(raw_labels).mean().rank()
        )
        high_label = score.idxmax()
        return pd.Series(np.where(raw_labels.eq(high_label), "high_vol", "low_vol"), index=raw_labels.index)
