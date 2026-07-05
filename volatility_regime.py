from __future__ import annotations

import logging
import warnings

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yfinance as yf

from hmmlearn.hmm import GaussianHMM


class HMMVolatilityRegime:
    """Two-state HMM volatility regime classifier."""

    DEFAULT_FEATURE_COLUMNS = ["lagged_VRP"]

    ALL_FEATURE_COLUMNS = [
        "VIX",
        "RV_22",
        "RV_63",
        "RV_63_22",
        "lagged_VRP",
        "trend_63",
        "drawdown_22",
        "sign_flip_22",
        "sharpe",
    ]

    FEATURE_SETS = {
        "default": DEFAULT_FEATURE_COLUMNS,
        "drawdown_chop": ["drawdown_22", "sign_flip_22"],
        "vol_slope_chop": ["VIX", "RV_63_22", "sign_flip_22"],
        "all_vol": ["VIX", "RV_22", "RV_63", "RV_63_22", "lagged_VRP"],
        "recommended": [
            "VIX", "RV_63_22", "trend_63", "drawdown_22",
            "lagged_VRP", "sign_flip_22",
        ],
        "full": ALL_FEATURE_COLUMNS,
    }

    MARKETS = {
        "SPX": {"underly": "^SPX", "vol": "^VIX"},
        "DJI": {"underly": "DIA", "vol": "^VXD"},
        "NSDQ": {"underly": "QQQ", "vol": "^VXN"},
    }

    CRISIS_WINDOWS = [
        ("2008-09-01", "2009-03-31"),
        ("2020-02-15", "2020-04-30"),
        ("2022-01-01", "2022-10-31"),
    ]

    # ~2.5y fetch: 2y HMM fit (504d) + lagged_VRP warm-up (22d shift + 22d RV)
    SIGNAL_FETCH_TRADING_DAYS = int(252 * 2.5)
    FEATURE_WARMUP_DAYS = 44

    def __init__(
        self,
        underly_ticker: str = "^SPX",
        vol_ticker: str = "^VIX",
        train_window: int = 504,
        refit_step: int = 1,
        data_period: str = "max",
        live_start: str = "2000-01-01",
        live_end: str | None = None,
        trading_days: int = 252,
        threshold_today: float = 0.5,
        threshold_tomorrow: float = 0.5,
        hmm_seed: int = 42,
        hmm_n_iter: int = 500,
        hmm_tol: float = 1e-1,
        feature_columns: list[str] | str | None = None,
    ):
        self.underly_ticker = underly_ticker
        self.vol_ticker = vol_ticker
        self.train_window = train_window
        self.refit_step = refit_step
        self.data_period = data_period
        self.live_start = live_start
        self.live_end = live_end
        self.trading_days = trading_days
        self.threshold_today = threshold_today
        self.threshold_tomorrow = threshold_tomorrow
        self.hmm_seed = hmm_seed
        self.hmm_n_iter = hmm_n_iter
        self.hmm_tol = hmm_tol
        self.feature_columns = self._resolve_feature_columns(feature_columns)
        self.prices: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.regimes: pd.DataFrame | None = None
        self.today_signal: dict | None = None
        self.today_date: pd.Timestamp | None = None

    @classmethod
    def from_market(cls, name: str, *, signal_mode: bool = False, **kwargs) -> HMMVolatilityRegime:
        if name not in cls.MARKETS:
            raise ValueError(f"Unknown market '{name}'. Expected one of {sorted(cls.MARKETS)}.")

        if signal_mode and "data_period" not in kwargs:
            kwargs["data_period"] = f"{cls.SIGNAL_FETCH_TRADING_DAYS}d"

        config = cls.MARKETS[name]
        return cls(
            underly_ticker=config["underly"],
            vol_ticker=config["vol"],
            **kwargs,
        )

    def download_data(self) -> pd.DataFrame:
        underly = self._download_close(self.underly_ticker).rename("SPX")
        try:
            vol = self._download_close(self.vol_ticker).rename("VIX")
        except ValueError:
            vol = pd.Series(dtype=float, name="VIX")
        prices = pd.concat([underly, vol], axis=1).sort_index()
        prices["VIX"] = prices["VIX"].ffill()
        # ^VXD / ^VXN can be sparse on yfinance; use 22d RV as vol proxy when needed.
        if prices["VIX"].notna().sum() < 50:
            ret = np.log(prices["SPX"] / prices["SPX"].shift(1))
            prices["VIX"] = ret.rolling(22).std() * np.sqrt(self.trading_days) * 100
        self.prices = prices.dropna(subset=["SPX", "VIX"])
        return self.prices

    @classmethod
    def download_market(cls, name: str, data_period: str = "max") -> pd.DataFrame:
        if name not in cls.MARKETS:
            raise ValueError(f"Unknown market '{name}'. Expected one of {sorted(cls.MARKETS)}.")

        config = cls.MARKETS[name]
        model = cls(
            underly_ticker=config["underly"],
            vol_ticker=config["vol"],
            data_period=data_period,
        )
        prices = model.download_data()
        return prices.rename(columns={"SPX": "UNDERLY", "VIX": "VOL"})

    def build_features(self, prices: pd.DataFrame | None = None) -> pd.DataFrame:
        if prices is None:
            prices = self.prices

        df = prices.copy()
        df["SPX_ret"] = np.log(df["SPX"] / df["SPX"].shift(1))
        df["RV"] = df["SPX_ret"].rolling(22).std() * np.sqrt(self.trading_days)
        df["RV_22"] = df["RV"] * 100
        df["IV"] = df["VIX"]
        df["lagged_VRP"] = df["IV"].shift(22) - df["RV_22"]
        df["RV_63"] = df["SPX_ret"].rolling(63).std() * np.sqrt(self.trading_days) * 100
        df["RV_63_22"] = df["RV_63"] - df["RV_22"]
        df["trend_63"] = df["SPX"].pct_change(63)
        df["drawdown_22"] = df["SPX"] / df["SPX"].rolling(22, min_periods=10).max() - 1
        df["sign_flip_22"] = (
            np.sign(df["SPX_ret"]).ne(np.sign(df["SPX_ret"]).shift(1)).rolling(22).mean()
        )
        df["sharpe"] = df["SPX_ret"] / df["RV_63"]
        df["future_ret"] = np.log(df["SPX"].shift(-1) / df["SPX"])

        self.features = df.dropna(subset=self.feature_columns).copy()
        return self.features

    def train(
        self,
        prices: pd.DataFrame | None = None,
        feature_columns: list[str] | str | None = None,
    ) -> pd.DataFrame:
        if GaussianHMM is None:
            raise ImportError("Install hmmlearn to run the HMM: pip install hmmlearn")

        features = self.build_features(prices)
        selected_features = self._resolve_feature_columns(feature_columns)
        regimes = self._train_from_features(features, selected_features)
        self.feature_columns = selected_features
        self.regimes = regimes
        return self.regimes

    def refresh_prices(self) -> pd.DataFrame:
        """Re-download the configured window and overlay latest intraday quotes."""
        self.download_data()
        return self.update_latest_prices()

    def fit_once(
        self,
        prices: pd.DataFrame | None = None,
        feature_columns: list[str] | str | None = None,
        exclude_today: bool = True,
    ) -> pd.DataFrame:
        """Fit HMM on the prior train_window days (today excluded) and score today OOS.

        Equivalent to the last iteration of the daily walk-forward loop in
        vol_regime_study.ipynb: same features, HMM params, train/score windows,
        state labeling, and signal rule.
        """
        if GaussianHMM is None:
            raise ImportError("Install hmmlearn to run the HMM: pip install hmmlearn")

        logging.getLogger("hmmlearn").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        features = self.build_features(prices)
        selected_features = self._resolve_feature_columns(feature_columns)
        self.feature_columns = selected_features

        missing = [column for column in selected_features if column not in features]
        if missing:
            raise ValueError(f"Missing HMM feature columns: {missing}")

        hmm_data = features[selected_features].replace([np.inf, -np.inf], np.nan).dropna()
        min_rows = self.train_window + 1
        if len(hmm_data) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows of feature data to fit the HMM "
                f"(got {len(hmm_data)}). Fetch more history — e.g. "
                f"data_period='{self.SIGNAL_FETCH_TRADING_DAYS}d'."
            )

        if exclude_today:
            loc_idx = len(hmm_data) - 1
            train_data = hmm_data.iloc[loc_idx - self.train_window : loc_idx]
            score_data = hmm_data.iloc[loc_idx - self.train_window + 1 : loc_idx + 1]
            today_date = hmm_data.index[-1]
        else:
            train_data = hmm_data.iloc[-self.train_window :]
            score_data = train_data
            today_date = None

        if len(train_data) < 2:
            raise ValueError("Need at least two training rows after excluding today.")

        x_train = train_data.to_numpy()
        model = self._fit_hmm(x_train)
        if model is None:
            raise RuntimeError("HMM fit failed on training data.")

        low_state, high_state = self._state_indices_from_model(model, train_data=train_data)
        train_probs = model.predict_proba(x_train)
        regimes = self._build_regime_frame(
            train_data.index,
            train_probs,
            high_state,
            low_state,
            model,
            in_sample=True,
        )

        if exclude_today:
            today_probs = model.predict_proba(score_data.to_numpy())[-1:]
            today_regime = self._build_regime_frame(
                score_data.index[-1:],
                today_probs,
                high_state,
                low_state,
                model,
                in_sample=False,
            )
            regimes = pd.concat([regimes, today_regime])
            self.today_date = today_date
            self.today_signal = {
                "date": today_date,
                "prob_low_vol": float(today_regime["prob_low_vol"].iloc[0]),
                "prob_low_vol_tmr": float(today_regime["prob_low_vol_tmr"].iloc[0]),
                "hmm": today_regime["hmm"].iloc[0],
                "trade_signal": bool(today_regime["trade_signal"].iloc[0]),
            }
        else:
            self.today_date = None
            self.today_signal = None

        self.regimes = regimes
        self.model = model
        return self.regimes

    def print_today_signal(self) -> None:
        if self.today_signal is None:
            raise ValueError("No today signal available. Call fit_once(exclude_today=True) first.")

        signal = self.today_signal
        market = self.underly_ticker.removeprefix("^")
        print(f"{market} signal as of {signal['date'].date()}:")
        print(f"  prob(low vol today): {signal['prob_low_vol']:.1%}")
        print(f"  prob(low vol tmr):   {signal['prob_low_vol_tmr']:.1%}")
        print(f"  regime today:        {signal['hmm']}")
        print(f"  trade signal:        {signal['trade_signal']}")

    def display(self, display_period: int = 22) -> None:
        """Update today's prices, fit once on 2y (today OOS), and plot a short recent window."""
        if display_period >= self.train_window:
            raise ValueError(
                f"display_period ({display_period}) must be less than "
                f"train_window ({self.train_window})."
            )

        if self.prices is None:
            self.download_data()
        self.update_latest_prices()
        self.fit_once()
        self._plot_signal_window(display_period)

    def add_strategy_pnl(self, features: pd.DataFrame | None = None) -> pd.DataFrame:
        if features is None:
            features = self.features
        if self.regimes is None:
            raise ValueError("Call train() before add_strategy_pnl().")

        pnl_df = features.join(self.regimes, how="inner")
        pnl_df = pnl_df.dropna(subset=["hmm", "SPX_ret", "trade_signal"]).copy()
        pnl_df["regime_signal"] = pnl_df["trade_signal"].astype(int)
        pnl_df["future_ret"] = pnl_df["SPX_ret"].shift(-1)
        pnl_df = pnl_df.dropna(subset=["future_ret"]).copy()
        pnl_df["strategy_ret"] = pnl_df["regime_signal"] * pnl_df["future_ret"]
        pnl_df["strategy_pnl"] = pnl_df["strategy_ret"].cumsum()
        pnl_df["strategy_cum_return_rate"] = np.expm1(pnl_df["strategy_pnl"])
        pnl_df["buy_hold_cum_return_rate"] = np.expm1(pnl_df["future_ret"].cumsum())
        self.study_df = pnl_df
        return pnl_df

    def return_summary(self, pnl_df: pd.DataFrame | None = None) -> pd.Series:
        if pnl_df is None:
            if getattr(self, "study_df", None) is None:
                pnl_df = self.add_strategy_pnl()
            else:
                pnl_df = self.study_df

        years_elapsed = (pnl_df.index[-1] - pnl_df.index[0]).days / 365.25
        total_strategy = pnl_df["strategy_cum_return_rate"].iloc[-1]
        total_buy_hold = pnl_df["buy_hold_cum_return_rate"].iloc[-1]
        cagr_strategy = self._cagr_from_total_return(total_strategy, years_elapsed)
        cagr_buy_hold = self._cagr_from_total_return(total_buy_hold, years_elapsed)
        return pd.Series(
            {
                "total_strategy": total_strategy,
                "total_buy_hold": total_buy_hold,
                "cagr_strategy": cagr_strategy,
                "cagr_buy_hold": cagr_buy_hold,
                "excess_total": total_strategy - total_buy_hold,
                "excess_cagr": cagr_strategy - cagr_buy_hold,
                "in_market_pct": pnl_df["regime_signal"].mean(),
            }
        )

    def evaluate_feature_sets(
        self,
        prices: pd.DataFrame | None = None,
        feature_sets: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        if GaussianHMM is None:
            raise ImportError("Install hmmlearn to run the HMM: pip install hmmlearn")

        features = self.build_features(prices)
        if feature_sets is None:
            feature_sets = self.FEATURE_SETS

        rows = []
        regimes_by_set = {}
        for name, columns in feature_sets.items():
            selected_features = self._resolve_feature_columns(columns)
            regimes = self._train_from_features(features, selected_features)
            if regimes.empty:
                continue

            regimes_by_set[name] = regimes
            metrics = self._regime_metrics(features, regimes)
            metrics["feature_set"] = name
            metrics["n_features"] = len(selected_features)
            metrics["features"] = ", ".join(selected_features)
            rows.append(metrics)

        if not rows:
            self.feature_set_results = pd.DataFrame()
            self.regimes_by_feature_set = regimes_by_set
            return self.feature_set_results

        results = pd.DataFrame(rows).set_index("feature_set")
        results["rank_score"] = (
            results["forward_rv_spread_22"].rank(ascending=False, pct=True)
            + results["drawdown_avoidance"].rank(ascending=False, pct=True)
            + results["avg_duration"].rank(ascending=False, pct=True)
            + results["flip_rate"].rank(ascending=True, pct=True)
        )
        results = results.sort_values(
            ["rank_score", "forward_rv_spread_22", "n_features"],
            ascending=[True, False, True],
        )
        self.feature_set_results = results
        self.regimes_by_feature_set = regimes_by_set
        return results

    def _train_from_features(
        self,
        features: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        missing = [column for column in feature_columns if column not in features]
        if missing:
            raise ValueError(f"Missing HMM feature columns: {missing}")

        logging.getLogger("hmmlearn").setLevel(logging.ERROR)
        hmm_data = features[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        live_end = pd.Timestamp(self.live_end) if self.live_end else hmm_data.index.max()
        df_model = hmm_data.loc[:live_end].copy()
        predict_indices = df_model.loc[self.live_start :].index

        regimes = pd.DataFrame(index=predict_indices)
        regimes["hmm"] = pd.Series(index=predict_indices, dtype="object")
        regimes["trade_signal"] = pd.Series(index=predict_indices, dtype="boolean")
        regimes["prob_high_vol"] = np.nan
        regimes["prob_low_vol"] = np.nan
        regimes["prob_high_vol_tmr"] = np.nan
        regimes["prob_low_vol_tmr"] = np.nan

        current_model = None
        low_state = high_state = None

        for i, current_date in enumerate(predict_indices):
            loc_idx = df_model.index.get_loc(current_date)
            if isinstance(loc_idx, slice):
                loc_idx = loc_idx.start

            if loc_idx < self.train_window:
                continue

            if i % self.refit_step == 0 or current_model is None:
                train = df_model.iloc[loc_idx - self.train_window : loc_idx]
                model = self._fit_hmm(train.to_numpy())
                if model is None:
                    continue
                current_model = model
                low_state, high_state = self._state_indices_from_model(model, train_data=train)

            score = df_model.iloc[loc_idx - self.train_window + 1 : loc_idx + 1]
            try:
                today_probs = current_model.predict_proba(score.to_numpy())[-1]
            except Exception:
                continue

            day_regime = self._build_regime_frame(
                pd.Index([current_date]),
                today_probs.reshape(1, -1),
                high_state,
                low_state,
                current_model,
            )
            regimes.loc[day_regime.index, day_regime.columns] = day_regime

        return regimes.dropna(subset=["hmm"])

    def _resolve_feature_columns(self, feature_columns: list[str] | str | None) -> list[str]:
        if feature_columns is None:
            return list(self.DEFAULT_FEATURE_COLUMNS)
        if isinstance(feature_columns, str):
            if feature_columns not in self.FEATURE_SETS:
                raise ValueError(
                    f"Unknown feature set '{feature_columns}'. "
                    f"Expected one of {sorted(self.FEATURE_SETS)}."
                )
            return list(self.FEATURE_SETS[feature_columns])
        return list(feature_columns)

    def _regime_metrics(
        self,
        features: pd.DataFrame,
        regimes: pd.DataFrame,
    ) -> dict[str, float | int | str]:
        aligned = features[["SPX_ret"]].join(regimes[["hmm", "trade_signal"]], how="inner").dropna()
        if aligned.empty:
            return {}

        regime_changes = aligned["hmm"].ne(aligned["hmm"].shift())
        flips = max(int(regime_changes.sum()) - 1, 0)
        durations = aligned["hmm"].groupby(regime_changes.cumsum()).size()
        high_vol = aligned["hmm"].eq("high_vol")

        forward_rv_22 = (
            features["SPX_ret"].shift(-1).rolling(22).std().shift(-21) * np.sqrt(self.trading_days) * 100
        )
        forward_eval = aligned[["hmm"]].join(forward_rv_22.rename("forward_RV_22")).dropna()
        forward_rv_by_regime = forward_eval.groupby("hmm")["forward_RV_22"].mean()
        high_forward_rv = float(forward_rv_by_regime.get("high_vol", np.nan))
        low_forward_rv = float(forward_rv_by_regime.get("low_vol", np.nan))
        # Detect and correct flipped state labels
        if (np.isfinite(high_forward_rv) and np.isfinite(low_forward_rv)
                and high_forward_rv < low_forward_rv):
            aligned["hmm"] = aligned["hmm"].map({"high_vol": "low_vol", "low_vol": "high_vol"})
            aligned["trade_signal"] = ~aligned["trade_signal"]
            high_vol = aligned["hmm"].eq("high_vol")
            regime_changes = aligned["hmm"].ne(aligned["hmm"].shift())
            flips = max(int(regime_changes.sum()) - 1, 0)
            durations = aligned["hmm"].groupby(regime_changes.cumsum()).size()
            forward_eval = aligned[["hmm"]].join(forward_rv_22.rename("forward_RV_22")).dropna()
            forward_rv_by_regime = forward_eval.groupby("hmm")["forward_RV_22"].mean()
            high_forward_rv = float(forward_rv_by_regime.get("high_vol", np.nan))
            low_forward_rv = float(forward_rv_by_regime.get("low_vol", np.nan))

        signal = aligned["trade_signal"].shift(1).fillna(False).astype(int)
        strategy_ret = aligned["SPX_ret"] * signal
        buy_hold_max_dd = self._max_drawdown(aligned["SPX_ret"])
        strategy_max_dd = self._max_drawdown(strategy_ret)

        crisis_mask = pd.Series(False, index=aligned.index)
        for start, end in self.CRISIS_WINDOWS:
            crisis_mask |= aligned.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
        crisis_high_vol_pct = float(high_vol[crisis_mask].mean()) if crisis_mask.any() else np.nan

        return {
            "observations": int(len(aligned)),
            "start": aligned.index.min().date().isoformat(),
            "end": aligned.index.max().date().isoformat(),
            "avg_duration": float(durations.mean()),
            "flips": flips,
            "flip_rate": float(flips / max(len(aligned) - 1, 1)),
            "high_vol_pct": float(high_vol.mean()),
            "forward_rv_high_22": high_forward_rv,
            "forward_rv_low_22": low_forward_rv,
            "forward_rv_spread_22": high_forward_rv - low_forward_rv,
            "buy_hold_max_dd": buy_hold_max_dd,
            "strategy_max_dd": strategy_max_dd,
            "drawdown_avoidance": strategy_max_dd - buy_hold_max_dd,
            "crisis_high_vol_pct": crisis_high_vol_pct,
        }

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        equity = np.exp(returns.fillna(0).cumsum())
        drawdown = equity / equity.cummax() - 1
        return float(drawdown.min())

    @staticmethod
    def _cagr_from_total_return(total_return: float, years: float) -> float:
        if years <= 0:
            return np.nan
        return (1.0 + total_return) ** (1.0 / years) - 1.0

    @staticmethod
    def implied_movement_pct(
        iv_pct: float,
        horizon_days: int,
        trading_days: int = 252,
    ) -> float:
        """One-sigma implied price move over *horizon_days*, in percent."""
        return float(iv_pct * np.sqrt(horizon_days / trading_days))

    def movement_summary(self, horizons: tuple[int, ...] = (1, 15, 22, 30)) -> pd.DataFrame:
        """Implied vs realized moves and spot ± implied ranges for each horizon."""
        if self.features is None:
            raise ValueError("Call build_features() or fit_once() first.")

        returns = self.features["SPX_ret"].dropna()
        iv = float(self.features["IV"].iloc[-1])
        spot = float(self.features["SPX"].iloc[-1])
        rows = []
        for n in horizons:
            implied = self.implied_movement_pct(iv, n, self.trading_days)
            if n == 1:
                historical = float(returns.tail(22).abs().mean() * 100)
            else:
                window = returns.tail(n)
                rv_n = float(window.std() * np.sqrt(self.trading_days) * 100)
                historical = self.implied_movement_pct(rv_n, n, self.trading_days)
            move = implied / 100
            rows.append(
                {
                    "implied_%": implied,
                    "historical_%": historical,
                    "implied_pts": spot * move,
                    "historical_pts": spot * historical / 100,
                    "range_low": spot * (1 - move),
                    "range_high": spot * (1 + move),
                }
            )

        summary = pd.DataFrame(rows, index=[f"{n}d" for n in horizons])
        summary.attrs["spot"] = spot
        summary.attrs["iv"] = iv
        return summary

    @staticmethod
    def _format_spot_price(price: float) -> str:
        if price >= 1000:
            return f"{price:,.0f}"
        return f"{price:.2f}"

    @staticmethod
    def _format_move_value_pct(points: float, pct: float) -> str:
        pts_str = f"{points:,.0f}" if points >= 10 else f"{points:.1f}"
        if pct < 10:
            return f"{pts_str}({pct:.1f}%)"
        return f"{pts_str}({pct:.0f}%)"

    def format_movement_table(self, summary: pd.DataFrame | None = None) -> pd.DataFrame:
        if summary is None:
            summary = self.movement_summary()
        rows = []
        for idx in summary.index:
            rows.append(
                {
                    "Implied": self._format_move_value_pct(
                        summary.loc[idx, "implied_pts"],
                        summary.loc[idx, "implied_%"],
                    ),
                    "Historical": self._format_move_value_pct(
                        summary.loc[idx, "historical_pts"],
                        summary.loc[idx, "historical_%"],
                    ),
                    "Spot ± implied": (
                        f"{self._format_spot_price(summary.loc[idx, 'range_low'])}"
                        f" | {self._format_spot_price(summary.loc[idx, 'range_high'])}"
                    ),
                }
            )
        return pd.DataFrame(rows, index=summary.index)

    def print_movement_table(self, summary: pd.DataFrame | None = None) -> pd.DataFrame:
        if summary is None:
            summary = self.movement_summary()
        spot = summary.attrs["spot"]
        iv = summary.attrs["iv"]
        label = self.underly_ticker.removeprefix("^")
        table = self.format_movement_table(summary)
        columns = ["Horizon", "Implied", "Historical", "Spot ± implied"]
        rows = [
            [idx, table.loc[idx, "Implied"], table.loc[idx, "Historical"], table.loc[idx, "Spot ± implied"]]
            for idx in table.index
        ]
        widths = [
            max(len(columns[i]), *(len(str(row[i])) for row in rows))
            for i in range(len(columns))
        ]

        def _pipe_row(cells: list[str]) -> str:
            return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))

        print(f"\n{label} implied vs realized move (1σ)")
        print(
            f"Spot = {self._format_spot_price(spot)}"
            f" | IV = {iv:.1f}% | 1d hist: avg |move| last 22d"
        )
        print(_pipe_row(columns))
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(_pipe_row(row))
        return table

    def update_latest_prices(self, prices: pd.DataFrame | None = None) -> pd.DataFrame:
        if prices is None:
            prices = self.prices

        updated = prices.copy()
        quotes: dict[str, tuple[pd.Timestamp, float]] = {}
        for column, ticker in {"SPX": self.underly_ticker, "VIX": self.vol_ticker}.items():
            latest = self._latest_price(ticker)
            if latest is not None:
                quotes[column] = latest

        if quotes:
            target_date = max(quote_date for quote_date, _ in quotes.values())
            for column, (_, quote_price) in quotes.items():
                updated.loc[target_date, column] = quote_price

        self.prices = updated.dropna(subset=["SPX", "VIX"]).sort_index()
        return self.prices

    def plot(
        self,
        features: pd.DataFrame | None = None,
        regimes: pd.DataFrame | None = None,
        display_period: int | None = None,
        update_latest: bool = False,
        show_strategy: bool = False,
        highlight_today: bool = True,
    ) -> None:
        if update_latest:
            features = self.build_features(self.update_latest_prices())
        if features is None:
            features = self.features
        if regimes is None:
            regimes = self.regimes

        hmm = regimes["hmm"] if isinstance(regimes, pd.DataFrame) else regimes
        plot_regimes = hmm.copy()

        plot_cols = ["SPX", "RV_22", "IV"]
        if show_strategy:
            if getattr(self, "study_df", None) is None:
                self.add_strategy_pnl(features.join(regimes, how="inner"))
            plot_cols.append("strategy_cum_return_rate")

        plot_df = features[plot_cols].join(plot_regimes.rename("regime"), how="inner").dropna()
        if display_period is not None:
            plot_df = plot_df.tail(display_period)

        n_rows = 3 if show_strategy and "strategy_cum_return_rate" in plot_df.columns else 2
        height_ratios = [2.2, 1, 1] if n_rows == 3 else [2.2, 1]
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(15, 10 if n_rows == 3 else 8),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if n_rows == 2:
            axes = list(axes)

        label = self.underly_ticker.removeprefix("^")
        axes[0].plot(plot_df.index, plot_df["SPX"], color="black", linewidth=1.2, label=label)
        high_vol = plot_df["regime"].eq("high_vol")
        blocks = high_vol.ne(high_vol.shift()).cumsum()
        for _, block in plot_df[high_vol].groupby(blocks[high_vol]):
            for ax in axes[:n_rows]:
                ax.axvspan(block.index[0], block.index[-1], color="red", alpha=0.16, linewidth=0)

        today_date = self.today_date
        if highlight_today and today_date is not None and today_date in plot_df.index:
            for ax in axes[:n_rows]:
                ax.axvline(today_date, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.85)
            axes[0].annotate(
                "today OOS",
                xy=(today_date, plot_df.loc[today_date, "SPX"]),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=9,
                color="tab:green",
            )

        axes[0].set_title(f"{label} with HMM high-vol regime shading")
        axes[0].set_ylabel(label)
        axes[0].legend(loc="upper left")

        vol_label = self.vol_ticker.removeprefix("^")
        axes[1].plot(plot_df.index, plot_df["RV_22"], color="tab:blue", linewidth=1.0, label="22d realized vol %")
        axes[1].plot(
            plot_df.index,
            plot_df["IV"],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.75,
            label=f"{vol_label} (IV)",
        )
        axes[1].set_ylabel("Annualized vol %")
        axes[1].legend(loc="upper left")

        if n_rows == 3:
            axes[2].plot(
                plot_df.index,
                plot_df["strategy_cum_return_rate"] * 100,
                color="tab:green",
                linewidth=1.2,
                label="cum strategy return",
            )
            axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
            axes[2].set_ylabel("Cumulative return (%)")
            axes[2].legend(loc="upper left")

        plt.show()

    def _shade_high_vol(self, axes: list, index: pd.Index, high_vol: pd.Series) -> None:
        """Red background spans for in-sample high-vol days."""
        if high_vol.empty or not high_vol.any():
            return
        for i in range(len(index) - 1):
            if high_vol.iloc[i]:
                for ax in axes:
                    ax.axvspan(index[i], index[i + 1], color="red", alpha=0.25, lw=0)

    @staticmethod
    def _plot_candles(ax: plt.Axes, ohlc: pd.DataFrame) -> None:
        """Draw daily OHLC candlesticks on *ax*."""
        width = 0.6
        for date, row in ohlc.iterrows():
            open_ = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            color = "tab:green" if close >= open_ else "tab:red"
            x = date2num(date)
            ax.vlines(x, low, high, color=color, linewidth=1.0)
            body_bottom = min(open_, close)
            body_height = abs(close - open_)
            if body_height == 0:
                ax.hlines(open_, x - width / 2, x + width / 2, color=color, linewidth=1.0)
            else:
                ax.bar(
                    x,
                    body_height,
                    bottom=body_bottom,
                    width=width,
                    color=color,
                    edgecolor=color,
                    align="center",
                )

    @staticmethod
    def _format_volume_tick(value: float, _pos: int) -> str:
        if value >= 1e9:
            return f"{value / 1e9:.1f}B"
        if value >= 1e6:
            return f"{value / 1e6:.0f}M"
        if value >= 1e3:
            return f"{value / 1e3:.0f}K"
        return f"{value:.0f}"

    @staticmethod
    def _plot_volume(ax: plt.Axes, ohlc: pd.DataFrame) -> None:
        """Draw daily volume bars colored by up/down day."""
        width = 0.6
        for date, row in ohlc.iterrows():
            volume = float(row.get("Volume", 0) or 0)
            if volume <= 0:
                continue
            color = "tab:green" if float(row["Close"]) >= float(row["Open"]) else "tab:red"
            ax.bar(
                date2num(date),
                volume,
                width=width,
                color=color,
                alpha=0.65,
                edgecolor=color,
                align="center",
            )
        ax.set_ylabel("Volume")
        ax.yaxis.set_major_formatter(FuncFormatter(HMMVolatilityRegime._format_volume_tick))

    def _plot_signal_window(self, display_period: int) -> None:
        if self.features is None or self.today_signal is None or self.regimes is None:
            raise ValueError("Call fit_once() before plotting the signal window.")

        plot_df = self.features[["SPX", "RV_22", "IV"]].tail(display_period)
        regime_plot = self.regimes.reindex(plot_df.index)
        in_sample_high = regime_plot["hmm"].eq("high_vol") & regime_plot.get(
            "sample", pd.Series("in_sample", index=plot_df.index)
        ).eq("in_sample")

        today_date = self.today_date
        signal = self.today_signal
        p_today = signal["prob_low_vol"]
        p_tmr = signal["prob_low_vol_tmr"]

        movement = self.movement_summary()

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 0.65, 1]},
        )
        axes = list(axes)
        price_ax, volume_ax, vol_ax = axes
        label = self.underly_ticker.removeprefix("^")

        self._shade_high_vol(axes, plot_df.index, in_sample_high.fillna(False))
        ohlc = self._download_ohlc(
            self.underly_ticker,
            start=plot_df.index[0],
            end=plot_df.index[-1],
        ).reindex(plot_df.index)
        ohlc = ohlc.dropna(how="any")
        if ohlc.empty:
            raise ValueError(
                f"No OHLC data for {self.underly_ticker} over the display window "
                f"({plot_df.index[0].date()} to {plot_df.index[-1].date()})."
            )
        self._plot_candles(price_ax, ohlc)
        (close_line,) = price_ax.plot(
            ohlc.index,
            ohlc["Close"],
            color="black",
            linewidth=1.1,
            alpha=0.85,
            label="close",
            zorder=3,
        )
        if "Volume" in ohlc.columns and ohlc["Volume"].fillna(0).gt(0).any():
            self._plot_volume(volume_ax, ohlc)
        else:
            volume_ax.set_visible(False)
        if today_date is not None and today_date in plot_df.index:
            price_ax.axvline(today_date, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.9)
        price_ax.set_title(label, fontsize=16, pad=28)
        price_ax.text(
            0.5,
            1.01,
            f"P(low vol today) = {p_today:.1%}  |  P(low vol tmr) = {p_tmr:.1%}",
            transform=price_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
        )
        price_ax.set_ylabel(label)
        price_ax.legend(
            handles=[
                Patch(facecolor="tab:green", label=f"{label} up"),
                Patch(facecolor="tab:red", label=f"{label} down"),
                close_line,
            ],
            loc="upper left",
            fontsize=8,
        )

        vol_label = self.vol_ticker.removeprefix("^")
        vol_ax.plot(plot_df.index, plot_df["RV_22"], color="tab:blue", linewidth=1.0, label="22d RV %")
        vol_ax.plot(
            plot_df.index,
            plot_df["IV"],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.75,
            label=f"{vol_label} IV",
        )
        if today_date is not None and today_date in self.features.index:
            vix_22d_ago = self.features["IV"].shift(22).loc[today_date]
            if pd.notna(vix_22d_ago):
                vol_ax.axhline(
                    vix_22d_ago,
                    color="tab:gray",
                    linestyle=":",
                    linewidth=1.2,
                )
        vol_ax.set_ylabel("Annualized vol %")
        vol_ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()
        self.print_movement_table(movement)

    def _download_close(self, ticker: str) -> pd.Series:
        raw = yf.download(ticker, period=self.data_period, auto_adjust=True, progress=False)
        if raw.empty or "Close" not in raw:
            raise ValueError(f"No close data returned for {ticker}")

        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.astype(float)

    def _download_ohlc(
        self,
        ticker: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        kwargs: dict = {"auto_adjust": True, "progress": False}
        if start is not None and end is not None:
            kwargs["start"] = pd.Timestamp(start).normalize()
            kwargs["end"] = (pd.Timestamp(end) + pd.Timedelta(days=1)).normalize()
        else:
            kwargs["period"] = self.data_period

        raw = yf.download(ticker, **kwargs)
        if raw.empty:
            raise ValueError(f"No OHLC data returned for {ticker}")

        ohlc_cols = ["Open", "High", "Low", "Close"]
        missing = [column for column in ohlc_cols if column not in raw.columns]
        if missing:
            raise ValueError(f"Missing OHLC columns for {ticker}: {missing}")

        ohlc = raw[ohlc_cols].copy()
        if isinstance(ohlc.columns, pd.MultiIndex):
            ohlc.columns = ohlc.columns.get_level_values(0)
        if "Volume" in raw.columns:
            volume = raw["Volume"]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            ohlc["Volume"] = volume.astype(float)
        return ohlc.astype(float)

    def _latest_price(self, ticker: str) -> tuple[pd.Timestamp, float] | None:
        # 5d/1m first: ^SPX and some vol indices (^VXD, ^VXN) have no 1d intraday data.
        for period, interval in (("5d", "1m"), ("1d", "1m"), ("5d", "1d")):
            quote = self._latest_price_from_download(ticker, period, interval)
            if quote is not None:
                return quote

        try:
            yf_logger = logging.getLogger("yfinance")
            prev_level = yf_logger.level
            try:
                yf_logger.setLevel(logging.CRITICAL)
                last_price = yf.Ticker(ticker).fast_info.last_price
            finally:
                yf_logger.setLevel(prev_level)
            if last_price is not None and not pd.isna(last_price):
                return pd.Timestamp.now().normalize(), float(last_price)
        except Exception:
            pass
        return None

    @staticmethod
    def _latest_price_from_download(
        ticker: str,
        period: str,
        interval: str,
    ) -> tuple[pd.Timestamp, float] | None:
        yf_logger = logging.getLogger("yfinance")
        prev_level = yf_logger.level
        try:
            yf_logger.setLevel(logging.CRITICAL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
        finally:
            yf_logger.setLevel(prev_level)

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
            latest_time = latest_time.tz_localize(None)

        return latest_time.normalize(), float(close.iloc[-1])

    def _make_hmm(self) -> GaussianHMM:
        return GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=self.hmm_n_iter,
            tol=self.hmm_tol,
            random_state=self.hmm_seed,
            transmat_prior=1.0,
        )

    def _fit_hmm(self, x_train: np.ndarray) -> GaussianHMM | None:
        model = self._make_hmm()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(x_train)
        except Exception:
            return None
        return model

    @staticmethod
    def _state_indices_from_model(
        model: GaussianHMM,
        train_data: pd.DataFrame | None = None,
    ) -> tuple[int, int]:
        """Return (low_state, high_state) from first feature mean.
        High-vol state = state with higher mean on the first feature.
        """
        if model.means_[0][0] >= model.means_[1][0]:
            return 0, 1
        return 1, 0

    def _build_regime_frame(
        self,
        index: pd.Index,
        probs: np.ndarray,
        high_state: int,
        low_state: int,
        model: GaussianHMM,
        in_sample: bool | None = None,
    ) -> pd.DataFrame:
        prob_high_vol = probs[:, high_state]
        prob_low_vol = probs[:, low_state]
        prob_high_vol_tmr = (
            prob_high_vol * model.transmat_[high_state, high_state]
            + prob_low_vol * model.transmat_[low_state, high_state]
        )
        prob_low_vol_tmr = (
            prob_high_vol * model.transmat_[high_state, low_state]
            + prob_low_vol * model.transmat_[low_state, low_state]
        )
        hmm = np.where(prob_low_vol >= self.threshold_today, "low_vol", "high_vol")
        trade_signal = (prob_low_vol >= self.threshold_today) & (
            prob_low_vol_tmr >= self.threshold_tomorrow
        )

        frame = pd.DataFrame(
            {
                "hmm": hmm,
                "trade_signal": trade_signal,
                "prob_high_vol": prob_high_vol,
                "prob_low_vol": prob_low_vol,
                "prob_high_vol_tmr": prob_high_vol_tmr,
                "prob_low_vol_tmr": prob_low_vol_tmr,
            },
            index=index,
        )
        if in_sample is not None:
            frame["sample"] = "in_sample" if in_sample else "oos"
        return frame

    @staticmethod
    def _label_high_low(data: pd.DataFrame, raw_labels: pd.Series) -> pd.Series:
        score = (
            data["RV_22"].groupby(raw_labels).mean().rank()
            + data["sign_flip_22"].groupby(raw_labels).mean().rank()
            - data["trend_63"].groupby(raw_labels).mean().rank()
        )
        high_label = score.idxmax()
        return pd.Series(np.where(raw_labels.eq(high_label), "high_vol", "low_vol"), index=raw_labels.index)
