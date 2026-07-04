"""
Surface Delta PCA -- sentiment decomposition from IV surface changes.

Pipeline:
  build_iv_grid_delta()  ->  feature matrix  ->  delta-surface  ->  PCA  ->  sentiment

Two grid functions, two uses:
  - K/S grid  (build_iv_grid):  smile visualization per expiry
  - Delta grid (build_iv_grid_delta):  PCA input, surface dynamics, local vol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from vol_surface import (
    TRADING_DAYS,
    build_iv_grid_delta,
    latest_business_session,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SurfacePCAConfig:
    n_components: int = 4
    delta_lo: float = -0.5
    delta_hi: float = 0.5
    n_delta: int = 21
    max_dte: int = 60
    n_dte: int | None = None
    baseline_window: int = 21
    smooth_days: int = 5
    z_threshold_mild: float = 1.0
    z_threshold_strong: float = 2.0
    percentile_extreme: float = 90.0


_DEFAULT_CONFIG = SurfacePCAConfig()


class SurfaceDeltaPCA:
    """PCA on delta-space IV surface changes for sentiment decomposition.

    Step-by-step:
    1.  For each session, build (DTE x delta) IV grid with OTM-only filtering
        (using build_iv_grid_delta) and flatten to a feature vector.
    2.  Compute delta = today's surface vector minus a rolling baseline
        (default 21-day average, excluding the most recent 5 days).
    3.  Standardize per grid node (z-score across time) then fit PCA.
    4.  Map today's PC scores to sentiment signals using the empirical
        distribution of each PC from the fit history.
    """

    def __init__(self, config: SurfacePCAConfig | None = None):
        if PCA is None:
            raise ImportError(
                "scikit-learn is required for SurfaceDeltaPCA. "
                "Install: pip install scikit-learn"
            )
        self.cfg = config or _DEFAULT_CONFIG
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._feature_shape: tuple[int, int] | None = None
        self._score_history: dict[str, np.ndarray] | None = None

    # -- feature extraction ------------------------------------------

    def extract_features(
        self, surfaces: dict[date, pd.DataFrame]
    ) -> tuple[np.ndarray, list[date], np.ndarray, np.ndarray]:
        """Build feature matrix X from surface history.

        For each valid session, builds (DTE x delta) grid via
        build_iv_grid_delta, then flattens IV values into a feature vector.

        Returns
        -------
        X : ndarray (n_sessions, n_dte * n_delta)
            Feature matrix, one row per date.
        dates : list of date
            Dates corresponding to each row.
        grid_dte, grid_delta : ndarray
            Grid coordinates for reshaping / plotting.
        """
        cfg = self.cfg
        dates = sorted(surfaces.keys())
        delta_grid = np.linspace(cfg.delta_lo, cfg.delta_hi, cfg.n_delta)

        rows: list[np.ndarray] = []
        valid_dates: list[date] = []
        ref_grid: tuple[np.ndarray, np.ndarray] | None = None

        for d in dates:
            df = surfaces[d]
            try:
                g_dte, g_delta, iv = build_iv_grid_delta(
                    df,
                    delta_grid=delta_grid,
                    max_dte=cfg.max_dte,
                )
            except ValueError:
                logger.warning("Skipping %s (insufficient OTM quotes)", d)
                continue

            if ref_grid is None:
                ref_grid = (g_dte, g_delta)
            rows.append(iv.ravel())
            valid_dates.append(d)

        if not rows:
            raise ValueError(
                "No valid delta-grid surfaces in history "
                "(need at least 4 OTM options per session)."
            )

        grid_dte, grid_delta = ref_grid
        X = np.array(rows)
        self._feature_shape = grid_dte.shape
        return X, valid_dates, grid_dte, grid_delta

    # -- delta computation -------------------------------------------

    def compute_deltas(
        self, X: np.ndarray, baseline: str = "rolling"
    ) -> np.ndarray:
        """Compute delta-surface = X[t] - baseline(t).

        Parameters
        ----------
        X : ndarray (n_sessions, n_features)
            Stacked feature vectors in chronological order.
        baseline : str
            - "rolling": rolling average of baseline_window days,
                excluding the most recent smooth_days.
            - "prev": X[t] - X[t-1] (day-over-day change).

        Returns
        -------
        D : ndarray (n_sessions, n_features)
        """
        cfg = self.cfg
        D = np.zeros_like(X)

        if baseline == "rolling":
            for i in range(cfg.smooth_days, len(X)):
                end = i - cfg.smooth_days
                start = max(0, i - cfg.baseline_window - cfg.smooth_days)
                if end > start:
                    baseline_mean = np.nanmean(X[start:end], axis=0)
                    D[i] = X[i] - baseline_mean
 
            # Fallback: if rolling produced too few valid deltas to fit PCA components,
            # use day-over-day changes instead so we don't crash with standard scaling or empty arrays.
            valid_count = (np.abs(D).sum(axis=1) > 1e-10).sum()
            if valid_count < min(cfg.n_components, len(X) - 1) or valid_count < 2:
                logger.warning(
                    "Not enough valid rolling delta-surfaces (%d) to fit PCA components (%d) or StandardScale safely. "
                    "Falling back to 'prev' (day-over-day change).",
                    valid_count, cfg.n_components,
                )
                D = np.zeros_like(X)
                D[1:] = X[1:] - X[:-1]
 
        elif baseline == "prev":
            D[1:] = X[1:] - X[:-1]
        else:
            raise ValueError(f"Unknown baseline method: {baseline}")

        return D

    # -- PCA ---------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | None = None,
        D: np.ndarray | None = None,
        surfaces: dict[date, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Fit PCA on delta-surface matrix.

        Provide either (X, D) precomputed, or surfaces to compute from scratch.

        Returns a result dict with PCA metadata and score history.
        """
        cfg = self.cfg
        if surfaces is not None:
            X, dates, grid_dte, grid_delta = self.extract_features(surfaces)
            D = self.compute_deltas(X)
        elif X is None or D is None:
            raise ValueError("Provide surfaces or (X, D) pair.")

        valid = np.isfinite(D).all(axis=1) & (np.abs(D).sum(axis=1) > 1e-10)

        if valid.sum() == 0:
            raise ValueError(
                "No valid delta-surface samples after filtering. "
                f"This typically happens when there are too few sessions ({len(D)}) "
                f"for the rolling baseline (baseline_window={cfg.baseline_window}, "
                f"smooth_days={cfg.smooth_days}). "
                "Try reducing baseline_window/smooth_days, increasing lookback_days, "
                "or using baseline='prev'."
            )
 
        scaler = StandardScaler()
        D_scaled = scaler.fit_transform(D[valid])

        n_samples, n_features = D_scaled.shape
        max_components = min(cfg.n_components, n_samples, n_features)
        if max_components < 1:
            raise ValueError(
                "Not enough valid delta-surface samples/features for PCA. "
                f"Need at least 1 component, got samples={n_samples}, features={n_features}."
            )
        if max_components < cfg.n_components:
            logger.warning(
                "Requested n_components=%d, but only %d are available "
                "(n_samples=%d, n_features=%d). Using %d.",
                cfg.n_components,
                min(n_samples, n_features),
                n_samples,
                n_features,
                max_components,
            )

        pca = PCA(n_components=max_components)
        pca.fit(D_scaled)

        self._pca = pca
        self._scaler = scaler

        scores_all = pca.transform(D_scaled)
        self._score_history = {
            "scores": scores_all,
            "dates": [d for d, v in zip(dates, valid) if v] if surfaces is not None else None,
        }

        n_dte, n_delta = self._feature_shape
        n_components_fit = pca.n_components_
        loadings_reshaped = pca.components_.reshape(
            n_components_fit, n_dte, n_delta
        )

        if surfaces is not None:
            grid_dte_arr, grid_delta_arr = grid_dte, grid_delta
        else:
            grid_dte_arr = grid_delta_arr = None

        return {
            "pca": pca,
            "scaler": scaler,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "loadings": loadings_reshaped,
            "grid_dte": grid_dte_arr,
            "grid_delta": grid_delta_arr,
            "score_history": self._score_history["scores"],
            "score_dates": self._score_history["dates"],
            "n_valid_sessions": int(valid.sum()),
            "n_components": int(n_components_fit),
        }

    def transform(self, delta_row: np.ndarray) -> np.ndarray:
        """Project a single delta-surface vector into PC space."""
        if self._pca is None or self._scaler is None:
            raise ValueError("Call fit() before transform().")
        scaled = self._scaler.transform(delta_row.reshape(1, -1))
        return self._pca.transform(scaled)[0]

    # -- sentiment ---------------------------------------------------

    def sentiment_from_scores(
        self,
        today_scores: np.ndarray,
        score_history: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Map PC z-scores to sentiment signals.

        Each PC's deviation from its historical mean is scored using
        empirical percentiles. Returns a dict with per-PC signals and
        a joint summary label.
        """
        hist = (
            score_history
            if score_history is not None
            else self._score_history["scores"]
            if self._score_history is not None
            else None
        )

        n_comp = len(today_scores)

        if hist is not None and len(hist) > 5:
            pc_means = np.nanmean(hist, axis=0)
            pc_stds = np.nanstd(hist, axis=0)
            pc_stds = np.where(pc_stds < 1e-10, 1.0, pc_stds)
        else:
            pc_means = np.zeros(n_comp)
            pc_stds = np.ones(n_comp)

        z_scores = (today_scores - pc_means[:n_comp]) / pc_stds[:n_comp]

        cfg = self.cfg
        per_pc: dict[str, dict[str, Any]] = {}
        joint_flags: list[str] = []

        for i in range(n_comp):
            z = float(z_scores[i])
            if z > cfg.z_threshold_strong:
                intensity = "strong_positive"
            elif z > cfg.z_threshold_mild:
                intensity = "mild_positive"
            elif z < -cfg.z_threshold_strong:
                intensity = "strong_negative"
            elif z < -cfg.z_threshold_mild:
                intensity = "mild_negative"
            else:
                intensity = "neutral"

            per_pc[f"PC{i+1}"] = {
                "z_score": z,
                "intensity": intensity,
                "pctile_est": float(
                    scipy_stats.norm.cdf(z) * 100
                    if hist is None
                    else (hist[:, i] < today_scores[i]).mean() * 100
                ),
            }

        # Joint labels anchored to loadings shape
        pc1 = per_pc.get("PC1", {}).get("z_score", 0)
        pc2 = per_pc.get("PC2", {}).get("z_score", 0)
        pc3 = per_pc.get("PC3", {}).get("z_score", 0)

        if pc1 > cfg.z_threshold_mild and pc2 > cfg.z_threshold_mild:
            joint_flags.append("risk_off_surge")
        elif pc1 < -cfg.z_threshold_mild and pc2 < -cfg.z_threshold_mild:
            joint_flags.append("risk_on_compression")
        elif pc1 > cfg.z_threshold_mild and abs(pc3) > cfg.z_threshold_mild:
            joint_flags.append("tail_uncertainty")
        elif pc2 > cfg.z_threshold_mild and pc3 < -cfg.z_threshold_mild:
            joint_flags.append("put_skew_widening_only")
        elif pc2 < -cfg.z_threshold_mild and pc3 > cfg.z_threshold_mild:
            joint_flags.append("call_skew_widening_only")

        return {
            "z_scores": z_scores.tolist(),
            "per_pc": per_pc,
            "joint_flags": joint_flags,
            "n_components": n_comp,
        }

    def sentiment(
        self,
        surfaces: dict[date, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run the full pipeline for today: extract, delta, transform, sentiment."""
        result = self.fit(surfaces=surfaces)
        today_scores = result["score_history"][-1]
        hist_scores = result["score_history"]
        sent = self.sentiment_from_scores(today_scores, hist_scores)

        return {
            "pca_meta": {
                "explained_variance": result["explained_variance_ratio"].tolist(),
                "cumulative_variance": result["cumulative_variance"].tolist(),
                "n_sessions": result["n_valid_sessions"],
            },
            "sentiment": sent,
            "today_scores": today_scores.tolist(),
            "today_date": str(result["score_dates"][-1]) if result["score_dates"] else None,
        }

    # -- plotting ----------------------------------------------------

    def plot_loadings(
        self,
        result: dict[str, Any],
        figsize: tuple[float, float] = (14, 8),
    ) -> plt.Figure:
        """Heatmap of each PC loading reshaped to (DTE x delta)."""
        loadings = result["loadings"]
        n_comp = int(loadings.shape[0])
        grid_dte = result["grid_dte"]
        grid_delta = result["grid_delta"]
        evr = result["explained_variance_ratio"]

        fig, axes = plt.subplots(2, (n_comp + 1) // 2, figsize=figsize)
        axes = axes.ravel()

        for i in range(n_comp):
            ax = axes[i]
            im = ax.imshow(
                loadings[i].T,
                aspect="auto",
                origin="lower",
                extent=[
                    grid_dte.min(),
                    grid_dte.max(),
                    grid_delta.min(),
                    grid_delta.max(),
                ],
                cmap="RdBu_r",
            )
            plt.colorbar(im, ax=ax, shrink=0.75)
            ax.set_title(f"PC{i+1}  ({evr[i]:.1%} var)")
            ax.set_xlabel("DTE")
            ax.set_ylabel("Delta")

        for j in range(n_comp, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("PCA loadings: each PC reshaped to (DTE x delta) surface")
        fig.tight_layout()
        return fig

    def plot_score_history(
        self,
        result: dict[str, Any],
        figsize: tuple[float, float] = (14, 6),
    ) -> plt.Figure:
        """Time series of PC scores."""
        scores = result["score_history"]
        dates = result["score_dates"]
        evr = result["explained_variance_ratio"]
        n_comp = int(scores.shape[1])

        fig, axes = plt.subplots(n_comp, 1, figsize=figsize, sharex=True)
        if n_comp == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(dates, scores[:, i], marker=".", linewidth=1.0, markersize=3)
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set_ylabel(f"PC{i+1} ({evr[i]:.1%})")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        fig.suptitle("PC score history")
        fig.tight_layout()
        return fig

    def surface_delta_loadings(
        self,
        loadings: np.ndarray,
        grid_dte: np.ndarray,
        grid_delta: np.ndarray,
    ) -> pd.DataFrame:
        """Return a tidy DataFrame of loadings for inspection."""
        records: list[dict[str, Any]] = []
        n_comp = int(loadings.shape[0])
        for i in range(n_comp):
            for j in range(grid_dte.shape[0]):
                for k in range(grid_dte.shape[1]):
                    records.append(
                        {
                            "pc": f"PC{i+1}",
                            "dte": float(grid_dte[j, k]),
                            "delta": float(grid_delta[j, k]),
                            "loading": float(loadings[i, j, k]),
                        }
                    )
        return pd.DataFrame(records)

    # -- LLM skill helper --------------------------------------------

    @staticmethod
    def build_report(
        sentiment_result: dict[str, Any],
        today_features: dict[str, Any] | None = None,
    ) -> str:
        """Render a plain-text sentiment report for an LLM skill."""
        meta = sentiment_result["pca_meta"]
        sent = sentiment_result["sentiment"]
        explained = meta.get("explained_variance")
        if explained is None:
            explained = meta.get("explained_variance_ratio", [])

        cumulative = meta.get("cumulative_variance")
        if cumulative is None and explained:
            cumulative = np.cumsum(np.asarray(explained)).tolist()
        elif cumulative is None:
            cumulative = []

        n_sessions = meta.get("n_sessions")
        if n_sessions is None:
            n_sessions = meta.get("n_valid_sessions", "?")

        if explained:
            ev_terms = [f"PC{i+1}={v:.1%}" for i, v in enumerate(explained)]
            var_line = (
                "Explained variance:  "
                + ", ".join(ev_terms)
                + f", total={cumulative[len(explained) - 1]:.1%} (PC1-{len(explained)})"
            )
        else:
            var_line = "Explained variance: unavailable"

        lines = [
            f"Surface PCA sentiment  |  date: {sentiment_result.get('today_date', '?')}",
            "",
            var_line,
            f"Fitted on {n_sessions} sessions",
            "",
            "Per-PC signals:",
        ]
        for pc_name, pc_data in sent["per_pc"].items():
            lines.append(
                f"  {pc_name}: z={pc_data['z_score']:+.2f}, "
                f"intensity={pc_data['intensity']}, "
                f"pctile={pc_data['pctile_est']:.0f}th"
            )
        if sent["joint_flags"]:
            lines.append("")
            lines.append("Joint flags: " + ", ".join(sent["joint_flags"]))
        if today_features:
            lines.append("")
            lines.append("Surface features:")
            for k, v in sorted(today_features.items()):
                if isinstance(v, float) and np.isfinite(v):
                    lines.append(f"  {k}: {v:.2f}")
        return "\n".join(lines)
