"""
Rate-limited, retry-safe data fetcher.

API limit: 600 calls / min (configurable).  This module wraps the
existing fetch functions from vol_surface.py with:
  - Token-bucket rate limiter
  - Exponential backoff + jitter on failure
  - Batch-aware throttling (snapshot calls = 1 token each, bulk = pooled)

Usage:
  from data_fetcher import RateLimiter, safe_fetch

  limiter = RateLimiter(calls_per_minute=600)

  # Option A: decorate individual calls
  df = safe_fetch(limiter, fetch_option_chain_futu, ctx, cfg)

  # Option B: context manager for a batch of calls
  with limiter.session("daily_fetch"):
      df = fetch_option_chain_futu(ctx, cfg)
      anchors = fetch_anchor_iv_histories(df, cfg)
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

import pandas as pd

from vol_surface import (
    VolSurfaceConfig,
    fetch_option_chain_futu,
    fetch_anchor_iv_histories,
    fetch_spot,
    fetch_spot_yfinance,
    fetch_vix_context,
    load_surface_history,
    save_surface,
)

try:
    import futu as ft
except ImportError:
    ft = None

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimitConfig:
    calls_per_minute: float = 600
    burst_multiplier: float = 1.0  # allow small bursts (600 * 1.0 = 600 tokens max)
    min_interval: float = 0.05  # minimum spacing between calls (50ms)


_DEFAULT_RL_CONFIG = RateLimitConfig()


class RateLimiter:
    """Token-bucket rate limiter, thread-safe.

    Allows bursts up to the full per-minute allowance, then refills
    at the steady-state rate.

    Example
    -------
    limiter = RateLimiter(600)       # 600 call/min

    limiter.wait()                   # acquire 1 token (blocking)
    limiter.wait(3)                  # acquire 3 tokens (for batched calls)
    """

    def __init__(self, calls_per_minute: float | RateLimitConfig = 600):
        if isinstance(calls_per_minute, RateLimitConfig):
            cfg = calls_per_minute
        else:
            cfg = RateLimitConfig(calls_per_minute=calls_per_minute)

        self._rate = cfg.calls_per_minute / 60.0  # tokens per second
        self._max_tokens = cfg.calls_per_minute * cfg.burst_multiplier
        self._min_interval = cfg.min_interval
        self._tokens = float(self._max_tokens)  # start full
        self._last_refill = time.monotonic()
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self, tokens: float = 1.0) -> float:
        """Wait until *tokens* are available. Returns actual wait time."""
        with self._lock:
            self._refill()

            # Enforce min interval between calls
            now = time.monotonic()
            since_last = now - self._last_call
            if since_last < self._min_interval:
                time.sleep(self._min_interval - since_last)
                self._last_refill += self._min_interval - since_last

            if self._tokens < -1:  # negative = we let them borrow a bit
                needed = -self._tokens
                wait = needed / self._rate
                time.sleep(wait)
                self._tokens = 0.0
                self._last_refill += wait
            else:
                self._tokens -= tokens

            self._last_call = time.monotonic()
            return self._last_call - now

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        gained = elapsed * self._rate
        self._tokens = min(self._tokens + gained, self._max_tokens)
        self._last_refill = now

    @property
    def tokens_available(self) -> float:
        with self._lock:
            self._refill()
            return max(self._tokens, 0.0)


# ---------------------------------------------------------------------------
# Retry with exponential backoff + jitter
# ---------------------------------------------------------------------------

def retry_call(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    rate_limiter: RateLimiter | None = None,
    log: logging.Logger | None = None,
) -> Any:
    """Call *fn* with exponential backoff + jitter.

    Parameters
    ----------
    fn : callable
        Zero-argument callable (use functools.partial to bind args).
    max_retries : int
        Max retry attempts before re-raising.
    base_delay : float
        Initial delay in seconds; doubles each attempt.
    rate_limiter : RateLimiter or None
        Optional rate limiter to pace the overall call rate.

    Returns
    -------
    fn() result
    """
    log = log or logger
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.wait(1.0)

        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2.0 ** attempt) + random.uniform(0, 0.5 * base_delay)
                log.warning(
                    "Retry %d/%d: %s -> waiting %.1fs",
                    attempt + 1, max_retries, exc, delay,
                )
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


def retryable(
    max_retries: int = 3,
    base_delay: float = 1.0,
    rate_limiter: RateLimiter | None = None,
) -> Callable[[F], F]:
    """Decorator version of retry_call."""
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from functools import partial
            return retry_call(
                partial(fn, *args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                rate_limiter=rate_limiter,
            )
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Safe wrapper for the existing fetch functions
# ---------------------------------------------------------------------------

def safe_fetch(
    limiter: RateLimiter,
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Rate-limited + retry-safe call to any fetch function.

    Example
    -------
    limiter = RateLimiter(600)
    df = safe_fetch(limiter, fetch_option_chain_futu, quote_ctx, cfg)
    """
    from functools import partial

    def _call() -> Any:
        return fn(*args, **kwargs)

    return retry_call(_call, max_retries=max_retries, rate_limiter=limiter)


# ---------------------------------------------------------------------------
# Safe batch fetcher – one-shot or scheduled
# ---------------------------------------------------------------------------

@dataclass
class SafeFetchSession:
    """Context for a single fetch session with rate limiting and cache.

    Typical daily workflow:
        session = SafeFetchSession(limiter, cfg)
        session.fetch_surface()        # fetch + cache today's surface
        session.fetch_anchors()        # fetch anchor IV histories
        session.fetch_vix()            # fetch VIX context
        result = session.summarize()   # metadata
    """
    limiter: RateLimiter
    cfg: VolSurfaceConfig = field(default_factory=VolSurfaceConfig)
    _surface: pd.DataFrame | None = field(default=None, init=False)
    _anchors: dict[str, Any] | None = field(default=None, init=False)
    _vix: dict[str, Any] | None = field(default=None, init=False)

    def fetch_surface(
        self,
        quote_ctx: Any | None = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch today's option chain, rate-limited and cached."""
        if self._surface is not None:
            return self._surface

        close_ctx = False
        if quote_ctx is None:
            if ft is None:
                raise ImportError("futu-api required")
            quote_ctx = ft.OpenQuoteContext(host=self.cfg.host, port=self.cfg.port)
            close_ctx = True

        try:
            df = safe_fetch(
                self.limiter,
                fetch_option_chain_futu,
                quote_ctx,
                self.cfg,
                max_retries=max_retries,
            )
        finally:
            if close_ctx:
                quote_ctx.close()

        self._surface = df
        return df

    def fetch_and_cache(
        self,
        quote_ctx: Any | None = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch and persist to disk."""
        df = self.fetch_surface(quote_ctx, max_retries)
        save_surface(df, self.cfg)
        return df

    def fetch_anchors(
        self,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Fetch anchor IV histories, rate-limited."""
        if self._surface is None:
            raise RuntimeError("Call fetch_surface() before fetch_anchors().")
        result = safe_fetch(
            self.limiter,
            fetch_anchor_iv_histories,
            self._surface,
            self.cfg,
            max_retries=max_retries,
        )
        self._anchors = result
        return result

    def fetch_vix(
        self,
        lookback_days: int | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Fetch VIX context, rate-limited."""
        lb = lookback_days or self.cfg.lookback_days
        result = safe_fetch(
            self.limiter,
            fetch_vix_context,
            lb,
            max_retries=max_retries,
        )
        self._vix = result
        return result

    def fetch_all(
        self,
        quote_ctx: Any | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """One-shot: surface + anchors + VIX, all rate-limited."""
        self.fetch_surface(quote_ctx, max_retries)
        self.fetch_anchors()
        self.fetch_vix()
        return self.summarize()

    def summarize(self) -> dict[str, Any]:
        return {
            "surface": self._surface,
            "anchors": self._anchors,
            "vix": self._vix,
        }


# ---------------------------------------------------------------------------
# Historical backfill with rate-limited pacing
# ---------------------------------------------------------------------------

def backfill_surfaces(
    limiter: RateLimiter,
    dates: list[date],
    cfg: VolSurfaceConfig | None = None,
    quote_ctx: Any | None = None,
    max_retries: int = 3,
    progress_callback: Callable[[int, int, date], None] | None = None,
) -> dict[date, pd.DataFrame]:
    """Fetch and cache surfaces for a list of dates, respecting rate limits.

    Skips dates that are already cached.  Accepts a pre-existing quote_ctx
    to avoid opening/closing on every date.
    """
    from datetime import date
    from vol_surface import load_surface, save_surface

    cfg = cfg or VolSurfaceConfig()
    total = len(dates)
    surfaces: dict[date, pd.DataFrame] = {}

    close_ctx = False
    if quote_ctx is None:
        if ft is None:
            raise ImportError("futu-api required")
        quote_ctx = ft.OpenQuoteContext(host=cfg.host, port=cfg.port)
        close_ctx = True

    try:
        for i, d in enumerate(dates):
            if progress_callback:
                progress_callback(i + 1, total, d)

            # Skip if already cached
            cached = load_surface(cfg, d)
            if cached is not None:
                surfaces[d] = cached
                continue

            logger.info("Backfill %d/%d: fetching %s", i + 1, total, d)
            try:
                df = safe_fetch(
                    limiter,
                    fetch_option_chain_futu,
                    quote_ctx,
                    cfg,
                    max_retries=max_retries,
                )
                save_surface(df, cfg)
                surfaces[d] = df
            except Exception as exc:
                logger.error("Backfill failed for %s: %s", d, exc)
                continue
    finally:
        if close_ctx:
            quote_ctx.close()

    return surfaces


# ---------------------------------------------------------------------------
# Convenience: daily pipeline builder
# ---------------------------------------------------------------------------

def daily_pipeline(
    limiter: RateLimiter | None = None,
    cfg: VolSurfaceConfig | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Fetch today's surface, anchor histories, VIX context.

    The standard "good morning" call.  Returns everything needed for
    VolSurfaceStudy.analyze().

    Returns
    -------
    dict with keys: surface, anchors, vix, study
    """
    from vol_surface import VolSurfaceStudy

    limiter = limiter or RateLimiter()
    cfg = cfg or VolSurfaceConfig()
    session = SafeFetchSession(limiter, cfg)
    result = session.fetch_all(max_retries=max_retries)

    study = VolSurfaceStudy(cfg)
    today = result["surface"]
    if today is not None:
        asof = pd.Timestamp(today["asof_date"].iloc[0]).date()
        study.surfaces[asof] = today
        study._compute_surface(asof, today)

    analysis = study.analyze()
    return {
        "session": result,
        "study": study,
        "analysis": analysis,
    }
