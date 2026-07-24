from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
ROLLING_YEAR_BARS = {"1h": 8760, "4h": 2190}


def validate_ohlcv_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must use a UTC DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) != "UTC":
        raise ValueError("frame must use a timezone-aware UTC DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be monotonic increasing")
    if frame.index.has_duplicates:
        raise ValueError("frame index must not contain duplicates")


def compute_features(frame: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    """Return causal feature columns for the frozen H1-H6 hypotheses.

    Prior baselines use ``shift(1)`` before rolling so the current bar cannot
    enter a comparator. Current observables such as RV20, entropy20, and NR7
    setups are stamped only after the current close is known.
    """

    validate_ohlcv_frame(frame)
    if timeframe not in ROLLING_YEAR_BARS:
        raise ValueError(f"timeframe must be one of {tuple(ROLLING_YEAR_BARS)}")

    out = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    log_return = np.log(close / close.shift(1))
    prior_returns = log_return.shift(1)
    prior_volume = volume.shift(1)

    out["log_return"] = log_return
    out["prior_volume_sma20"] = prior_volume.rolling(20, min_periods=20).mean()
    out["prior_return_mean20"] = prior_returns.rolling(20, min_periods=20).mean()
    out["prior_return_std20"] = prior_returns.rolling(20, min_periods=20).std(ddof=0)
    out["return_z20_prior"] = (
        (log_return - out["prior_return_mean20"]) / out["prior_return_std20"]
    ).replace([np.inf, -np.inf], np.nan)

    out["donchian_high20_prior"] = high.shift(1).rolling(20, min_periods=20).max()
    out["donchian_low20_prior"] = low.shift(1).rolling(20, min_periods=20).min()
    out["rv20"] = log_return.rolling(20, min_periods=20).std(ddof=0)

    year_bars = ROLLING_YEAR_BARS[timeframe]
    out["rv20_prior_year_lower_tercile"] = (
        out["rv20"].shift(1).rolling(year_bars, min_periods=year_bars).quantile(1.0 / 3.0)
    )

    out["volume_entropy20"] = _normalized_rolling_entropy(volume, 20)
    out["entropy20_prior_year_median"] = (
        out["volume_entropy20"].shift(1).rolling(year_bars, min_periods=year_bars).median()
    )

    bar_range = high - low
    out["range"] = bar_range
    out["nr7"] = bar_range <= bar_range.rolling(7, min_periods=7).min()
    out["up_streak"] = _exact_streak(log_return, sign=1)
    out["down_streak"] = _exact_streak(log_return, sign=-1)
    return out


def _normalized_rolling_entropy(values: pd.Series, window: int) -> pd.Series:
    def entropy(window_values: np.ndarray) -> float:
        total = float(np.nansum(window_values))
        if total <= 0 or not np.isfinite(total):
            return np.nan
        probs = window_values / total
        probs = probs[np.isfinite(probs) & (probs > 0)]
        if probs.size == 0:
            return np.nan
        return float(-(probs * np.log(probs)).sum() / np.log(window))

    return values.rolling(window, min_periods=window).apply(entropy, raw=True)


def _exact_streak(returns: pd.Series, *, sign: int) -> pd.Series:
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    good = returns.gt(0) if sign == 1 else returns.lt(0)
    groups = good.ne(good.shift(fill_value=False)).cumsum()
    streak = good.groupby(groups).cumcount() + 1
    return streak.where(good, 0).astype(int)
