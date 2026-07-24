from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
EPSILON = 1e-10


def _validate_inputs(
    frame: pd.DataFrame,
    length: int,
    smooth: int,
    tick_size: float,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")

    if not isinstance(length, int) or isinstance(length, bool) or length <= 1:
        raise ValueError("length must be an integer greater than 1")

    if not isinstance(smooth, int) or isinstance(smooth, bool) or smooth <= 0:
        raise ValueError("smooth must be a positive integer")

    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    volume = frame["volume"]
    if not np.isfinite(volume.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("volume must contain only finite values")
    if (volume.to_numpy(dtype=float, copy=False) < 0.0).any():
        raise ValueError("volume must be nonnegative")


def pine_ema(source: pd.Series, length: int) -> pd.Series:
    alpha = 2.0 / (length + 1.0)
    result = pd.Series(np.nan, index=source.index, dtype=float, name=source.name)

    started = False
    prev = np.nan
    for idx, value in source.items():
        if pd.isna(value):
            continue
        numeric = float(value)
        if not started:
            prev = numeric
            started = True
        else:
            prev = alpha * numeric + (1.0 - alpha) * prev
        result.at[idx] = prev

    return result


def crossover(series: pd.Series, threshold: float | pd.Series) -> pd.Series:
    threshold_series = _coerce_threshold(series.index, threshold)
    prev_series = series.shift(1)
    prev_threshold = threshold_series.shift(1)
    return (series > threshold_series) & (prev_series <= prev_threshold)


def crossunder(series: pd.Series, threshold: float | pd.Series) -> pd.Series:
    threshold_series = _coerce_threshold(series.index, threshold)
    prev_series = series.shift(1)
    prev_threshold = threshold_series.shift(1)
    return (series < threshold_series) & (prev_series >= prev_threshold)


def _coerce_threshold(index: pd.Index, threshold: float | pd.Series) -> pd.Series:
    if isinstance(threshold, pd.Series):
        return threshold.reindex(index)
    return pd.Series(float(threshold), index=index, dtype=float)


def _rolling_entropy(volume: pd.Series, length: int) -> pd.Series:
    sum_vol = volume.rolling(length, min_periods=length).sum()
    denom = sum_vol.clip(lower=EPSILON)
    volume_log_volume = pd.Series(0.0, index=volume.index, dtype=float)
    positive = volume > 0.0
    volume_log_volume.loc[positive] = volume.loc[positive] * np.log(volume.loc[positive])
    rolling_volume_log_volume = volume_log_volume.rolling(length, min_periods=length).sum()
    entropy = (sum_vol / denom) * np.log(denom) - (rolling_volume_log_volume / denom)
    return entropy / math.log(length)


def wek(
    frame: pd.DataFrame,
    length: int = 20,
    smooth: int = 5,
    tick_size: float = EPSILON,
    *,
    return_components: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    _validate_inputs(frame, length=length, smooth=smooth, tick_size=tick_size)

    data = frame.loc[:, REQUIRED_COLUMNS].astype(float)

    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]

    event_factor = 1.0 - _rolling_entropy(volume, length)

    price_range = (high - low).clip(lower=float(tick_size))
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    wick_asym = (lower_wick - upper_wick) / price_range

    volume_sma = volume.rolling(length, min_periods=length).mean().clip(lower=EPSILON)
    vol_norm = volume / volume_sma

    wick_sig = (wick_asym * vol_norm).rolling(smooth, min_periods=smooth).mean()
    conviction = (close - open_).abs() / price_range
    conv_sig = conviction.rolling(smooth, min_periods=smooth).mean()

    raw = wick_sig * (0.5 + conv_sig) * (0.3 + event_factor * 1.4)
    wek_raw = raw * 100.0
    wek_series = pine_ema(wek_raw, smooth).clip(lower=-100.0, upper=100.0)
    wek_series.name = "WEK"

    if not return_components:
        return wek_series

    components = pd.DataFrame(
        {
            "event_factor": event_factor,
            "range": price_range,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "wick_asym": wick_asym,
            "vol_norm": vol_norm,
            "wick_sig": wick_sig,
            "conviction": conviction,
            "conv_sig": conv_sig,
            "raw": raw,
            "wek_raw": wek_raw,
            "WEK": wek_series,
        },
        index=frame.index,
    )
    return wek_series, components


__all__: Iterable[str] = ("wek", "pine_ema", "crossover", "crossunder")
