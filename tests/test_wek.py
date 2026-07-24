from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wek import crossover, crossunder, pine_ema, wek


def _frame(length: int = 30) -> pd.DataFrame:
    index = pd.RangeIndex(length)
    open_ = pd.Series(np.linspace(10.0, 15.8, length), index=index)
    close = open_ + np.where(np.arange(length) % 2 == 0, 0.6, -0.2)
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.linspace(0.4, 0.9, length)
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.linspace(0.3, 0.7, length)
    volume = pd.Series(np.linspace(100.0, 190.0, length), index=index)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_wek_preserves_index_name_and_input() -> None:
    frame = _frame()
    original = frame.copy(deep=True)

    series, components = wek(frame, return_components=True)

    pd.testing.assert_frame_equal(frame, original)
    assert list(series.index) == list(frame.index)
    assert series.name == "WEK"
    assert list(components.index) == list(frame.index)


def test_wek_uses_full_window_nans_then_emits_values() -> None:
    frame = _frame(40)

    series, components = wek(frame, length=20, smooth=5, return_components=True)

    assert series.iloc[:23].isna().all()
    assert pd.notna(series.iloc[23])
    assert components["event_factor"].iloc[:19].isna().all()
    assert pd.notna(components["event_factor"].iloc[19])


def test_wek_handles_zero_volume_and_flat_range_with_floors() -> None:
    frame = _frame(40)
    frame.loc[10:29, "volume"] = 0.0
    frame.loc[15:25, ["open", "high", "low", "close"]] = 12.0

    series, components = wek(frame, length=20, smooth=5, tick_size=0.01, return_components=True)

    assert np.isfinite(components["range"].iloc[20:26]).all()
    assert (components["range"].iloc[20:26] >= 0.01).all()
    assert np.isfinite(components["vol_norm"].iloc[19:30]).all()
    assert np.isfinite(series.dropna()).all()


def test_event_factor_matches_naive_window_shannon_reference() -> None:
    frame = _frame(35)
    frame["volume"] = pd.Series(
        [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5, 5.0, 8.0, 9.5, 7.0, 9.0,
         3.0, 2.5, 3.5, 8.5, 4.0, 6.5, 2.5, 6.0, 4.5, 3.0, 3.5, 8.0, 3.0, 2.0, 7.5,
         9.0, 5.0, 0.5, 2.0, 8.0],
        index=frame.index,
    )

    _, components = wek(frame, length=6, smooth=4, return_components=True)

    expected = []
    window = 6
    for end in range(len(frame)):
        if end < window - 1:
            expected.append(np.nan)
            continue
        values = frame["volume"].iloc[end - window + 1 : end + 1].to_numpy(dtype=float)
        total = values.sum()
        denom = max(total, 1e-10)
        entropy = 0.0
        for value in values:
            probability = value / denom
            if probability > 0.0:
                entropy += -probability * np.log(probability)
        expected.append(1.0 - entropy / np.log(window))

    expected_series = pd.Series(expected, index=frame.index, dtype=float)
    pd.testing.assert_series_equal(
        components["event_factor"],
        expected_series,
        check_names=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_wek_raw_matches_naive_reference_inputs() -> None:
    frame = _frame(18)
    frame["volume"] = pd.Series(
        [2.0, 5.0, 1.0, 4.0, 3.0, 6.0, 8.0, 7.0, 9.0, 5.0, 4.0, 3.0, 2.0, 6.0, 1.5, 2.5, 4.5, 7.5],
        index=frame.index,
    )

    _, components = wek(frame, length=5, smooth=3, tick_size=0.01, return_components=True)

    expected_event = []
    expected_vol_norm = []
    expected_wick_sig = []
    expected_conv_sig = []
    length = 5
    smooth = 3

    wick_asym = components["wick_asym"].to_numpy(dtype=float)
    conviction = components["conviction"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)

    for end in range(len(frame)):
        if end < length - 1:
            expected_event.append(np.nan)
            expected_vol_norm.append(np.nan)
        else:
            v_window = volume[end - length + 1 : end + 1]
            total = v_window.sum()
            denom = max(total, 1e-10)
            entropy = 0.0
            for value in v_window:
                probability = value / denom
                if probability > 0.0:
                    entropy += -probability * np.log(probability)
            expected_event.append(1.0 - entropy / np.log(length))
            expected_vol_norm.append(volume[end] / max(v_window.mean(), 1e-10))

        if end < smooth - 1:
            expected_conv_sig.append(np.nan)
        else:
            expected_conv_sig.append(np.mean(conviction[end - smooth + 1 : end + 1]))

        if end < (length - 1) + (smooth - 1):
            expected_wick_sig.append(np.nan)
        else:
            start = end - smooth + 1
            expected_wick_sig.append(np.mean(wick_asym[start : end + 1] * np.array(expected_vol_norm[start : end + 1])))

    expected_event_series = pd.Series(expected_event, index=frame.index, dtype=float)
    expected_wick_sig_series = pd.Series(expected_wick_sig, index=frame.index, dtype=float)
    expected_conv_sig_series = pd.Series(expected_conv_sig, index=frame.index, dtype=float)
    expected_raw = expected_wick_sig_series * (0.5 + expected_conv_sig_series) * (0.3 + expected_event_series * 1.4)

    pd.testing.assert_series_equal(
        components["event_factor"],
        expected_event_series,
        check_names=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_series_equal(
        components["wick_sig"],
        expected_wick_sig_series,
        check_names=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_series_equal(
        components["conv_sig"],
        expected_conv_sig_series,
        check_names=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_series_equal(
        components["raw"],
        expected_raw,
        check_names=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_pine_ema_seeds_from_first_valid_value() -> None:
    source = pd.Series([np.nan, np.nan, 10.0, 16.0, 4.0], dtype=float)

    result = pine_ema(source, length=3)

    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(10.0)
    assert result.iloc[3] == pytest.approx(13.0)
    assert result.iloc[4] == pytest.approx(8.5)


def test_wek_matches_manual_seed_and_clip_path() -> None:
    frame = _frame(45)
    series, components = wek(frame, length=20, smooth=5, return_components=True)

    first_valid = components["wek_raw"].first_valid_index()
    assert first_valid is not None
    assert series.loc[first_valid] == pytest.approx(
        np.clip(components.loc[first_valid, "wek_raw"], -100.0, 100.0)
    )


def test_crossover_and_crossunder_follow_pine_equality_rules() -> None:
    series = pd.Series([-1.0, 0.0, 0.0, 0.5, 0.0, -0.2], dtype=float)

    cross_up = crossover(series, 0.0)
    cross_down = crossunder(series, 0.0)

    expected_up = pd.Series([False, False, False, True, False, False])
    expected_down = pd.Series([False, False, False, False, False, True])
    pd.testing.assert_series_equal(cross_up, expected_up)
    pd.testing.assert_series_equal(cross_down, expected_down)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length": 1}, "length must be an integer greater than 1"),
        ({"smooth": 0}, "smooth must be a positive integer"),
        ({"tick_size": 0.0}, "tick_size must be positive"),
    ],
)
def test_wek_rejects_invalid_parameters(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wek(_frame(), **kwargs)


def test_wek_requires_ohlcv_columns() -> None:
    frame = _frame().drop(columns=["volume"])

    with pytest.raises(ValueError, match="missing required columns"):
        wek(frame)


@pytest.mark.parametrize(
    ("volume", "message"),
    [
        ([1.0] * 29 + [np.inf], "volume must contain only finite values"),
        ([1.0] * 29 + [-1.0], "volume must be nonnegative"),
    ],
)
def test_wek_rejects_invalid_volume_values(volume: list[float], message: str) -> None:
    frame = _frame(30)
    frame["volume"] = volume

    with pytest.raises(ValueError, match=message):
        wek(frame)
