from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab.features import compute_features
from research_lab.hypotheses import Candidate, enumerate_candidates, generate_signals


def _frame(length: int, freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq=freq, tz="UTC")
    close = pd.Series(100.0 + np.arange(length) * 0.1, index=index)
    open_ = close.copy()
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1000.0, index=index)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_candidate_enumerator_matches_frozen_counts_and_h3_only_1h() -> None:
    candidates = enumerate_candidates()
    counts = pd.Series([candidate.hypothesis_id for candidate in candidates]).value_counts().to_dict()

    assert len(candidates) == 186
    assert counts == {"H4": 48, "H2": 36, "H5": 36, "H6": 36, "H1": 18, "H3": 12}
    assert {candidate.timeframe for candidate in candidates if candidate.hypothesis_id == "H3"} == {"1h"}
    assert not any(candidate.hypothesis_id == "H4" and candidate.side for candidate in candidates)


def test_features_use_prior_baselines_and_frozen_year_windows() -> None:
    df = _frame(8805)
    df["volume"] = 100.0
    df.iloc[20, df.columns.get_loc("volume")] = 10_000.0
    features = compute_features(df, timeframe="1h")

    assert features.loc[df.index[20], "prior_volume_sma20"] == pytest.approx(100.0)
    assert features.loc[df.index[21], "prior_volume_sma20"] == pytest.approx((19 * 100.0 + 10_000.0) / 20.0)
    assert features["rv20_prior_year_lower_tercile"].first_valid_index() == df.index[8780]


def test_h1_signal_uses_current_volume_and_prior_return_baseline() -> None:
    df = _frame(25)
    df["close"] = 100.0
    df.iloc[24, df.columns.get_loc("close")] = 90.0
    df["open"] = df["close"]
    df["high"] = df["close"] + 1.0
    df["low"] = df["close"] - 1.0
    df["volume"] = 100.0
    df.iloc[24, df.columns.get_loc("volume")] = 400.0
    features = compute_features(df, timeframe="1h")
    features.loc[df.index[24], "return_z20_prior"] = -3.0
    candidate = Candidate("h1", "H1", "BTC/USDT", "1h", hold_bars=1)

    signals = generate_signals(df, candidate, features=features)

    assert signals.signal.loc[df.index[24]] == 1


def test_h4_exact_streak_stamps_only_nth_bar_and_handles_both_signs() -> None:
    df = _frame(10)
    df["close"] = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]
    df["open"] = df["close"]
    df["high"] = df["close"] + 1
    df["low"] = df["close"] - 1
    candidate = Candidate("h4", "H4", "BTC/USDT", "1h", mode="reversal", streak_length=3, hold_bars=1)

    signals = generate_signals(df, candidate)

    assert signals.signal.loc[df.index[3]] == -1
    assert signals.signal.loc[df.index[4]] == 0
    assert signals.signal.loc[df.index[8]] == 1
    assert signals.allow_reversal is True


def test_h5_confirms_on_bar_after_nr7_and_stamps_confirmation_close() -> None:
    df = _frame(10)
    df["high"] = [110, 109, 108, 107, 106, 105, 100, 103, 104, 105]
    df["low"] = [90, 91, 92, 93, 94, 95, 99, 98, 99, 100]
    df["close"] = [100, 100, 100, 100, 100, 100, 99.5, 101, 101, 101]
    df["open"] = df["close"]
    candidate = Candidate("h5", "H5", "BTC/USDT", "1h", side="long", hold_bars=1)

    signals = generate_signals(df, candidate)

    assert signals.signal.loc[df.index[6]] == 0
    assert signals.signal.loc[df.index[7]] == 1


def test_new_research_lab_modules_do_not_read_parquet() -> None:
    root = Path(__file__).resolve().parents[1] / "research_lab"
    for name in ("features.py", "hypotheses.py", "engine.py"):
        assert "read_parquet" not in (root / name).read_text(encoding="utf-8")
