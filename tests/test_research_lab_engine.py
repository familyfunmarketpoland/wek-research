from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab.engine import (
    returns_from_position_path,
    run_signal_backtest,
    walk_forward_candidate,
    walk_forward_fixed_candidates,
)
from research_lab.hypotheses import Candidate, SignalBundle, generate_signals


def _frame(length: int, freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq=freq, tz="UTC")
    open_ = pd.Series(100.0, index=index)
    close = pd.Series(100.0, index=index)
    high = pd.Series(101.0, index=index)
    low = pd.Series(99.0, index=index)
    volume = pd.Series(1000.0, index=index)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_close_signal_fills_next_open_and_charges_entry_exit_costs() -> None:
    df = _frame(5)
    df.iloc[2, df.columns.get_loc("open")] = 100.0
    df.iloc[2, df.columns.get_loc("close")] = 105.0
    df.iloc[3, df.columns.get_loc("open")] = 110.0
    signal = pd.Series([0, 1, 0, 0, 0], index=df.index, dtype=int)
    candidate = Candidate("c", "H1", "BTC/USDT", "1h", hold_bars=1)
    bundle = SignalBundle(candidate, "close", signal, pd.Series(False, index=df.index))

    result = run_signal_backtest(df, bundle)

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == df.index[2]
    assert trade["exit_time"] == df.index[3]
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(110.0)
    expected_terminal = (1 - 0.0015) * (105 / 100) * (110 / 105) * (1 - 0.0015)
    assert trade["net_return"] == pytest.approx(expected_terminal - 1.0)
    assert result.equity.iloc[-1] == pytest.approx(expected_terminal)
    assert result.positions.loc[df.index[2]] == 1
    pd.testing.assert_series_equal(
        result.returns,
        returns_from_position_path(df, result.positions),
    )


def test_reversal_pays_two_sides_and_exposes_position_path() -> None:
    df = _frame(6)
    df.loc[df.index[2], ["open", "close"]] = [100.0, 100.0]
    df.loc[df.index[3], ["open", "close"]] = [100.0, 100.0]
    df.loc[df.index[4], ["open", "close"]] = [90.0, 90.0]
    signal = pd.Series([0, 1, 0, -1, 0, 0], index=df.index, dtype=int)
    candidate = Candidate("h4", "H4", "BTC/USDT", "1h", mode="continuation", streak_length=3, hold_bars=10)
    bundle = SignalBundle(candidate, "close", signal, pd.Series(False, index=df.index), allow_reversal=True)

    result = run_signal_backtest(df, bundle)

    assert result.orders["cost"].tolist()[:3] == pytest.approx([0.0015, 0.001347975, 0.001347975])
    assert result.orders.iloc[1]["from_position"] == 1
    assert result.orders.iloc[2]["to_position"] == -1
    assert result.positions.loc[df.index[2]] == 1
    assert result.positions.loc[df.index[4]] == -1
    assert len(result.trades) == 2
    pd.testing.assert_series_equal(
        result.returns,
        returns_from_position_path(df, result.positions),
    )


def test_h3_open_target_enters_session_open_and_exits_session_end_open() -> None:
    df = _frame(30)
    df.loc[df.index[8], "open"] = 110.0
    candidate = Candidate("h3", "H3", "BTC/USDT", "1h", side="long", session_utc="Asia")
    signals = generate_signals(df, candidate)

    result = run_signal_backtest(df, signals)

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == df.index[0]
    assert trade["exit_time"] == df.index[8]
    assert result.positions.loc[df.index[0]] == 1
    assert result.positions.loc[df.index[7]] == 1
    assert result.positions.loc[df.index[8]] == 0
    pd.testing.assert_series_equal(
        result.returns,
        returns_from_position_path(df, result.positions),
    )


def test_h3_walk_forward_starting_mid_session_waits_for_next_boundary() -> None:
    index = pd.date_range("2024-01-01 07:00", "2024-03-01 07:00", freq="1h", tz="UTC")
    df = _frame(len(index))
    df.index = index
    candidate = Candidate("h3", "H3", "BTC/USDT", "1h", side="long", session_utc="Asia")
    config = {
        "walk_forward": {"train_months": 1, "oos_months": 1, "step_months": 1},
        "costs": {"cost_rate_per_side": 0.0015},
    }

    result = walk_forward_candidate(
        df,
        candidate,
        config=config,
        features=pd.DataFrame(index=df.index),
    )

    oos_start = pd.Timestamp("2024-02-01 07:00", tz="UTC")
    next_session_open = pd.Timestamp("2024-02-02 00:00", tz="UTC")
    assert result.folds.iloc[0]["oos_start"] == oos_start
    assert result.fold_positions.loc[oos_start, "position"] == 0
    assert result.fold_positions.loc[pd.Timestamp("2024-02-01 08:00", tz="UTC"), "position"] == 0
    assert result.fold_positions.loc[next_session_open, "position"] == 1
    assert result.trades.iloc[0]["entry_time"] == next_session_open


def test_h2_opposite_channel_exit_uses_close_signal_to_next_open() -> None:
    df = _frame(8)
    features = pd.DataFrame(index=df.index)
    features["rv20"] = 1.0
    features["rv20_prior_year_lower_tercile"] = 2.0
    features["donchian_high20_prior"] = 105.0
    features["donchian_low20_prior"] = 95.0
    df.loc[df.index[1], "close"] = 106.0
    df.loc[df.index[3], "close"] = 94.0
    df.loc[df.index[2], "open"] = 100.0
    df.loc[df.index[4], "open"] = 90.0
    candidate = Candidate("h2", "H2", "BTC/USDT", "1h", side="long", hold_bars=10)
    signals = generate_signals(df, candidate, features=features)

    result = run_signal_backtest(df, signals)

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == df.index[2]
    assert trade["exit_time"] == df.index[4]
    assert trade["reason"] == "opposite_channel"


def test_short_observed_returns_match_position_replay() -> None:
    df = _frame(5)
    df.loc[df.index[2], ["open", "close"]] = [100.0, 95.0]
    df.loc[df.index[3], ["open", "close"]] = [90.0, 90.0]
    signal = pd.Series([0, -1, 0, 0, 0], index=df.index, dtype=int)
    candidate = Candidate("short", "H4", "BTC/USDT", "1h", mode="continuation", streak_length=3, hold_bars=1)
    bundle = SignalBundle(candidate, "close", signal, pd.Series(False, index=df.index))

    result = run_signal_backtest(df, bundle)

    assert result.trades.iloc[0]["side"] == "short"
    pd.testing.assert_series_equal(
        result.returns,
        returns_from_position_path(df, result.positions),
    )


def test_final_liquidation_keeps_final_bar_exposure_for_replay() -> None:
    df = _frame(4)
    df.loc[df.index[2], ["open", "close"]] = [100.0, 110.0]
    df.loc[df.index[3], ["open", "close"]] = [110.0, 120.0]
    signal = pd.Series([0, 1, 0, 0], index=df.index, dtype=int)
    candidate = Candidate("final", "H1", "BTC/USDT", "1h", hold_bars=10)
    bundle = SignalBundle(candidate, "close", signal, pd.Series(False, index=df.index))

    result = run_signal_backtest(df, bundle)

    assert result.positions.iloc[-1] == 1
    assert result.trades.iloc[-1]["reason"] == "final_liquidation"
    pd.testing.assert_series_equal(
        result.returns,
        returns_from_position_path(df, result.positions),
    )


def test_position_replay_resets_and_liquidates_at_fold_boundaries() -> None:
    df = _frame(6)
    df["open"] = [100.0, 105.0, 110.0, 200.0, 190.0, 180.0]
    df["close"] = [105.0, 110.0, 120.0, 190.0, 180.0, 170.0]
    positions = pd.Series([1, 1, 1, 1, 0, -1], index=df.index, name="position")
    fold_ids = pd.Series([0, 0, 0, 1, 1, 1], index=df.index)

    stitched = returns_from_position_path(df, positions, fold_ids=fold_ids)
    expected = pd.concat(
        [
            returns_from_position_path(df.iloc[:3], positions.iloc[:3]),
            returns_from_position_path(df.iloc[3:], positions.iloc[3:]),
        ]
    )

    pd.testing.assert_series_equal(stitched, expected)
    assert stitched.loc[df.index[3]] == pytest.approx((1 - 0.0015) * (190.0 / 200.0) - 1.0)


def test_walk_forward_candidate_uses_previous_bar_context_but_starts_each_fold_flat() -> None:
    df = _frame(550, freq="1D")
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    signal_time = oos_start - pd.Timedelta(days=1)
    df.loc[oos_start, "open"] = 100.0
    df.loc[oos_start, "close"] = 110.0
    signal = pd.Series(0, index=df.index, dtype=int)
    signal.loc[signal_time] = 1
    candidate = Candidate("c", "H1", "BTC/USDT", "1h", hold_bars=1)

    def fake_generate(frame, received_candidate, *, features=None):
        return SignalBundle(received_candidate, "close", signal, pd.Series(False, index=frame.index))

    import research_lab.engine as engine

    original = engine.generate_signals
    engine.generate_signals = fake_generate
    try:
        result = walk_forward_candidate(df, candidate, bars_per_year=365.0)
    finally:
        engine.generate_signals = original

    assert result.folds.iloc[0]["oos_start"] == oos_start
    assert result.returns.index[0] == oos_start
    assert result.returns.iloc[0] == pytest.approx((1 - 0.0015) * (110.0 / 100.0) - 1.0)
    assert result.trades.iloc[0]["entry_time"] == oos_start


def test_walk_forward_complete_folds_and_fixed_candidates_outputs() -> None:
    df = _frame(len(pd.date_range("2021-01-01", "2024-01-01", freq="1D", tz="UTC")), freq="1D")
    df.index = pd.date_range("2021-01-01", "2024-01-01", freq="1D", tz="UTC")
    candidates = [
        Candidate("a", "H1", "BTC/USDT", "1h", hold_bars=1),
        Candidate("b", "H1", "BTC/USDT", "1h", hold_bars=3),
    ]

    result = walk_forward_fixed_candidates(df, candidates)

    assert result.candidate_results["candidate_id"].tolist() == ["a", "b"]
    assert result.folds.groupby("candidate_id").size().to_dict() == {"a": 8, "b": 8}
    assert set(result.returns) == {"a", "b"}
    assert {"candidate_id", "fold", "position"}.issubset(result.fold_positions.columns)
