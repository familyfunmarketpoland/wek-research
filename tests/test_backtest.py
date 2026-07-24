from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
from backtest import grid_search, run_backtest, walk_forward


def _frame(length: int, freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq=freq, tz="UTC")
    open_ = pd.Series(100.0, index=index)
    close = pd.Series(100.0, index=index)
    high = pd.Series(101.0, index=index)
    low = pd.Series(99.0, index=index)
    volume = pd.Series(1000.0, index=index)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _wek(values: list[float], index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values, index=index, dtype=float, name="WEK")


def _three_year_daily_frame() -> pd.DataFrame:
    index = pd.date_range("2021-01-01", "2024-01-01", freq="1D", tz="UTC")
    frame = _frame(len(index), freq="1D")
    frame.index = index
    return frame


def test_next_open_execution_and_both_side_costs() -> None:
    df = _frame(6)
    df.iloc[2, df.columns.get_loc("open")] = 100.0
    df.iloc[4, df.columns.get_loc("open")] = 110.0
    wek = _wek([30.0, 60.0, 60.0, -1.0, -1.0, -1.0], df.index)

    result = run_backtest(df, wek, "mean_reversion", threshold=50.0, exit_bars=10)

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == df.index[2]
    assert trade["exit_time"] == df.index[4]
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["gross_return"] == pytest.approx(0.10)
    assert trade["net_return"] == pytest.approx(0.09685)
    assert result.equity.iloc[-1] == pytest.approx(1.09685)


def test_cross_threshold_equality_is_not_a_cross_until_strictly_above() -> None:
    df = _frame(6)
    wek = _wek([49.0, 50.0, 50.0, 50.1, 60.0, -1.0], df.index)

    result = run_backtest(df, wek, "mean_reversion", threshold=50.0, exit_bars=10)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["entry_time"] == df.index[4]
    assert result.trades.iloc[0]["reason"] == "final_liquidation"


def test_n_bar_exit_happens_after_fully_held_bars_at_next_open() -> None:
    df = _frame(6)
    wek = _wek([0.0, 60.0, 60.0, 60.0, 60.0, 60.0], df.index)

    result = run_backtest(df, wek, "mean_reversion", threshold=50.0, exit_bars=2)

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == df.index[2]
    assert trade["exit_time"] == df.index[4]
    assert trade["bars"] == 2
    assert trade["reason"] == "time"


def test_final_liquidation_is_explicit_and_charges_exit_cost() -> None:
    df = _frame(4)
    df.iloc[2, df.columns.get_loc("open")] = 100.0
    df.iloc[3, df.columns.get_loc("close")] = 120.0
    wek = _wek([0.0, 60.0, 60.0, 60.0], df.index)

    result = run_backtest(df, wek, "mean_reversion", threshold=50.0, exit_bars=10)

    trade = result.trades.iloc[0]
    assert trade["exit_time"] == df.index[-1]
    assert trade["exit_price"] == pytest.approx(120.0)
    assert trade["reason"] == "final_liquidation"
    assert result.equity.iloc[-1] == pytest.approx(-0.0015 + 1.2 - 0.0018)


def test_synthetic_short_manual_path() -> None:
    df = _frame(6)
    df.iloc[2, df.columns.get_loc("open")] = 100.0
    df.iloc[4, df.columns.get_loc("open")] = 90.0
    wek = _wek([0.0, -60.0, -60.0, 1.0, 1.0, 1.0], df.index)

    result = run_backtest(df, wek, "long_short", threshold=50.0, exit_bars=10)

    trade = result.trades.iloc[0]
    assert trade["side"] == "short"
    assert trade["entry_time"] == df.index[2]
    assert trade["exit_time"] == df.index[4]
    assert trade["gross_return"] == pytest.approx(0.10)
    assert trade["net_return"] == pytest.approx(0.09715)
    assert result.equity.iloc[-1] == pytest.approx(1.09715)


def test_metrics_sanity_on_profitable_trade() -> None:
    df = _frame(6)
    df.iloc[2, df.columns.get_loc("open")] = 100.0
    df.iloc[4, df.columns.get_loc("open")] = 110.0
    wek = _wek([0.0, 60.0, 60.0, -1.0, -1.0, -1.0], df.index)

    result = run_backtest(df, wek, "mean_reversion", threshold=50.0, exit_bars=10, bars_per_year=365)

    assert result.metrics["total_return"] > 0
    assert result.metrics["max_drawdown"] <= 0
    assert result.metrics["trades"] == 1
    assert result.metrics["win_rate"] == 1.0
    assert np.isinf(result.metrics["profit_factor"])
    assert 0 < result.metrics["exposure"] < 1


@pytest.mark.parametrize("variant", ["mean_reversion", "trend_filter", "long_short", "breakout"])
def test_all_variants_smoke(variant: str) -> None:
    df = _frame(240)
    ramp = np.linspace(100.0, 220.0, len(df))
    df["open"] = ramp
    df["close"] = ramp + 0.5
    df["high"] = df["close"] + 0.25
    df["low"] = df["open"] - 0.25
    wek_values = np.zeros(len(df))
    wek_values[209:] = 60.0
    wek = pd.Series(wek_values, index=df.index)

    result = run_backtest(df, wek, variant, threshold=50.0, exit_bars=5)

    assert result.equity.index.equals(df.index)
    assert set(["total_return", "Sharpe", "max_drawdown", "trades"]).issubset(result.metrics)


@pytest.mark.parametrize("variant", ["mean_reversion", "trend_filter", "long_short", "breakout"])
def test_explicit_market_features_match_public_backtest_fallback(variant: str) -> None:
    df = _frame(240)
    ramp = np.linspace(100.0, 220.0, len(df))
    df["open"] = ramp
    df["close"] = ramp + 0.5
    df["high"] = df["close"] + 0.25
    df["low"] = df["open"] - 0.25
    wek_values = np.zeros(len(df))
    wek_values[209:215] = 60.0
    wek_values[220:225] = -60.0
    wek = pd.Series(wek_values, index=df.index)
    config = backtest.BacktestConfig(variant=variant, threshold=50.0, exit_bars=5)

    fallback = run_backtest(df, wek, variant, threshold=50.0, exit_bars=5)
    explicit = backtest._run_backtest_engine(
        df=df,
        wek=wek,
        config=config,
        features=backtest._compute_market_features(df),
    )

    pd.testing.assert_series_equal(explicit.equity, fallback.equity)
    pd.testing.assert_series_equal(explicit.returns, fallback.returns)
    pd.testing.assert_series_equal(explicit.exposure, fallback.exposure)
    pd.testing.assert_frame_equal(explicit.trades, fallback.trades)
    for name, expected in fallback.metrics.items():
        if np.isnan(expected):
            assert np.isnan(explicit.metrics[name])
        else:
            assert explicit.metrics[name] == pytest.approx(expected)


def test_grid_search_uses_subset_and_returns_sortable_frame() -> None:
    df = _frame(50)
    results = grid_search(
        df,
        lengths=[14],
        smooths=[3],
        thresholds=[40.0, 50.0],
        exit_bars_options=[5],
        variants=["mean_reversion", "long_short"],
    )

    assert len(results) == 4
    assert {"length", "smooth", "variant", "threshold", "exit_bars", "total_return"}.issubset(results.columns)


def test_walk_forward_selects_from_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _frame(460, freq="1D")
    df.iloc[11, df.columns.get_loc("open")] = 100.0
    df.iloc[13, df.columns.get_loc("open")] = 120.0
    df.iloc[21, df.columns.get_loc("open")] = 100.0
    df.iloc[23, df.columns.get_loc("open")] = 90.0
    df.iloc[381, df.columns.get_loc("open")] = 100.0
    df.iloc[383, df.columns.get_loc("open")] = 150.0

    series_by_length: dict[int, pd.Series] = {}
    for length in (14, 20):
        values = np.zeros(len(df))
        if length == 14:
            values[10:12] = 60.0
        else:
            values[20:22] = 60.0
            values[380:382] = 60.0
        series_by_length[length] = pd.Series(values, index=df.index, name="WEK")

    def fake_compute_wek(frame: pd.DataFrame, length: int, smooth: int) -> pd.Series:
        return series_by_length[length].loc[frame.index]

    monkeypatch.setattr(backtest, "_compute_wek", fake_compute_wek)

    kwargs = dict(
        lengths=[14, 20],
        smooths=[3],
        thresholds=[50.0],
        exit_bars_options=[10],
        variants=["mean_reversion"],
        train_months=12,
        oos_months=3,
        step_months=3,
        min_trades=1,
    )
    result = walk_forward(df, **kwargs)

    assert not result.folds.empty
    assert int(result.folds.iloc[0]["length"]) == 14
    assert result.folds.iloc[0]["oos_start"] >= result.folds.iloc[0]["train_end"]

    changed = df.copy()
    first_oos = changed.index >= result.folds.iloc[0]["train_end"]
    changed.loc[first_oos, ["open", "high", "low", "close"]] *= 100.0
    changed_result = walk_forward(changed, **kwargs)

    selection_columns = ["length", "smooth", "variant", "threshold", "exit_bars", "Sharpe"]
    pd.testing.assert_series_equal(
        result.selected_train_results.loc[0, selection_columns],
        changed_result.selected_train_results.loc[0, selection_columns],
    )


def test_walk_forward_precomputes_full_history_features_once_for_later_train_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _frame(100 * 24)
    original_compute = backtest._compute_market_features
    original_grid = backtest._grid_from_cache
    compute_calls = 0
    train_boundaries: list[tuple[pd.Timestamp, float, float, float]] = []

    def tracked_compute(frame: pd.DataFrame) -> backtest._MarketFeatures:
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(frame)

    def tracked_grid(
        frame: pd.DataFrame,
        wek_cache: dict[tuple[int, int], pd.Series],
        **kwargs: object,
    ) -> pd.DataFrame:
        features = kwargs["features"]
        assert isinstance(features, backtest._MarketFeatures)
        start = frame.index[0]
        train_boundaries.append(
            (
                start,
                float(features.ema200.loc[start]),
                float(features.prior_high.loc[start]),
                float(features.prior_low.loc[start]),
            )
        )
        return original_grid(frame, wek_cache, **kwargs)

    monkeypatch.setattr(backtest, "_compute_market_features", tracked_compute)
    monkeypatch.setattr(backtest, "_grid_from_cache", tracked_grid)

    walk_forward(
        df,
        lengths=[14],
        smooths=[3],
        thresholds=[50.0],
        exit_bars_options=[5],
        variants=["trend_filter"],
        train_months=1,
        oos_months=1,
        step_months=1,
        objective="total_return",
        min_trades=0,
    )

    assert compute_calls == 1
    assert len(train_boundaries) >= 2
    _, ema200, prior_high, prior_low = train_boundaries[1]
    assert np.isfinite(ema200)
    assert np.isfinite(prior_high)
    assert np.isfinite(prior_low)


@pytest.mark.parametrize("variant", ["trend_filter", "breakout"])
def test_walk_forward_oos_boundary_uses_full_history_features_and_signal_context(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    df = _frame(456, freq="1D")
    close = np.linspace(100.0, 550.0, len(df))
    df["open"] = close
    df["close"] = close
    df["high"] = close + 0.1
    df["low"] = close - 0.1
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    signal_time = oos_start - pd.Timedelta(days=1)
    wek = pd.Series(0.0, index=df.index, name="WEK")
    wek.loc[signal_time] = 60.0

    def fake_compute_wek(frame: pd.DataFrame, length: int, smooth: int) -> pd.Series:
        return wek.loc[frame.index]

    monkeypatch.setattr(backtest, "_compute_wek", fake_compute_wek)

    result = walk_forward(
        df,
        lengths=[14],
        smooths=[3],
        thresholds=[50.0],
        exit_bars_options=[2],
        variants=[variant],
        train_months=12,
        oos_months=3,
        step_months=3,
        objective="total_return",
        min_trades=0,
        include_fixed_oos=True,
    )

    features = backtest._compute_market_features(df)
    assert np.isfinite(features.ema200.loc[signal_time])
    assert np.isfinite(features.prior_high.loc[signal_time])
    assert result.oos_trades.iloc[0]["entry_time"] == oos_start
    assert result.fixed_oos_results.iloc[0]["oos_trades"] == 1.0


def test_first_active_return_and_metrics_include_queued_entry_cost_and_intrabar_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _frame(456, freq="1D")
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    signal_time = oos_start - pd.Timedelta(days=1)
    active_prices = [(100.0, 110.0), (110.0, 108.0), (108.0, 107.0), (107.0, 107.0)]
    for offset, (open_price, close_price) in enumerate(active_prices):
        timestamp = oos_start + pd.Timedelta(days=offset)
        df.loc[timestamp, "open"] = open_price
        df.loc[timestamp, "close"] = close_price
        df.loc[timestamp, "high"] = max(open_price, close_price) + 1.0
        df.loc[timestamp, "low"] = min(open_price, close_price) - 1.0

    wek = pd.Series(0.0, index=df.index, name="WEK")
    wek.loc[signal_time : oos_start + pd.Timedelta(days=2)] = 60.0

    def fake_compute_wek(frame: pd.DataFrame, length: int, smooth: int) -> pd.Series:
        return wek.loc[frame.index]

    monkeypatch.setattr(backtest, "_compute_wek", fake_compute_wek)

    result = walk_forward(
        df,
        lengths=[14],
        smooths=[3],
        thresholds=[50.0],
        exit_bars_options=[3],
        variants=["mean_reversion"],
        train_months=12,
        oos_months=3,
        step_months=3,
        objective="total_return",
        min_trades=0,
        bars_per_year=365.0,
        include_fixed_oos=True,
    )

    expected_first_return = -0.0015 + (110.0 / 100.0) - 1.0
    assert result.oos_returns.index[0] == oos_start
    assert result.oos_returns.iloc[0] == pytest.approx(expected_first_return)
    assert result.oos_equity.iloc[0] == pytest.approx(1.0 + expected_first_return)

    expected_sharpe = (
        result.oos_returns.mean() / result.oos_returns.std(ddof=0) * np.sqrt(365.0)
    )
    downside = result.oos_returns[result.oos_returns < 0]
    expected_sortino = result.oos_returns.mean() / downside.std(ddof=0) * np.sqrt(365.0)
    assert result.folds.iloc[0]["oos_Sharpe"] == pytest.approx(expected_sharpe)
    assert result.oos_metrics["Sharpe"] == pytest.approx(expected_sharpe)
    assert result.oos_metrics["Sortino"] == pytest.approx(expected_sortino)
    assert result.fixed_oos_results.iloc[0]["oos_Sharpe"] == pytest.approx(expected_sharpe)
    assert result.fixed_oos_results.iloc[0]["oos_Sortino"] == pytest.approx(expected_sortino)


def test_walk_forward_emits_only_eight_complete_folds_for_exact_three_year_frame() -> None:
    df = _three_year_daily_frame()

    result = walk_forward(
        df,
        lengths=[14],
        smooths=[3],
        thresholds=[50.0],
        exit_bars_options=[5],
        variants=["mean_reversion"],
        train_months=12,
        oos_months=3,
        step_months=3,
        objective="total_return",
        min_trades=0,
    )

    expected_oos_index = pd.date_range("2022-01-01", "2023-12-31", freq="1D", tz="UTC")
    assert len(result.folds) == 8
    assert result.folds.iloc[-1]["oos_end"] == expected_oos_index[-1]
    pd.testing.assert_index_equal(result.oos_returns.index, expected_oos_index)


def test_walk_forward_all_nan_objectives_emit_flat_folds_and_fixed_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _three_year_daily_frame()
    zero_wek = pd.Series(0.0, index=df.index, name="WEK")

    def fake_compute_wek(frame: pd.DataFrame, length: int, smooth: int) -> pd.Series:
        return zero_wek.loc[frame.index]

    monkeypatch.setattr(backtest, "_compute_wek", fake_compute_wek)

    result = walk_forward(
        df,
        lengths=[14, 20],
        smooths=[3],
        thresholds=[50.0],
        exit_bars_options=[5],
        variants=["mean_reversion"],
        train_months=12,
        oos_months=3,
        step_months=3,
        objective="Sharpe",
        min_trades=1,
        include_fixed_oos=True,
    )

    expected_oos_index = pd.date_range("2022-01-01", "2023-12-31", freq="1D", tz="UTC")
    assert len(result.folds) == 8
    assert len(result.selected_train_results) == 8
    assert (result.folds["selection_fallback"] == "no_finite_objective").all()
    assert (
        result.selected_train_results["selection_fallback"] == "no_finite_objective"
    ).all()
    assert (result.selected_train_results["length"] == 14).all()
    assert result.selected_train_results["Sharpe"].isna().all()
    pd.testing.assert_index_equal(result.oos_returns.index, expected_oos_index)
    assert (result.oos_returns == 0.0).all()
    assert (result.oos_equity == 1.0).all()
    assert result.oos_trades.empty
    assert result.oos_metrics["total_return"] == 0.0
    assert result.oos_metrics["trades"] == 0.0
    assert result.oos_metrics["exposure"] == 0.0
    assert len(result.fixed_oos_results) == 16
    assert (result.fixed_oos_results["oos_total_return"] == 0.0).all()
    assert (result.fixed_oos_results["oos_trades"] == 0.0).all()
    assert result.fixed_oos_results["oos_Sharpe"].isna().all()
