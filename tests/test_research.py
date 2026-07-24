from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research
from research import (
    ResearchConfig,
    buy_and_hold_benchmark,
    construct_ablated_wek,
    equity_returns,
    has_credible_edge,
    monte_carlo_permutations,
    select_final_config,
    stability_neighbor_diagnostic,
)


def _frame(length: int = 460, freq: str = "1D") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq=freq, tz="UTC")
    base = 100.0 + np.sin(np.arange(length) / 11.0) * 3.0 + np.arange(length) * 0.03
    open_ = pd.Series(base, index=index)
    close = open_ + np.cos(np.arange(length) / 7.0) * 0.8
    high = pd.concat([open_, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([open_, close], axis=1).min(axis=1) - 1.0
    volume = pd.Series(1000.0 + (np.arange(length) % 13) * 17.0, index=index)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_monte_carlo_is_deterministic_and_terminal_return_is_invariant() -> None:
    trade_returns = [0.1, -0.05, 0.02, -0.01]

    first = monte_carlo_permutations(trade_returns, n=25, seed=42)
    second = monte_carlo_permutations(trade_returns, n=25, seed=42)

    pd.testing.assert_frame_equal(first, second)
    terminal = np.prod(1.0 + np.asarray(trade_returns)) - 1.0
    assert first["terminal_return"].nunique() == 1
    assert first["terminal_return"].iloc[0] == pytest.approx(terminal)


def test_final_postprocess_uses_fixed_result_and_ignores_wfo_trade_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(research, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(research, "RESULTS_DIR", results_dir)
    results_dir.mkdir()
    pd.DataFrame({"net_return": [9.0]}).to_csv(results_dir / "eth_usdt_1h_oos_trades.csv", index=False)

    index = pd.date_range("2024-07-01", periods=3, freq="1h", tz="UTC")
    trade_returns = [0.10, -0.05]
    fixed_result = research.backtest.BacktestResult(
        equity=pd.Series([1.0, 1.10, 1.045], index=index, name="equity"),
        returns=pd.Series([0.0, 0.10, -0.05], index=index, name="returns"),
        trades=pd.DataFrame(
            {
                "entry_time": index[:2],
                "exit_time": index[1:],
                "net_return": trade_returns,
            }
        ),
        metrics={
            "total_return": 0.045,
            "CAGR": 0.02,
            "Sharpe": 0.5,
            "Sortino": 0.7,
            "max_drawdown": -0.05,
            "win_rate": 0.5,
            "profit_factor": 2.0,
            "trades": 2.0,
            "exposure": 0.5,
        },
        config=research.backtest.BacktestConfig(
            variant="mean_reversion",
            threshold=50.0,
            exit_bars=20,
        ),
        exposure=pd.Series([False, True, True], index=index),
    )
    final_config = {
        "dataset": "eth_usdt_1h",
        "symbol": "ETH/USDT",
        "timeframe": "1h",
        "length": 30,
        "smooth": 3,
        "variant": "mean_reversion",
        "threshold": 50.0,
        "exit_bars": 20,
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "tick_size": 1e-10,
    }

    output = research.postprocess_final_strategy(
        final_config,
        ResearchConfig(mc_permutations=7, seed=42),
        result=fixed_result,
    )

    expected_terminal = np.prod(1.0 + np.asarray(trade_returns)) - 1.0
    mc = output["monte_carlo"]
    assert mc["terminal_return"].nunique() == 1
    assert mc["terminal_return"].iloc[0] == pytest.approx(expected_terminal)
    assert mc["trades"].eq(len(trade_returns)).all()
    saved_trades = pd.read_csv(results_dir / "final_strategy_trades.csv")
    assert saved_trades["net_return"].tolist() == pytest.approx(trade_returns)
    saved_equity = pd.read_csv(results_dir / "final_strategy_equity.csv")
    assert list(saved_equity.columns) == ["timestamp", "dataset", "symbol", "timeframe", "returns", "equity"]
    assert saved_equity["equity"].tolist() == pytest.approx(fixed_result.equity.tolist())

    saved_config = json.loads((results_dir / "final_config.json").read_text(encoding="utf-8"))
    assert saved_config["fixed_evaluation"]["trade_count"] == 2
    assert saved_config["fixed_evaluation"]["metrics"]["total_return"] == pytest.approx(0.045)
    assert saved_config["fixed_evaluation"]["oos_start"] == index[0].isoformat()
    assert saved_config["monte_carlo"]["source"] == "results/final_strategy_trades.csv"
    assert saved_config["monte_carlo"]["source_column"] == "net_return"


def test_benchmark_uses_oos_timestamps_and_entry_exit_costs() -> None:
    df = _frame(5, freq="1D")
    df["open"] = [100, 110, 120, 130, 140]
    df["close"] = [105, 115, 125, 135, 150]
    timestamps = df.index[[1, 2, 4]]

    equity = buy_and_hold_benchmark(df, timestamps, cost_rate=0.0015)

    assert list(equity.index) == list(timestamps)
    units = (1.0 - 0.0015) / 110.0
    assert equity.iloc[0] == pytest.approx(units * 115.0)
    assert equity.iloc[-1] == pytest.approx(units * 150.0 * (1.0 - 0.0015))
    returns = equity_returns(equity)
    assert returns.iloc[0] == pytest.approx(equity.iloc[0] - 1.0)


def test_positive_strategy_below_buy_hold_is_no_edge() -> None:
    oos = {"total_return": 0.05, "Sharpe": 0.7, "trades": 30.0}
    benchmark = {"total_return": 0.12, "Sharpe": 0.8}

    assert has_credible_edge(oos, benchmark, min_trades=20) is False


def test_run_final_strategy_uses_full_context_and_common_oos_start(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame(460, freq="1D")
    final_config = {
        "dataset": "btc_usdt_1d",
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "length": 20,
        "smooth": 5,
        "variant": "mean_reversion",
        "threshold": 40.0,
        "exit_bars": 10,
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "tick_size": 1e-10,
    }
    expected_result = object()
    captured = {}

    def fake_compute_wek(received_frame: pd.DataFrame, **kwargs) -> pd.Series:
        assert received_frame is frame
        assert kwargs == {"length": 20, "smooth": 5, "tick_size": 1e-10}
        return pd.Series(0.0, index=frame.index, name="WEK")

    def fake_backtest(received_frame: pd.DataFrame, wek_series: pd.Series, **kwargs):
        captured["frame"] = received_frame
        captured["wek"] = wek_series
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(research, "compute_wek", fake_compute_wek)
    monkeypatch.setattr(research, "run_backtest_with_active_start", fake_backtest)

    result = research.run_final_strategy(final_config, ResearchConfig(), frame=frame)

    assert result is expected_result
    assert captured["frame"] is frame
    assert len(captured["wek"]) == len(frame)
    assert captured["active_start"] == frame.index[0] + pd.DateOffset(months=12)


def test_ablation_formulas_match_neutral_multiplier_intent() -> None:
    df = _frame(80)
    length = 14
    smooth = 3
    _, components = research.compute_wek(df, length=length, smooth=smooth, return_components=True)

    entropy_removed = construct_ablated_wek(df, length=length, smooth=smooth, remove_entropy=True)
    conviction_removed = construct_ablated_wek(df, length=length, smooth=smooth, remove_conviction=True)
    wick_removed = construct_ablated_wek(df, length=length, smooth=smooth, remove_wick=True)

    expected_entropy = research.pine_ema(components["wick_sig"] * (0.5 + components["conv_sig"]) * 100.0, smooth).clip(-100, 100)
    expected_conviction = research.pine_ema(components["wick_sig"] * (0.3 + components["event_factor"] * 1.4) * 100.0, smooth).clip(-100, 100)

    pd.testing.assert_series_equal(entropy_removed, expected_entropy.rename("WEK"))
    pd.testing.assert_series_equal(conviction_removed, expected_conviction.rename("WEK"))
    assert wick_removed.dropna().abs().max() == pytest.approx(0.0)


def test_stability_neighbor_rule_flags_isolated_peak() -> None:
    rows = []
    for length in [14, 20, 30]:
        rows.append(
            {
                "dataset": "btc_usdt_1d",
                "length": length,
                "smooth": 3,
                "variant": "mean_reversion",
                "threshold": 50.0,
                "exit_bars": 5,
                "mean_oos_Sharpe": 3.0 if length == 20 else -0.2,
            }
        )
    grid = pd.DataFrame(rows)

    diagnostic = stability_neighbor_diagnostic(grid, rows[1])

    assert diagnostic["neighbor_count"] == 2
    assert diagnostic["positive_sharpe_share"] == 0.0
    assert diagnostic["median_neighbor_sharpe"] == pytest.approx(-0.2)
    assert diagnostic["overfit"] is True


def test_final_config_prefers_powered_candidates_and_persists_cost_metadata() -> None:
    fixed = pd.DataFrame(
        [
            {
                "dataset": "btc_usdt_1h",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "fold": 0,
                "length": 14,
                "smooth": 3,
                "variant": "mean_reversion",
                "threshold": 50.0,
                "exit_bars": 5,
                "oos_Sharpe": 5.0,
                "oos_total_return": 0.2,
                "oos_trades": 2.0,
                "oos_max_drawdown": -0.01,
            },
            {
                "dataset": "btc_usdt_1h",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "fold": 0,
                "length": 20,
                "smooth": 3,
                "variant": "mean_reversion",
                "threshold": 50.0,
                "exit_bars": 5,
                "oos_Sharpe": 1.0,
                "oos_total_return": 0.1,
                "oos_trades": 21.0,
                "oos_max_drawdown": -0.03,
            },
        ]
    )

    selected = select_final_config(fixed, config=ResearchConfig.quick_config())

    assert selected["length"] == 20
    assert selected["underpowered_fallback"] is False
    assert selected["fee_rate"] == pytest.approx(0.001)
    assert selected["slippage_rate"] == pytest.approx(0.0005)
    assert selected["tick_size"] == pytest.approx(1e-10)


def test_report_contains_honesty_is_oos_and_benchmark_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research, "REPORT_PATH", tmp_path / "report.md")
    leaderboard = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "Sharpe": 0.5,
                "total_return": 0.05,
                "trades": 30.0,
                "benchmark_Sharpe": 0.7,
                "benchmark_total_return": 0.1,
                "credible_edge": False,
            }
        ]
    )
    result = research.DatasetResult(
        dataset="synthetic",
        symbol="BTC/USDT",
        timeframe="1d",
        rows=10,
        start="2024-01-01",
        end="2024-01-10",
        min_trades_required=8,
        fallback_folds=1,
        fold_count=1,
        power_warning="1d/no-signal underpower",
        credible_edge=False,
        wfo_metrics={},
        benchmark_metrics={},
    )

    final_config = {
        "fixed_evaluation": {
            "trade_count": 2,
            "trade_compound_return": 0.045,
            "metrics": {"total_return": 0.045},
        },
        "monte_carlo": {"source": "results/final_strategy_trades.csv"},
    }
    research.write_report(leaderboard, [result], final_config, pd.DataFrame(), pd.DataFrame())

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## IS" in text
    assert "## OOS" in text
    assert "Benchmark buy&hold" in text
    assert "NO EDGE" in text
    assert "look-ahead" in text
    assert "multiple" in text or "wielokrotnego" in text
    assert "fixed final candidate" in text
    assert "trade-order path risk" in text
    assert "2 transakcji" in text
    assert "4.5000%" in text


def test_small_synthetic_end_to_end_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(research, "CHARTS_DIR", tmp_path / "charts")
    monkeypatch.setattr(research, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(research, "PROJECT_ROOT", tmp_path)

    frame = _frame(500, freq="1D")

    def fake_load(symbol: str, timeframe: str, **kwargs) -> pd.DataFrame:
        return frame

    monkeypatch.setattr(research, "load_cached_data", fake_load)
    config = ResearchConfig(
        lengths=(14,),
        smooths=(3,),
        thresholds=(40.0,),
        exit_bars_options=(5,),
        variants=("mean_reversion",),
        symbols=("BTC/USDT",),
        timeframes=("1d",),
        mc_permutations=5,
        quick=True,
    )

    output = research.run_study(config)

    assert not output["leaderboard"].empty
    assert (tmp_path / "results" / "btc_usdt_1d_full_grid.csv").exists()
    assert (tmp_path / "results" / "aggregate_leaderboard.csv").exists()
    assert (tmp_path / "results" / "monte_carlo_1000_permutation.csv").exists()
    assert (tmp_path / "results" / "final_strategy_trades.csv").exists()
    assert (tmp_path / "results" / "final_strategy_equity.csv").exists()
    final_config = json.loads((tmp_path / "results" / "final_config.json").read_text(encoding="utf-8"))
    assert final_config["monte_carlo"]["source"] == "results/final_strategy_trades.csv"
    assert final_config["fixed_evaluation"]["trade_count"] == final_config["fixed_evaluation"]["metrics"]["trades"]
    assert (tmp_path / "report.md").exists()

    def fail_grid(*args, **kwargs):
        raise AssertionError("artifact refresh must not rerun grid or walk-forward")

    monkeypatch.setattr(research.backtest, "grid_search", fail_grid)
    monkeypatch.setattr(research.backtest, "walk_forward", fail_grid)
    refreshed = research.refresh_final_outputs_from_artifacts(config)
    assert refreshed["final_config"]["monte_carlo"]["source"] == "results/final_strategy_trades.csv"
