from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab import runner
from research_lab.config import EXPECTED_TOTAL_TRIALS
from research_lab.data_guard import FINAL_CLAIM_FILENAME, administrative_split, build_candidate_hash
from research_lab.hypotheses import enumerate_candidates
from research_lab.statistics import DeflatedSharpeResult, ExecutionGroup


SYMBOL_DATASETS = (
    "btc_usdt_1h",
    "btc_usdt_4h",
    "eth_usdt_1h",
    "eth_usdt_4h",
    "sol_usdt_1h",
    "sol_usdt_4h",
)


def _ohlcv_frame(*, periods: int = 760, falling_tail: bool = False) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=periods, freq="1D", tz="UTC", name="timestamp")
    price = np.full(periods, 100.0, dtype=float)
    close = price.copy()
    if falling_tail:
        price[-200:] = np.linspace(100.0, 60.0, 200)
        close[-200:] = price[-200:] * 0.97
    return pd.DataFrame(
        {
            "open": price,
            "high": np.maximum(price, close) + 1.0,
            "low": np.minimum(price, close) - 1.0,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=index,
    )


def _guarded_data_dir(tmp_path: Path, *, falling_tail: bool = False) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for dataset in SYMBOL_DATASETS:
        _ohlcv_frame(falling_tail=falling_tail).to_parquet(data_dir / f"{dataset}.parquet")
    plans = administrative_split(data_dir=data_dir)
    assert len(plans) == len(SYMBOL_DATASETS)
    return data_dir


def _fake_group_null(
    groups: list[ExecutionGroup],
    *,
    permutations: int = 500,
    seed: int = 42,
    chunk_size: int = 16,
    family_size: int = EXPECTED_TOTAL_TRIALS,
    cost: float = 0.0015,
) -> np.ndarray:
    assert permutations == 500
    assert seed == 42
    assert chunk_size == 16
    assert family_size == EXPECTED_TOTAL_TRIALS
    assert cost == 0.0015
    assert sum(len(np.asarray(group.candidate_ids)) for group in groups) == EXPECTED_TOTAL_TRIALS
    return np.zeros(500, dtype=float)


def test_research_phase_no_edge_writes_artifacts_and_never_uses_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _guarded_data_dir(tmp_path)
    output_dir = tmp_path / "results2"
    report_path = tmp_path / "report2.md"
    monkeypatch.setattr(runner, "signal_path_family_null_execution_groups", _fake_group_null)

    decision = runner.run_research_phase(data_dir=data_dir, output_dir=output_dir, report_path=report_path)

    assert decision["decision"] == "NO_EDGE"
    assert decision["holdout_allowed"] is False
    expected_artifacts = {
        "candidates.csv",
        "hypothesis_summary.csv",
        "folds.csv",
        "trades.csv",
        "permutation_best_sharpe.csv",
        "study_decision.json",
        "report2.md",
    }
    assert expected_artifacts.issubset({path.name for path in output_dir.iterdir()})
    assert not (output_dir / runner.FROZEN_WINNER_JSON).exists()
    assert not (data_dir / "holdout" / ".claims" / FINAL_CLAIM_FILENAME).exists()
    assert report_path.exists()
    assert (output_dir / "report2.md").exists()

    candidates = pd.read_csv(output_dir / "candidates.csv")
    hypotheses = pd.read_csv(output_dir / "hypothesis_summary.csv")
    permutations = pd.read_csv(output_dir / "permutation_best_sharpe.csv")
    report = report_path.read_text(encoding="utf-8")

    assert len(candidates) == EXPECTED_TOTAL_TRIALS
    assert candidates["candidate_index"].tolist() == list(range(EXPECTED_TOTAL_TRIALS))
    assert hypotheses["hypothesis_id"].tolist() == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert len(permutations) == 500
    assert "NO_EDGE" in report
    assert "| Hipoteza | Verdict | candidate_id | symbol | timeframe | params | OOS total return | annualized Sharpe | DSR | familywise permutation p | trades |" in report
    assert "| H1 | UNDERPOWERED |" in report
    assert "H1 caveat: sygnal wolumenowy korzysta z proxy wolumenu dostepnego w OHLCV" in report
    assert "## Zamrozone uzasadnienia ekonomiczne" in report
    assert "H2 — Realized-Volatility Compression Donchian Breakout" in report
    assert "annualized benchmark Sharpe" in report
    assert "bez adaptacyjnego wyboru parametrow" in report
    assert "observed Sharpe > q95" in report
    assert "- positive_net_oos_return" in report
    assert "\n- p\n" not in report
    assert "Holdout nie zostal odczytany" in report
    assert "zapieczetowany 6-miesieczny zbior finalny" in report


def test_winner_freeze_and_one_shot_holdout_negative_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _guarded_data_dir(tmp_path, falling_tail=True)
    output_dir = tmp_path / "results2"
    report_path = tmp_path / "report2.md"
    candidates = enumerate_candidates()
    winner = next(
        candidate
        for candidate in candidates
        if candidate.hypothesis_id == "H3"
        and candidate.symbol == "BTC/USDT"
        and candidate.timeframe == "1h"
        and candidate.session_utc == "Asia"
        and candidate.side == "long"
    )

    def fake_evaluate(received_candidates, frames, *, config):
        rows = []
        returns_by_candidate = {}
        for index, candidate in enumerate(received_candidates):
            is_winner = candidate.candidate_id == winner.candidate_id
            returns = np.array([0.0, 0.02, 0.01, -0.002, 0.015]) if is_winner else np.zeros(5)
            returns_by_candidate[candidate.candidate_id] = returns
            rows.append(
                {
                    **candidate.as_dict(),
                    "candidate_index": index,
                    "dataset": candidate.dataset,
                    "total_return": 0.5 if is_winner else -0.01,
                    "CAGR": 0.1 if is_winner else -0.01,
                    "Sharpe": 2.0 if is_winner else 0.0,
                    "Sortino": 2.0 if is_winner else 0.0,
                    "max_drawdown": -0.01,
                    "win_rate": 0.7 if is_winner else np.nan,
                    "profit_factor": 3.0 if is_winner else np.nan,
                    "trades": 31.0 if is_winner else 0.0,
                    "exposure": 0.5 if is_winner else 0.0,
                    "fold_count": 2,
                    "observed_sharpe": 2.0 if is_winner else 0.0,
                    "benchmark_total_return": 0.0,
                    "benchmark_observed_sharpe": 0.0,
                    "benchmark_Sharpe": 0.0,
                }
            )
        group = ExecutionGroup(
            opens=np.array([100.0, 101.0]),
            closes=np.array([101.0, 102.0]),
            fold_ids=np.array([0, 0], dtype=int),
            candidate_ids=np.arange(EXPECTED_TOTAL_TRIALS),
            positions=np.zeros((EXPECTED_TOTAL_TRIALS, 2)),
        )
        return rows, [], pd.DataFrame(), returns_by_candidate, [group]

    def fake_dsr(returns, observed_family_sharpes):
        passes = bool(np.asarray(returns).max() > 0.01)
        return DeflatedSharpeResult(
            sharpe=2.0 if passes else 0.0,
            sr0=0.0,
            dsr=0.99 if passes else 0.0,
            passes=passes,
            sample_size=len(returns),
            sample_skew=0.0,
            pearson_kurtosis=3.0,
            sigma_sr=0.1,
        )

    monkeypatch.setattr(runner, "_evaluate_research_candidates", fake_evaluate)
    monkeypatch.setattr(runner, "signal_path_family_null_execution_groups", _fake_group_null)
    monkeypatch.setattr(runner, "deflated_sharpe_ratio", fake_dsr)

    decision = runner.run_research_phase(data_dir=data_dir, output_dir=output_dir, report_path=report_path)

    assert decision["decision"] == "WINNER_FROZEN"
    frozen = json.loads((output_dir / runner.FROZEN_WINNER_JSON).read_text(encoding="utf-8"))
    frozen_sha = (output_dir / runner.FROZEN_WINNER_SHA256).read_text(encoding="utf-8").strip()
    assert frozen["candidate_id"] == winner.candidate_id
    assert build_candidate_hash(frozen) == frozen_sha

    monkeypatch.setattr(
        runner,
        "run_signal_backtest",
        lambda *args, **kwargs: SimpleNamespace(metrics={"total_return": -0.25, "Sharpe": -1.2, "trades": 1}),
    )

    holdout = runner.run_holdout_phase(data_dir=data_dir, output_dir=output_dir, report_path=report_path)

    assert holdout["candidate_id"] == winner.candidate_id
    assert holdout["final_status"] == "REJECTED_FINAL_NEGATIVE"
    assert holdout["final_negative"] is True
    claim = data_dir / "holdout" / ".claims" / FINAL_CLAIM_FILENAME
    assert claim.exists()
    with pytest.raises(RuntimeError, match="holdout refused|already claimed"):
        runner.run_holdout_phase(data_dir=data_dir, output_dir=output_dir, report_path=report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "REJECTED_FINAL_NEGATIVE" in report
    assert "| H3 | PASS |" in report
    assert winner.candidate_id in report


def test_holdout_refuses_without_frozen_winner(tmp_path: Path) -> None:
    data_dir = _guarded_data_dir(tmp_path)

    with pytest.raises(RuntimeError, match="no frozen winner"):
        runner.run_holdout_phase(data_dir=data_dir, output_dir=tmp_path / "results2")

    assert not (data_dir / "holdout" / ".claims" / FINAL_CLAIM_FILENAME).exists()


def test_buy_and_hold_reports_annualized_and_per_bar_sharpe_separately() -> None:
    frame = _ohlcv_frame(periods=365)
    frame["close"] = np.linspace(100.2, 130.0, len(frame))
    frame["open"] = frame["close"].shift(fill_value=100.0)
    frame["high"] = frame[["open", "close"]].max(axis=1) + 1.0
    frame["low"] = frame[["open", "close"]].min(axis=1) - 1.0

    metrics = runner._buy_and_hold_metrics(frame, cost_rate_per_side=0.0015)

    assert metrics["observed_sharpe"] is not None
    assert metrics["annualized_sharpe"] is not None
    assert metrics["annualized_sharpe"] == pytest.approx(
        metrics["observed_sharpe"] * np.sqrt(len(frame) / (len(frame) - 1)) * np.sqrt(365.0)
    )


def test_confirmatory_cli_help_and_forbidden_static_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    help_result = subprocess.run(
        [sys.executable, str(root / "run_confirmatory.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert help_result.returncode == 0
    assert "research" in help_result.stdout
    assert "holdout" in help_result.stdout
    assert "--report-path" in help_result.stdout
    for relative in ("research_lab/runner.py", "research_lab/reporting.py", "run_confirmatory.py"):
        assert "read_parquet" not in (root / relative).read_text(encoding="utf-8")
