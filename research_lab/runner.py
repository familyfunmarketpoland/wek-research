from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research_lab.config import (
    EXPECTED_TOTAL_TRIALS,
    PRE_REGISTERED_CONFIG_PATH,
    load_preregistered_config,
    preregistered_config_fingerprint,
    validate_preregistered_config,
)
from research_lab.data_guard import (
    DATA_DIR,
    build_candidate_hash,
    load_holdout_once,
    load_manifest,
    load_research_dataset,
    manifest_fingerprint,
)
from research_lab.engine import infer_bars_per_year, run_signal_backtest, walk_forward_fixed_candidates
from research_lab.hypotheses import Candidate, enumerate_candidates, generate_signals
from research_lab.reporting import load_report_inputs, write_report
from research_lab.statistics import (
    DEFAULT_COST_RATE_PER_SIDE,
    DEFAULT_FAMILY_SIZE,
    DEFAULT_PERMUTATIONS,
    DEFAULT_PERMUTATION_SEED,
    ExecutionGroup,
    deflated_sharpe_ratio,
    family_sharpe_for_multiple_testing,
    familywise_permutation_result_from_sharpe,
    per_bar_sharpe,
)
try:
    from research_lab.statistics import signal_path_family_null_execution_groups
except ImportError:  # pragma: no cover - exercised only while the companion statistics patch is absent.
    signal_path_family_null_execution_groups = None  # type: ignore[assignment]


PREREG_COMMIT = "96bebdf"
EXPECTED_CONFIG_FINGERPRINT = "7e77ff5103e7b8e82a2d3ae3d16db0afebc9cfa3c105985ddf3fe35b1fd84990"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRNAME = "results2"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "report2.md"
FROZEN_WINNER_JSON = "frozen_winner.json"
FROZEN_WINNER_SHA256 = "frozen_winner.sha256"


def run_research_phase(
    *,
    data_dir: str | Path = DATA_DIR,
    output_dir: str | Path = RESULTS_DIRNAME,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    manifest_path: str | Path | None = None,
    config_path: str | Path = PRE_REGISTERED_CONFIG_PATH,
) -> dict[str, Any]:
    """Run the preregistered research phase and write confirmatory artifacts."""

    config = load_preregistered_config(Path(config_path))
    validate_preregistered_config(config)
    config_fp = preregistered_config_fingerprint(config)
    if config_fp != EXPECTED_CONFIG_FINGERPRINT:
        raise RuntimeError(
            "pre-registered config fingerprint mismatch: "
            f"expected {EXPECTED_CONFIG_FINGERPRINT}, got {config_fp}"
        )
    manifest = load_manifest(data_dir=data_dir, manifest_path=manifest_path)
    manifest_fp = manifest_fingerprint(manifest)
    candidates = enumerate_candidates(config)
    if len(candidates) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError(f"candidate universe drifted from {EXPECTED_TOTAL_TRIALS}")

    frames = _load_research_frames(candidates, data_dir=data_dir, manifest_path=manifest_path)
    candidate_rows, fold_rows, trades, candidate_returns, execution_groups = _evaluate_research_candidates(
        candidates,
        frames,
        config=config,
    )
    family_sharpes = np.array(
        [family_sharpe_for_multiple_testing(candidate_returns[c.candidate_id]) for c in candidates],
        dtype=float,
    )
    null_sharpes = _compute_family_null(execution_groups)

    enriched_rows = _apply_confirmatory_rules(
        candidates=candidates,
        rows=candidate_rows,
        returns_by_candidate=candidate_returns,
        family_sharpes=family_sharpes,
        null_sharpes=null_sharpes,
    )
    candidate_frame = pd.DataFrame(enriched_rows)
    hypothesis_summary = _hypothesis_summary(candidate_frame)
    decision = _study_decision(
        candidate_frame,
        config_fingerprint=config_fp,
        manifest_fingerprint=manifest_fp,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(output / "candidates.csv", index=False)
    hypothesis_summary.to_csv(output / "hypothesis_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    pd.DataFrame({"permutation": np.arange(DEFAULT_PERMUTATIONS), "best_sharpe": null_sharpes}).to_csv(
        output / "permutation_best_sharpe.csv",
        index=False,
    )
    _write_json(output / "study_decision.json", decision)

    frozen_path = output / FROZEN_WINNER_JSON
    frozen_sha_path = output / FROZEN_WINNER_SHA256
    if decision["decision"] == "WINNER_FROZEN":
        frozen = _frozen_winner_payload(decision["winner"])
        frozen["config_fingerprint"] = config_fp
        frozen["manifest_fingerprint"] = manifest_fp
        frozen["prereg_commit"] = PREREG_COMMIT
        digest = build_candidate_hash(frozen)
        _write_json(frozen_path, frozen)
        frozen_sha_path.write_text(digest + "\n", encoding="utf-8")
        decision["frozen_winner_sha256"] = digest
        _write_json(output / "study_decision.json", decision)
    else:
        if frozen_path.exists():
            frozen_path.unlink()
        if frozen_sha_path.exists():
            frozen_sha_path.unlink()

    write_report(
        output_dir=output,
        report_path=report_path,
        decision=decision,
        hypothesis_summary=hypothesis_summary,
        candidates=candidate_frame,
    )
    return decision


def run_holdout_phase(
    *,
    data_dir: str | Path = DATA_DIR,
    output_dir: str | Path = RESULTS_DIRNAME,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    manifest_path: str | Path | None = None,
    config_path: str | Path = PRE_REGISTERED_CONFIG_PATH,
) -> dict[str, Any]:
    """Run the one-shot holdout phase for the frozen research winner."""

    output = Path(output_dir)
    frozen_path = output / FROZEN_WINNER_JSON
    frozen_sha_path = output / FROZEN_WINNER_SHA256
    if not frozen_path.exists() or not frozen_sha_path.exists():
        raise RuntimeError("holdout refused: no frozen winner artifact")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_sha = frozen_sha_path.read_text(encoding="utf-8").strip()
    if build_candidate_hash(frozen) != frozen_sha:
        raise RuntimeError("holdout refused: frozen winner hash mismatch")

    config = load_preregistered_config(Path(config_path))
    config_fp = preregistered_config_fingerprint(config)
    if config_fp != EXPECTED_CONFIG_FINGERPRINT:
        raise RuntimeError("holdout refused: config fingerprint mismatch")
    manifest = load_manifest(data_dir=data_dir, manifest_path=manifest_path)
    manifest_fp = manifest_fingerprint(manifest)
    if frozen.get("config_fingerprint") != config_fp:
        raise RuntimeError("holdout refused: config fingerprint mismatch")
    if frozen.get("manifest_fingerprint") != manifest_fp:
        raise RuntimeError("holdout refused: manifest fingerprint mismatch")

    candidate = _candidate_from_frozen(frozen)
    research = load_research_dataset(
        candidate.symbol,
        candidate.timeframe,
        data_dir=data_dir,
        manifest_path=manifest_path,
    )
    holdout = load_holdout_once(
        candidate.symbol,
        candidate.timeframe,
        frozen_candidate_hash=frozen_sha,
        data_dir=data_dir,
        manifest_path=manifest_path,
        candidate=frozen,
    )
    if holdout.empty:
        raise RuntimeError("holdout refused: holdout dataset is empty after claim")

    combined = pd.concat([research, holdout]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    active_start = holdout.index[0]
    signals = generate_signals(combined, candidate)
    result = run_signal_backtest(
        combined,
        signals,
        active_start=active_start,
        cost_rate_per_side=float(config["costs"]["cost_rate_per_side"]),  # type: ignore[index]
        bars_per_year=infer_bars_per_year(holdout.index),
    )
    benchmark = _buy_and_hold_metrics(holdout, cost_rate_per_side=float(config["costs"]["cost_rate_per_side"]))  # type: ignore[index]
    total_return = _finite_or_none(result.metrics.get("total_return"))
    final_negative = bool(total_return is not None and total_return < 0.0)
    holdout_result = {
        "phase": "holdout",
        "candidate_id": candidate.candidate_id,
        "candidate_index": int(frozen["candidate_index"]),
        "dataset": candidate.dataset,
        "holdout_start": _timestamp(active_start),
        "holdout_end": _timestamp(holdout.index[-1]),
        "bars": int(len(holdout)),
        "trades": int(result.metrics.get("trades", 0) or 0),
        "total_return": total_return,
        "sharpe": _finite_or_none(result.metrics.get("Sharpe")),
        "benchmark_total_return": benchmark["total_return"],
        "benchmark_sharpe": benchmark["annualized_sharpe"],
        "benchmark_observed_sharpe": benchmark["observed_sharpe"],
        "final_negative": final_negative,
        "final_status": "REJECTED_FINAL_NEGATIVE" if final_negative else "FINAL_EVALUATED",
    }
    _write_json(output / "holdout_result.json", holdout_result)
    decision, hypotheses, candidates = load_report_inputs(output)
    write_report(
        output_dir=output,
        report_path=report_path,
        decision=decision,
        hypothesis_summary=hypotheses,
        candidates=candidates,
        holdout_result=holdout_result,
    )
    return holdout_result


def _load_research_frames(
    candidates: Sequence[Candidate],
    *,
    data_dir: str | Path,
    manifest_path: str | Path | None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        if candidate.dataset in frames:
            continue
        frames[candidate.dataset] = load_research_dataset(
            candidate.symbol,
            candidate.timeframe,
            data_dir=data_dir,
            manifest_path=manifest_path,
        )
    return frames


def _evaluate_research_candidates(
    candidates: Sequence[Candidate],
    frames: Mapping[str, pd.DataFrame],
    *,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, dict[str, np.ndarray], list[ExecutionGroup]]:
    candidate_index = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    trade_parts: list[pd.DataFrame] = []
    returns_by_candidate: dict[str, np.ndarray] = {}
    groups: list[ExecutionGroup] = []

    for dataset, frame in frames.items():
        dataset_candidates = [candidate for candidate in candidates if candidate.dataset == dataset]
        fixed = walk_forward_fixed_candidates(frame, dataset_candidates, config=config)
        for row in fixed.candidate_results.to_dict("records"):
            cid = str(row["candidate_id"])
            candidate = candidates[candidate_index[cid]]
            returns = fixed.returns.get(cid, pd.Series(dtype=float, name="returns"))
            benchmark = _buy_and_hold_metrics(
                frame.loc[returns.index] if len(returns) else frame.iloc[0:0],
                cost_rate_per_side=float(config["costs"]["cost_rate_per_side"]),  # type: ignore[index]
            )
            row.update(
                {
                    "candidate_index": candidate_index[cid],
                    "dataset": dataset,
                    "fold_count": int((fixed.folds["candidate_id"] == cid).sum()) if "candidate_id" in fixed.folds else 0,
                    "observed_sharpe": per_bar_sharpe(returns.to_numpy(dtype=float)),
                    "benchmark_total_return": benchmark["total_return"],
                    "benchmark_observed_sharpe": benchmark["observed_sharpe"],
                    "benchmark_Sharpe": benchmark["annualized_sharpe"],
                }
            )
            rows.append(row)
            returns_by_candidate[cid] = returns.to_numpy(dtype=float)

        groups.append(_execution_group_for_dataset(frame, fixed.fold_positions, dataset_candidates, candidate_index))
        if not fixed.folds.empty:
            fold_frame = fixed.folds.copy()
            fold_frame["candidate_index"] = fold_frame["candidate_id"].map(candidate_index)
            fold_frame["dataset"] = dataset
            folds.extend(fold_frame.to_dict("records"))
        if not fixed.trades.empty:
            trades = fixed.trades.copy()
            trades["candidate_index"] = trades["candidate_id"].map(candidate_index)
            trades["dataset"] = dataset
            trade_parts.append(trades)

    ordered_rows = sorted(rows, key=lambda row: int(row["candidate_index"]))
    if len(ordered_rows) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError("research evaluation did not produce all 186 candidates")
    observed_group_candidates = sum(len(np.asarray(group.candidate_ids)) for group in groups)
    if observed_group_candidates != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError("execution groups did not cover all 186 candidates")
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    return ordered_rows, folds, trades, returns_by_candidate, groups


def _compute_family_null(execution_groups: Sequence[ExecutionGroup]) -> np.ndarray:
    if signal_path_family_null_execution_groups is None:
        raise RuntimeError("statistics.signal_path_family_null_execution_groups is required")
    return np.asarray(
        signal_path_family_null_execution_groups(
            execution_groups,
            permutations=DEFAULT_PERMUTATIONS,
            seed=DEFAULT_PERMUTATION_SEED,
            chunk_size=16,
            family_size=DEFAULT_FAMILY_SIZE,
            cost=DEFAULT_COST_RATE_PER_SIDE,
        ),
        dtype=float,
    )


def _execution_group_for_dataset(
    frame: pd.DataFrame,
    fold_positions: pd.DataFrame,
    candidates: Sequence[Candidate],
    candidate_index: Mapping[str, int],
) -> ExecutionGroup:
    if fold_positions.empty:
        return ExecutionGroup(
            opens=np.array([], dtype=float),
            closes=np.array([], dtype=float),
            fold_ids=np.array([], dtype=int),
            candidate_ids=np.array([candidate_index[c.candidate_id] for c in candidates], dtype=int),
            positions=np.zeros((len(candidates), 0), dtype=float),
        )
    timestamps = pd.Index(sorted(fold_positions.index.unique()))
    active = frame.loc[timestamps]
    fold_ids = (
        fold_positions.reset_index()
        .drop_duplicates(subset=["timestamp", "fold"])
        .set_index("timestamp")
        .loc[timestamps, "fold"]
        .to_numpy(dtype=int)
    )
    matrix = np.zeros((len(candidates), len(timestamps)), dtype=float)
    for row_index, candidate in enumerate(candidates):
        positions = _positions_for_candidate(fold_positions, candidate.candidate_id)
        if positions.empty:
            continue
        matrix[row_index] = positions.reindex(timestamps)["position"].fillna(0).to_numpy(dtype=float)
    return ExecutionGroup(
        opens=active["open"].astype(float).to_numpy(dtype=float),
        closes=active["close"].astype(float).to_numpy(dtype=float),
        fold_ids=fold_ids,
        candidate_ids=np.array([candidate_index[c.candidate_id] for c in candidates], dtype=int),
        positions=matrix,
    )


def _apply_confirmatory_rules(
    *,
    candidates: Sequence[Candidate],
    rows: Sequence[Mapping[str, Any]],
    returns_by_candidate: Mapping[str, np.ndarray],
    family_sharpes: np.ndarray,
    null_sharpes: np.ndarray,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row, candidate in zip(rows, candidates, strict=True):
        returns = returns_by_candidate[candidate.candidate_id]
        dsr = deflated_sharpe_ratio(returns, family_sharpes)
        permutation = familywise_permutation_result_from_sharpe(float(row["observed_sharpe"]), null_sharpes)
        total_return = _as_float(row.get("total_return"))
        trades = _as_float(row.get("trades"))
        benchmark_return = _as_float(row.get("benchmark_total_return"))
        benchmark_sharpe = _as_float(row.get("benchmark_observed_sharpe"))
        observed_sharpe = _as_float(row.get("observed_sharpe"))
        checks = {
            "positive_net_oos_return": _gt(total_return, 0.0),
            "minimum_oos_trades": _gte(trades, 30.0),
            "beats_buy_hold_return": _gt(total_return, benchmark_return),
            "beats_buy_hold_sharpe": _gt(observed_sharpe, benchmark_sharpe),
            "dsr_pass": bool(dsr.passes),
            "permutation_q95_pass": bool(math.isfinite(permutation.observed_sharpe) and permutation.observed_sharpe > permutation.q95),
            "permutation_p_pass": bool(math.isfinite(permutation.empirical_p) and permutation.empirical_p <= 0.05),
        }
        failure_reasons = [name for name, passed in checks.items() if not passed]
        enriched.append(
            {
                **dict(row),
                "dsr": dsr.dsr,
                "dsr_sr0": dsr.sr0,
                "dsr_sample_size": dsr.sample_size,
                "dsr_sample_skew": dsr.sample_skew,
                "dsr_pearson_kurtosis": dsr.pearson_kurtosis,
                "dsr_sigma_sr": dsr.sigma_sr,
                "permutation_q95": permutation.q95,
                "permutation_empirical_p": permutation.empirical_p,
                **checks,
                "passes_all": all(checks.values()),
                "failure_reasons": ";".join(failure_reasons),
            }
        )
    return enriched


def _hypothesis_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hypothesis_id, group in candidates.groupby("hypothesis_id", sort=True):
        ranked = _rank_candidates(group)
        nearest = ranked.iloc[0] if not ranked.empty else {}
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "candidate_count": int(len(group)),
                "eligible_candidates": int(group["minimum_oos_trades"].sum()),
                "passing_candidates": int(group["passes_all"].sum()),
                "best_dsr": _finite_or_none(pd.to_numeric(group["dsr"], errors="coerce").max()),
                "best_empirical_p": _finite_or_none(pd.to_numeric(group["permutation_empirical_p"], errors="coerce").min()),
                "best_total_return": _finite_or_none(pd.to_numeric(group["total_return"], errors="coerce").max()),
                "nearest_candidate_id": nearest.get("candidate_id") if len(ranked) else None,
            }
        )
    return pd.DataFrame(rows)


def _study_decision(
    candidates: pd.DataFrame,
    *,
    config_fingerprint: str,
    manifest_fingerprint: str,
) -> dict[str, Any]:
    ranked = _rank_candidates(candidates)
    passing = ranked.loc[ranked["passes_all"] == True]  # noqa: E712
    nearest = ranked.iloc[0].to_dict() if not ranked.empty else None
    winner = passing.iloc[0].to_dict() if not passing.empty else None
    return {
        "phase": "research",
        "decision": "WINNER_FROZEN" if winner else "NO_EDGE",
        "holdout_allowed": bool(winner),
        "candidate_count": EXPECTED_TOTAL_TRIALS,
        "prereg_commit": PREREG_COMMIT,
        "config_fingerprint": config_fingerprint,
        "manifest_fingerprint": manifest_fingerprint,
        "winner": _json_ready(winner) if winner else None,
        "nearest_candidate": _json_ready(nearest) if nearest else None,
    }


def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ranked = candidates.copy()
    ranked["_rank_dsr"] = pd.to_numeric(ranked["dsr"], errors="coerce").fillna(-math.inf)
    ranked["_rank_p"] = pd.to_numeric(ranked["permutation_empirical_p"], errors="coerce").fillna(math.inf)
    tie_breakers = [
        "hypothesis_id",
        "symbol",
        "timeframe",
        "side",
        "session_utc",
        "mode",
        "hold_bars",
        "streak_length",
        "candidate_index",
    ]
    for key in tie_breakers:
        if key not in ranked:
            ranked[key] = ""
    return ranked.sort_values(
        by=["_rank_dsr", "_rank_p", *tie_breakers],
        ascending=[False, True, *([True] * len(tie_breakers))],
        kind="mergesort",
    ).drop(columns=["_rank_dsr", "_rank_p"])


def _frozen_winner_payload(winner: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "candidate_index",
        "candidate_id",
        "hypothesis_id",
        "symbol",
        "timeframe",
        "dataset",
        "side",
        "session_utc",
        "mode",
        "hold_bars",
        "streak_length",
        "dsr",
        "permutation_empirical_p",
        "observed_sharpe",
        "total_return",
        "trades",
    ]
    return {field: _json_ready(winner.get(field)) for field in fields}


def _candidate_from_frozen(frozen: Mapping[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(frozen["candidate_id"]),
        hypothesis_id=str(frozen["hypothesis_id"]),
        symbol=str(frozen["symbol"]),
        timeframe=str(frozen["timeframe"]),
        side=_optional_str(frozen.get("side")),
        session_utc=_optional_str(frozen.get("session_utc")),
        mode=_optional_str(frozen.get("mode")),
        hold_bars=_optional_int(frozen.get("hold_bars")),
        streak_length=_optional_int(frozen.get("streak_length")),
    )


def _positions_for_candidate(fold_positions: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    if fold_positions.empty or "candidate_id" not in fold_positions:
        return pd.DataFrame(columns=["fold", "candidate_id", "position"])
    positions = fold_positions.loc[fold_positions["candidate_id"] == candidate_id].copy()
    return positions.sort_index()


def _buy_and_hold_metrics(frame: pd.DataFrame, *, cost_rate_per_side: float) -> dict[str, float | None]:
    if frame.empty:
        return {"total_return": None, "observed_sharpe": None, "annualized_sharpe": None}
    returns = pd.Series(0.0, index=frame.index, dtype=float, name="returns")
    opens = frame["open"].astype(float)
    closes = frame["close"].astype(float)
    returns.iloc[0] = (1.0 - cost_rate_per_side) * (closes.iloc[0] / opens.iloc[0]) - 1.0
    if len(frame) > 1:
        returns.iloc[1:] = closes.iloc[1:].to_numpy() / closes.iloc[:-1].to_numpy() - 1.0
    returns.iloc[-1] = (1.0 + returns.iloc[-1]) * (1.0 - cost_rate_per_side) - 1.0
    equity = (1.0 + returns).cumprod()
    observed_sharpe = per_bar_sharpe(returns.to_numpy(dtype=float))
    bars_per_year = infer_bars_per_year(frame.index)
    return_std = float(returns.std(ddof=0))
    annualized_sharpe = (
        float(returns.mean() / return_std * math.sqrt(bars_per_year))
        if math.isfinite(return_std) and return_std > 0.0
        else None
    )
    return {
        "total_return": _finite_or_none(float(equity.iloc[-1] - 1.0)),
        "observed_sharpe": _finite_or_none(observed_sharpe),
        "annualized_sharpe": _finite_or_none(annualized_sharpe),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return _timestamp(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _timestamp(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_or_none(value: Any) -> float | None:
    number = _as_float(value)
    return number if math.isfinite(number) else None


def _gt(left: float, right: float) -> bool:
    return bool(math.isfinite(left) and math.isfinite(right) and left > right)


def _gte(left: float, right: float) -> bool:
    return bool(math.isfinite(left) and left >= right)


def _optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text and text != "nan" else None


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)
