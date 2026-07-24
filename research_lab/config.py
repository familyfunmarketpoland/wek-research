from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTERED_CONFIG_PATH = PROJECT_ROOT / "configs" / "pre_registered.json"
EXPECTED_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
EXPECTED_TIMEFRAMES = ("1h", "4h")
EXPECTED_TRIAL_COUNTS = {
    "H1": 18,
    "H2": 36,
    "H3": 12,
    "H4": 48,
    "H5": 36,
    "H6": 36,
}
EXPECTED_TOTAL_TRIALS = 186


class ConfigValidationError(ValueError):
    """Raised when the pre-registered study configuration drifts from the frozen design."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _to_plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def preregistered_config_fingerprint(config: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _product_of_axis_lengths(axes: Mapping[str, Any], keys: tuple[str, ...] | list[str] | None = None) -> int:
    selected = keys if keys is not None else tuple(axes.keys())
    total = 1
    for key in selected:
        values = axes[key]
        if not isinstance(values, list) or not values:
            raise ConfigValidationError(f"Axis {key!r} must be a non-empty list.")
        total *= len(values)
    return total


def _validate_hypothesis(
    hypothesis: Mapping[str, Any],
    symbols: list[str],
    all_timeframes: list[str],
) -> int:
    hypothesis_id = hypothesis["id"]
    applicability = hypothesis["applicability"]
    applicable_timeframes = applicability["timeframes"]
    timeframes_na = applicability["timeframes_na"]
    execution_axes = hypothesis["parameter_grid"]["execution_axes"]
    model_combo_axes = hypothesis["parameter_grid"]["model_combo_axes"]

    if sorted(applicable_timeframes + timeframes_na) != sorted(all_timeframes):
        raise ConfigValidationError(f"{hypothesis_id} must account for every study timeframe exactly once.")

    if set(applicable_timeframes).intersection(timeframes_na):
        raise ConfigValidationError(f"{hypothesis_id} has overlapping applicable and N/A timeframes.")

    model_combo_count = _product_of_axis_lengths(execution_axes, model_combo_axes)
    if model_combo_count > 12:
        raise ConfigValidationError(
            f"{hypothesis_id} exceeds the 12-combo cap with {model_combo_count} model combinations."
        )

    expected_trials = hypothesis["expected_trial_count"]
    derived_trials = len(symbols) * len(applicable_timeframes) * _product_of_axis_lengths(execution_axes)
    if derived_trials != expected_trials:
        raise ConfigValidationError(
            f"{hypothesis_id} expected {expected_trials} trials but derives to {derived_trials}."
        )

    if EXPECTED_TRIAL_COUNTS[hypothesis_id] != expected_trials:
        raise ConfigValidationError(
            f"{hypothesis_id} trial count drifted from frozen design: {expected_trials}."
        )

    return derived_trials


def validate_preregistered_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = _to_plain(config)
    if plain["study"]["packet_reference"] != "P2-05-PREREG-CONFIG":
        raise ConfigValidationError("Unexpected packet reference.")

    symbols = plain["universe"]["symbols"]
    if symbols != list(EXPECTED_SYMBOLS):
        raise ConfigValidationError(f"Unexpected symbols: {symbols!r}")

    timeframes = plain["universe"]["timeframes"]
    if timeframes != list(EXPECTED_TIMEFRAMES):
        raise ConfigValidationError(f"Unexpected timeframes: {timeframes!r}")

    costs = plain["costs"]
    if costs["fee_rate_per_side"] != 0.001 or costs["slippage_rate_per_side"] != 0.0005:
        raise ConfigValidationError("Entry and exit costs must remain frozen at 10 bps fee and 5 bps slippage.")
    if costs["cost_rate_per_side"] != 0.0015:
        raise ConfigValidationError("Cost rate per side must equal fee plus slippage.")

    windows = plain["rolling_windows"]
    if windows["rolling_year_bars"] != {"1h": 8760, "4h": 2190}:
        raise ConfigValidationError("Rolling-year bar counts must remain frozen at 8760/2190.")
    if not windows["causal_baselines_exclude_current_bar"]:
        raise ConfigValidationError("Rolling baselines must exclude the current bar.")

    walk_forward = plain["walk_forward"]
    if (
        walk_forward["train_months"],
        walk_forward["oos_months"],
        walk_forward["step_months"],
    ) != (12, 3, 3):
        raise ConfigValidationError("Walk-forward windows must remain 12m/3m/3m.")
    if not walk_forward["require_complete_folds"]:
        raise ConfigValidationError("Walk-forward must require complete folds.")
    if walk_forward["adaptive_parameter_selection"]:
        raise ConfigValidationError("Adaptive parameter selection is forbidden.")
    if walk_forward["scoring_scope"] != "stitched_oos":
        raise ConfigValidationError("Candidates must be scored on stitched out-of-sample only.")

    eligibility = plain["eligibility"]
    if eligibility["minimum_oos_trades"] != 30:
        raise ConfigValidationError("Minimum out-of-sample trade count must be 30.")

    multiple_testing = plain["multiple_testing"]
    if multiple_testing["candidate_count"] != EXPECTED_TOTAL_TRIALS:
        raise ConfigValidationError("Candidate count must remain frozen at 186.")
    dsr = multiple_testing["deflated_sharpe_ratio"]
    if (
        not dsr["enabled"]
        or dsr["family_size_n"] != EXPECTED_TOTAL_TRIALS
        or not dsr["positive_oos_sharpe_only"]
        or dsr["threshold"] != 0.95
    ):
        raise ConfigValidationError("DSR metadata drifted from the frozen design.")
    formula_metadata = dsr["formula_metadata"]
    if (
        formula_metadata["sharpe_definition"]
        != "SR* is the observed per-bar nonannualized out-of-sample Sharpe ratio for a stitched candidate path."
        or formula_metadata["dsr_formula"]
        != "DSR = Phi(((SR* - SR0) * sqrt(T - 1)) / sqrt(1 - sample_skew * SR* + ((pearson_kurtosis - 1) / 4) * SR*^2))"
        or formula_metadata["benchmark_formula"]
        != "SR0 = sigma_SR * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - exp(-1) / N))"
        or formula_metadata["sample_size_symbol"]
        != "T is the stitched out-of-sample bar count used to estimate SR*."
        or formula_metadata["distribution_moments"]["sample_skew"]
        != "sample skewness of stitched out-of-sample bar returns"
        or formula_metadata["distribution_moments"]["pearson_kurtosis"]
        != "sample Pearson kurtosis of stitched out-of-sample bar returns with normal benchmark equal to 3"
        or formula_metadata["cross_sectional_dispersion"]
        != "sigma_SR is the sample standard deviation across the 186 observed candidate Sharpe ratios."
        or formula_metadata["failure_conditions"]
        != [
            "candidate is not evaluated for DSR when observed stitched out-of-sample Sharpe <= 0",
            "candidate fails when DSR <= 0.95",
            "candidate fails when T < 2",
            "candidate fails when the DSR denominator is non-finite or non-positive",
        ]
    ):
        raise ConfigValidationError("DSR formula metadata drifted from the frozen design.")
    permutation = multiple_testing["permutation_test"]
    if (
        not permutation["enabled"]
        or permutation["permutations"] != 500
        or permutation["seed"] != 42
        or permutation["seed_mode"] != "deterministic numpy SeedSequence seeded with 42"
        or permutation["family_null"] != "maximum Sharpe across all 186 candidates"
        or permutation["null_summary"] != "best-of-all-candidates null Sharpe across the full frozen family for each permutation draw"
        or permutation["empirical_p_formula"] != "(1 + count(null >= observed)) / 501"
        or not permutation["pass_rule"]["observed_sharpe_strictly_above_q95"]
        or permutation["pass_rule"]["empirical_p_lte"] != 0.05
    ):
        raise ConfigValidationError("Permutation-test rules drifted from the frozen design.")

    selection = plain["winner_selection"]
    if selection["no_winner_result"] != "NO_EDGE":
        raise ConfigValidationError("No-edge sentinel must remain NO_EDGE.")
    if not selection["holdout_forbidden_without_winner"]:
        raise ConfigValidationError("Holdout must stay forbidden without a winner.")
    holdout = selection["holdout"]
    if holdout["window"] != "last 6 calendar months" or not holdout["burn_before_read"] or holdout["max_final_winner_count"] != 1:
        raise ConfigValidationError("Holdout rules drifted from the frozen design.")

    hypotheses = plain["hypotheses"]
    if [hypothesis["id"] for hypothesis in hypotheses] != list(EXPECTED_TRIAL_COUNTS):
        raise ConfigValidationError("Hypothesis ordering or membership drifted from H1-H6.")

    total_trials = 0
    for hypothesis in hypotheses:
        total_trials += _validate_hypothesis(hypothesis, symbols, timeframes)

    if total_trials != EXPECTED_TOTAL_TRIALS:
        raise ConfigValidationError(f"Expected {EXPECTED_TOTAL_TRIALS} total trials but derived {total_trials}.")

    return config


def load_preregistered_config(path: Path = PRE_REGISTERED_CONFIG_PATH) -> Mapping[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    frozen = _deep_freeze(config)
    validate_preregistered_config(frozen)
    return frozen


PRE_REGISTERED_CONFIG = load_preregistered_config()
