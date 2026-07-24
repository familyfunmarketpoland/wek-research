from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab.config import (
    EXPECTED_TOTAL_TRIALS,
    EXPECTED_TRIAL_COUNTS,
    PRE_REGISTERED_CONFIG,
    PRE_REGISTERED_CONFIG_PATH,
    ConfigValidationError,
    load_preregistered_config,
    preregistered_config_fingerprint,
    validate_preregistered_config,
)


def _mutable_config():
    return json.loads(PRE_REGISTERED_CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_returns_immutable_validated_config() -> None:
    config = load_preregistered_config()

    assert config["universe"]["symbols"] == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    assert config["eligibility"]["minimum_oos_trades"] == 30
    with pytest.raises(TypeError):
        config["eligibility"]["minimum_oos_trades"] = 10
    with pytest.raises(TypeError):
        config["universe"]["symbols"] += ("XRP/USDT",)


def test_fingerprint_is_stable_for_loaded_and_plain_json() -> None:
    loaded = load_preregistered_config()
    plain = _mutable_config()

    assert preregistered_config_fingerprint(loaded) == preregistered_config_fingerprint(plain)
    assert preregistered_config_fingerprint(loaded) == preregistered_config_fingerprint(PRE_REGISTERED_CONFIG)


def test_validator_derives_exact_trial_counts_and_total() -> None:
    config = _mutable_config()

    validate_preregistered_config(config)

    counts = {hypothesis["id"]: hypothesis["expected_trial_count"] for hypothesis in config["hypotheses"]}
    assert counts == EXPECTED_TRIAL_COUNTS
    assert sum(counts.values()) == EXPECTED_TOTAL_TRIALS == 186


def test_validator_enforces_timeframe_applicability_and_walk_forward_names() -> None:
    config = _mutable_config()
    applicability = {hypothesis["id"]: hypothesis["applicability"] for hypothesis in config["hypotheses"]}

    assert config["walk_forward"] == {
        "train_months": 12,
        "oos_months": 3,
        "step_months": 3,
        "require_complete_folds": True,
        "scoring_scope": "stitched_oos",
        "adaptive_parameter_selection": False,
        "candidate_policy": "each pre-registered fixed candidate is scored on stitched out-of-sample results only",
    }
    assert applicability["H3"]["timeframes"] == ["1h"]
    assert applicability["H3"]["timeframes_na"] == ["4h"]
    assert applicability["H4"]["timeframes"] == ["1h", "4h"]
    assert applicability["H4"]["timeframes_na"] == []


def test_validator_enforces_h4_both_signs_within_candidate() -> None:
    config = _mutable_config()
    h4 = next(hypothesis for hypothesis in config["hypotheses"] if hypothesis["id"] == "H4")

    assert h4["trade_construction"]["side_policy"] == "both_signs_within_candidate"
    assert h4["parameter_grid"]["execution_axes"] == {
        "streak_length": [3, 4, 5, 6],
        "mode": ["continuation", "reversal"],
    }
    assert h4["parameter_grid"]["model_combo_axes"] == ["streak_length", "mode"]
    assert "side" not in h4["parameter_grid"]["execution_axes"]


def test_validator_rejects_trial_count_drift() -> None:
    config = _mutable_config()
    config["hypotheses"][3]["applicability"]["timeframes"] = ["1h"]
    config["hypotheses"][3]["applicability"]["timeframes_na"] = ["4h"]

    with pytest.raises(ConfigValidationError, match="H4 expected 48 trials but derives to 24"):
        validate_preregistered_config(config)


def test_validator_rejects_combo_cap_breach() -> None:
    config = _mutable_config()
    config["hypotheses"][1]["parameter_grid"]["execution_axes"]["hold_bars"].extend([60, 80, 100, 120])

    with pytest.raises(ConfigValidationError, match="H2 exceeds the 12-combo cap"):
        validate_preregistered_config(config)


def test_validator_rejects_min_trade_or_candidate_count_drift() -> None:
    config = _mutable_config()
    config["eligibility"]["minimum_oos_trades"] = 29

    with pytest.raises(ConfigValidationError, match="Minimum out-of-sample trade count must be 30"):
        validate_preregistered_config(config)

    config = _mutable_config()
    config["multiple_testing"]["candidate_count"] = 185

    with pytest.raises(ConfigValidationError, match="Candidate count must remain frozen at 186"):
        validate_preregistered_config(config)


def test_validator_enforces_dsr_and_permutation_metadata() -> None:
    config = _mutable_config()
    dsr = config["multiple_testing"]["deflated_sharpe_ratio"]["formula_metadata"]
    permutation = config["multiple_testing"]["permutation_test"]
    h3 = next(hypothesis for hypothesis in config["hypotheses"] if hypothesis["id"] == "H3")

    assert "per-bar nonannualized" in dsr["sharpe_definition"]
    assert "SR*" in dsr["dsr_formula"]
    assert "SR0" in dsr["benchmark_formula"]
    assert "Pearson kurtosis" in dsr["distribution_moments"]["pearson_kurtosis"]
    assert "sigma_SR" in dsr["cross_sectional_dispersion"]
    assert "T < 2" in dsr["failure_conditions"][2]
    assert permutation["seed_mode"] == "deterministic numpy SeedSequence seeded with 42"
    assert permutation["empirical_p_formula"] == "(1 + count(null >= observed)) / 501"
    assert h3["signal_definition"]["sessions_utc"]["Asia"]["exit_open_hour"] == "08:00"
    assert h3["signal_definition"]["sessions_utc"]["USA"]["exit_open_hour"] == "21:00"
