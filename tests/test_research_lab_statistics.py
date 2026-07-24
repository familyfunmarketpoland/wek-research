from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab.config import EXPECTED_TOTAL_TRIALS
from research_lab.engine import returns_from_position_path
from research_lab.statistics import (
    CandidatePath,
    DEFAULT_COST_RATE_PER_SIDE,
    DEFAULT_PERMUTATIONS,
    ExecutionGroup,
    circular_shift_by_fold,
    deflated_sharpe_ratio,
    family_sharpe_for_multiple_testing,
    familywise_permutation_result,
    familywise_permutation_result_from_sharpe,
    net_returns_from_position_path,
    per_bar_sharpe,
    sample_pearson_kurtosis,
    sample_skewness,
    signal_path_family_null,
    signal_path_family_null_execution_groups,
    signal_path_family_null_ragged,
)


EULER_GAMMA = 0.5772156649015329


def _exact_execution_groups() -> list[ExecutionGroup]:
    first_fold_positions = np.array([0, 1, -1, 1, -1], dtype=int)
    second_fold_positions = np.array([1, 1, -1, 0, -1, 1], dtype=int)
    desired_shifted = np.concatenate((first_fold_positions, second_fold_positions))
    fold_ids = np.repeat([0, 1], [first_fold_positions.size, second_fold_positions.size])
    base_positions = np.empty_like(desired_shifted)
    for fold_id in (0, 1):
        mask = fold_ids == fold_id
        fold_length = int(mask.sum())
        rng = np.random.default_rng(np.random.SeedSequence([42, 0, 0, fold_id]))
        offset = int(rng.integers(1, fold_length))
        base_positions[mask] = np.roll(desired_shifted[mask], -offset)

    opens = np.empty(desired_shifted.size, dtype=float)
    closes = np.empty(desired_shifted.size, dtype=float)
    for fold_id in (0, 1):
        indices = np.flatnonzero(fold_ids == fold_id)
        for local_index, bar_index in enumerate(indices):
            if local_index == 0:
                opens[bar_index] = 100.0 + 20.0 * fold_id
            else:
                previous_bar = indices[local_index - 1]
                previous_position = desired_shifted[previous_bar]
                opens[bar_index] = closes[previous_bar] * (1.0 + 0.01 * previous_position)
            closes[bar_index] = opens[bar_index] * (1.0 + 0.02 * desired_shifted[bar_index])

    first_positions = np.zeros((93, desired_shifted.size), dtype=int)
    first_positions[0] = base_positions
    first_group = ExecutionGroup(
        opens=opens,
        closes=closes,
        fold_ids=fold_ids,
        candidate_ids=np.arange(93),
        positions=first_positions,
    )

    second_bars = 8
    second_opens = np.array([200.0, 201.0, 199.0, 204.0, 150.0, 153.0, 151.0, 154.0])
    second_closes = np.array([201.0, 199.0, 202.0, 203.0, 153.0, 151.0, 154.0, 152.0])
    second_group = ExecutionGroup(
        opens=second_opens,
        closes=second_closes,
        fold_ids=np.repeat([0, 1], second_bars // 2),
        candidate_ids=np.arange(93, EXPECTED_TOTAL_TRIALS),
        positions=np.zeros((EXPECTED_TOTAL_TRIALS - 93, second_bars), dtype=int),
    )
    return [first_group, second_group]


def _execution_frame(group: ExecutionGroup, *, frequency: str) -> pd.DataFrame:
    opens = np.asarray(group.opens, dtype=float)
    closes = np.asarray(group.closes, dtype=float)
    index = pd.date_range("2024-01-01", periods=opens.size, freq=frequency, tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 1.0,
            "low": np.minimum(opens, closes) - 1.0,
            "close": closes,
            "volume": 1000.0,
        },
        index=index,
    )


def test_per_bar_sharpe_is_nonannualized_and_includes_flat_bars() -> None:
    returns = np.array([0.0, 0.10, 0.0, -0.05, 0.02])

    assert per_bar_sharpe(returns) == pytest.approx(float(returns.mean() / returns.std(ddof=1)))

    without_flat_bars = returns[returns != 0.0]
    assert per_bar_sharpe(returns) != pytest.approx(
        float(without_flat_bars.mean() / without_flat_bars.std(ddof=1))
    )


def test_sample_skewness_and_pearson_kurtosis_are_moment_based() -> None:
    symmetric = np.array([-1.0, 0.0, 1.0])
    skewed = np.array([-2.0, -1.0, 0.0, 1.0, 4.0])
    centered = skewed - skewed.mean()
    m2 = np.mean(centered**2)

    assert sample_skewness(symmetric) == pytest.approx(0.0)
    assert sample_pearson_kurtosis(symmetric) == pytest.approx(1.5)
    assert sample_pearson_kurtosis(symmetric) != pytest.approx(1.5 - 3.0)
    assert sample_skewness(skewed) == pytest.approx(float(np.mean(centered**3) / (m2**1.5)))
    assert sample_pearson_kurtosis(skewed) == pytest.approx(float(np.mean(centered**4) / (m2**2)))


def test_deflated_sharpe_ratio_matches_preregistered_formula() -> None:
    returns = np.array([0.0, 0.03, 0.00, 0.02, -0.01, 0.04, 0.00, 0.01])
    family_sharpes = np.linspace(-0.08, 0.12, EXPECTED_TOTAL_TRIALS)

    result = deflated_sharpe_ratio(returns, family_sharpes)

    sharpe = float(returns.mean() / returns.std(ddof=1))
    sigma_sr = float(np.std(family_sharpes, ddof=1))
    normal = NormalDist()
    sr0 = sigma_sr * (
        (1.0 - EULER_GAMMA) * normal.inv_cdf(1.0 - 1.0 / EXPECTED_TOTAL_TRIALS)
        + EULER_GAMMA * normal.inv_cdf(1.0 - math.exp(-1.0) / EXPECTED_TOTAL_TRIALS)
    )
    skew = sample_skewness(returns)
    kurtosis = sample_pearson_kurtosis(returns)
    denominator = math.sqrt(1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2)
    expected_dsr = normal.cdf(((sharpe - sr0) * math.sqrt(len(returns) - 1.0)) / denominator)

    assert result.sample_size == len(returns)
    assert result.sharpe == pytest.approx(sharpe)
    assert result.sigma_sr == pytest.approx(sigma_sr)
    assert result.sr0 == pytest.approx(sr0)
    assert result.sample_skew == pytest.approx(skew)
    assert result.pearson_kurtosis == pytest.approx(kurtosis)
    assert result.dsr == pytest.approx(expected_dsr)
    assert result.passes is (expected_dsr > 0.95)


def test_deflated_sharpe_ratio_invalid_cases_return_nan_and_false() -> None:
    family_sharpes = np.linspace(-0.08, 0.12, EXPECTED_TOTAL_TRIALS)

    negative = deflated_sharpe_ratio(np.array([-0.01, 0.0, -0.02]), family_sharpes)
    flat_family = deflated_sharpe_ratio(np.array([0.01, 0.02, 0.03]), np.zeros(EXPECTED_TOTAL_TRIALS))

    assert math.isnan(negative.dsr)
    assert negative.passes is False
    assert math.isnan(flat_family.dsr)
    assert flat_family.passes is False
    with pytest.raises(ValueError, match="exactly 186"):
        deflated_sharpe_ratio(np.array([0.01, 0.02]), np.zeros(EXPECTED_TOTAL_TRIALS - 1))


def test_deflated_sharpe_ratio_maps_missing_family_sharpes_to_zero_for_sigma_only() -> None:
    returns = np.array([0.0, 0.03, -0.01, 0.02, 0.04])
    family_sharpes = np.linspace(-0.05, 0.11, EXPECTED_TOTAL_TRIALS)
    family_sharpes[5] = np.nan
    family_sharpes[17] = np.inf

    result = deflated_sharpe_ratio(returns, family_sharpes)

    expected_sigma = np.std(np.where(np.isfinite(family_sharpes), family_sharpes, 0.0), ddof=1)
    assert result.sigma_sr == pytest.approx(float(expected_sigma))
    assert math.isfinite(result.dsr)


def test_family_sharpe_policy_keeps_observed_missing_as_nan_but_family_missing_as_zero() -> None:
    flat_returns = np.zeros(5)

    observed = familywise_permutation_result(flat_returns, np.zeros(DEFAULT_PERMUTATIONS))

    assert math.isnan(per_bar_sharpe(flat_returns))
    assert family_sharpe_for_multiple_testing(flat_returns) == 0.0
    assert math.isnan(observed.observed_sharpe)
    assert observed.passes is False


def test_net_returns_from_position_path_recomputes_entry_exit_and_terminal_costs() -> None:
    market_returns = np.array([0.01, 0.02, -0.01, 0.00, 0.03])
    positions = np.array([0.0, 1.0, 1.0, 0.0, -1.0])

    net = net_returns_from_position_path(market_returns, positions)

    expected_costs = DEFAULT_COST_RATE_PER_SIDE * np.array([0.0, 1.0, 0.0, 1.0, 2.0])
    expected = positions * market_returns - expected_costs
    np.testing.assert_allclose(net, expected)


def test_net_returns_from_position_path_starts_flat_and_liquidates_each_fold() -> None:
    market_returns = np.array([0.01, 0.02, -0.01, 0.04, -0.02, 0.01])
    positions = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 0.0])
    folds = np.array([0, 0, 0, 1, 1, 1])

    net = net_returns_from_position_path(market_returns, positions, fold_ids=folds)

    expected_costs = DEFAULT_COST_RATE_PER_SIDE * np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    expected = positions * market_returns - expected_costs
    np.testing.assert_allclose(net, expected)


def test_net_returns_from_position_path_charges_reversal_as_two_sides() -> None:
    market_returns = np.array([0.01, 0.02, 0.03])
    positions = np.array([1.0, -1.0, 0.0])

    net = net_returns_from_position_path(market_returns, positions, terminal_liquidation=False)

    expected_costs = DEFAULT_COST_RATE_PER_SIDE * np.array([1.0, 2.0, 1.0])
    expected = positions * market_returns - expected_costs
    np.testing.assert_allclose(net, expected)


def test_circular_shift_by_fold_is_deterministic_and_uses_nonzero_offsets() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    folds = np.array([0, 0, 0, 0, 1, 1, 1])

    first = circular_shift_by_fold(values, folds, iteration=7, candidate_id=13)
    second = circular_shift_by_fold(values, folds, iteration=7, candidate_id=13)

    np.testing.assert_array_equal(first, second)
    for fold_id in (0, 1):
        mask = folds == fold_id
        assert sorted(first[mask].tolist()) == sorted(values[mask].tolist())
        assert not np.array_equal(first[mask], values[mask])


def test_execution_group_null_matches_engine_for_gaps_reversals_and_fold_liquidation() -> None:
    groups = _exact_execution_groups()

    null = signal_path_family_null_execution_groups(
        groups,
        chunk_size=16,
        cost=DEFAULT_COST_RATE_PER_SIDE,
    )

    winner_group = groups[0]
    shifted = circular_shift_by_fold(
        np.asarray(winner_group.positions)[0],
        winner_group.fold_ids,
        iteration=0,
        candidate_id=0,
    )
    frame = _execution_frame(winner_group, frequency="1h")
    engine_returns = returns_from_position_path(
        frame,
        pd.Series(shifted, index=frame.index),
        fold_ids=winner_group.fold_ids,
        cost_rate_per_side=DEFAULT_COST_RATE_PER_SIDE,
    )
    expected_sharpe = family_sharpe_for_multiple_testing(engine_returns.to_numpy())

    assert expected_sharpe > 0.0
    assert np.any(np.abs(np.diff(shifted)) == 2)
    for fold_id in (0, 1):
        fold_indices = np.flatnonzero(np.asarray(winner_group.fold_ids) == fold_id)
        assert shifted[fold_indices[-1]] != 0
        assert np.any(
            np.asarray(winner_group.opens)[fold_indices[1:]]
            != np.asarray(winner_group.closes)[fold_indices[:-1]]
        )
    assert null[0] == pytest.approx(expected_sharpe, rel=1e-12, abs=1e-12)


def test_execution_group_null_is_deterministic_chunk_invariant_and_supports_mixed_lengths() -> None:
    groups = _exact_execution_groups()

    by_chunk = {
        chunk_size: signal_path_family_null_execution_groups(groups, chunk_size=chunk_size)
        for chunk_size in (1, 7, 16, 64)
    }
    repeated = signal_path_family_null_execution_groups(groups, chunk_size=16)

    assert len(np.asarray(groups[0].opens)) != len(np.asarray(groups[1].opens))
    for result in by_chunk.values():
        assert result.shape == (DEFAULT_PERMUTATIONS,)
        np.testing.assert_array_equal(result, by_chunk[16])
    np.testing.assert_array_equal(repeated, by_chunk[16])


def test_execution_group_null_maps_no_variance_family_trials_to_zero() -> None:
    groups = _exact_execution_groups()
    flat_groups = [
        ExecutionGroup(
            opens=group.opens,
            closes=group.closes,
            fold_ids=group.fold_ids,
            candidate_ids=group.candidate_ids,
            positions=np.zeros_like(np.asarray(group.positions)),
        )
        for group in groups
    ]

    null = signal_path_family_null_execution_groups(flat_groups, chunk_size=64)

    np.testing.assert_array_equal(null, np.zeros(DEFAULT_PERMUTATIONS))


def test_execution_group_null_requires_the_frozen_unique_186_candidate_family() -> None:
    groups = _exact_execution_groups()
    short_second = groups[1]
    missing_one = [
        groups[0],
        ExecutionGroup(
            opens=short_second.opens,
            closes=short_second.closes,
            fold_ids=short_second.fold_ids,
            candidate_ids=np.asarray(short_second.candidate_ids)[:-1],
            positions=np.asarray(short_second.positions)[:-1],
        ),
    ]

    with pytest.raises(ValueError, match="exactly 186"):
        signal_path_family_null_execution_groups(missing_one)


def test_signal_path_family_null_has_deterministic_500_shape_and_best_of_family_max() -> None:
    bars = 16
    folds = np.tile(np.repeat(np.arange(4), bars // 4), (EXPECTED_TOTAL_TRIALS, 1))
    market = np.tile(np.linspace(-0.015, 0.018, bars), (EXPECTED_TOTAL_TRIALS, 1))
    positions = np.zeros((EXPECTED_TOTAL_TRIALS, bars))
    for candidate_id in range(EXPECTED_TOTAL_TRIALS):
        positions[candidate_id] = np.where((np.arange(bars) + candidate_id) % 3 == 0, 1.0, 0.0)
    positions[-1] = np.array([1.0, 1.0, 0.0, -1.0] * 4)

    first = signal_path_family_null(positions, market, folds)
    second = signal_path_family_null(positions, market, folds)

    assert first.shape == (DEFAULT_PERMUTATIONS,)
    np.testing.assert_allclose(first, second)
    manual_iteration_zero = []
    for candidate_id in range(EXPECTED_TOTAL_TRIALS):
        shifted = circular_shift_by_fold(
            positions[candidate_id],
            folds[candidate_id],
            iteration=0,
            candidate_id=candidate_id,
        )
        manual_iteration_zero.append(
            family_sharpe_for_multiple_testing(
                net_returns_from_position_path(market[candidate_id], shifted, fold_ids=folds[candidate_id])
            )
        )
    assert first[0] == pytest.approx(float(np.nanmax(manual_iteration_zero)))


def test_signal_path_family_null_ragged_handles_variable_lengths_and_maps_missing_to_zero() -> None:
    candidate_paths: list[CandidatePath] = []
    for candidate_id in range(EXPECTED_TOTAL_TRIALS):
        bars = 9 + candidate_id % 5
        fold_ids = np.repeat(np.arange(3), math.ceil(bars / 3))[:bars]
        market_returns = np.linspace(-0.012, 0.018, bars) + candidate_id * 0.00001
        positions = np.where((np.arange(bars) + candidate_id) % 4 == 0, 1.0, 0.0)
        if candidate_id == 7:
            positions = np.zeros(bars)
            market_returns = np.zeros(bars)
        candidate_paths.append(
            CandidatePath(
                candidate_id=candidate_id,
                positions=positions,
                market_returns=market_returns,
                fold_ids=fold_ids,
            )
        )

    first = signal_path_family_null_ragged(candidate_paths)
    second = signal_path_family_null_ragged(candidate_paths)

    assert first.shape == (DEFAULT_PERMUTATIONS,)
    np.testing.assert_allclose(first, second)
    manual_iteration_zero = []
    for path in candidate_paths:
        shifted = circular_shift_by_fold(path.positions, path.fold_ids, iteration=0, candidate_id=path.candidate_id)
        returns = net_returns_from_position_path(path.market_returns, shifted, fold_ids=path.fold_ids)
        manual_iteration_zero.append(family_sharpe_for_multiple_testing(returns))
    assert manual_iteration_zero[7] == 0.0
    assert first[0] == pytest.approx(float(np.max(manual_iteration_zero)))


def test_signal_path_family_null_accepts_recompute_callback() -> None:
    bars = 8
    positions = np.ones((EXPECTED_TOTAL_TRIALS, bars))
    market = np.tile(np.linspace(-0.01, 0.02, bars), (EXPECTED_TOTAL_TRIALS, 1))
    folds = np.tile(np.repeat([0, 1], bars // 2), (EXPECTED_TOTAL_TRIALS, 1))
    calls = 0

    def recompute_callback(**kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        shifted_positions = np.asarray(kwargs["positions"], dtype=float)
        market_returns = np.asarray(kwargs["market_returns"], dtype=float)
        return shifted_positions * market_returns

    null = signal_path_family_null(positions, market, folds, recompute_callback=recompute_callback)

    assert null.shape == (DEFAULT_PERMUTATIONS,)
    assert calls == DEFAULT_PERMUTATIONS * EXPECTED_TOTAL_TRIALS


def test_signal_path_family_null_ragged_accepts_recompute_callback() -> None:
    candidate_paths = []
    for candidate_id in range(EXPECTED_TOTAL_TRIALS):
        bars = 6 + candidate_id % 4
        candidate_paths.append(
            CandidatePath(
                candidate_id=candidate_id,
                positions=np.ones(bars),
                market_returns=np.linspace(-0.01, 0.02, bars),
                fold_ids=np.repeat(np.arange(2), math.ceil(bars / 2))[:bars],
            )
        )
    calls = 0

    def recompute_callback(**kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        shifted_positions = np.asarray(kwargs["positions"], dtype=float)
        market_returns = np.asarray(kwargs["market_returns"], dtype=float)
        return shifted_positions * market_returns

    null = signal_path_family_null_ragged(candidate_paths, recompute_callback=recompute_callback)

    assert null.shape == (DEFAULT_PERMUTATIONS,)
    assert calls == DEFAULT_PERMUTATIONS * EXPECTED_TOTAL_TRIALS


def test_familywise_permutation_result_uses_q95_empirical_p_and_strict_pass_rule() -> None:
    passing_null = np.array([0.0] * 476 + [1.0] * 24)
    failing_null = np.array([0.0] * 475 + [1.0] * 25)
    observed_returns = np.array([0.0, 0.0, 0.0, 1.0])
    observed_sharpe = per_bar_sharpe(observed_returns)

    passing = familywise_permutation_result(observed_returns, passing_null)
    failing = familywise_permutation_result_from_sharpe(0.5, failing_null)

    assert passing.observed_sharpe == pytest.approx(observed_sharpe)
    assert passing.q95 == pytest.approx(float(np.quantile(passing_null, 0.95)))
    assert passing.empirical_p == pytest.approx((1 + 24) / 501)
    assert passing.passes is True
    assert failing.empirical_p == pytest.approx((1 + 25) / 501)
    assert failing.passes is False


def test_familywise_permutation_result_nonfinite_fails() -> None:
    null = np.zeros(DEFAULT_PERMUTATIONS)
    null[0] = np.nan

    result = familywise_permutation_result_from_sharpe(1.0, null)

    assert math.isnan(result.q95)
    assert math.isnan(result.empirical_p)
    assert result.passes is False
