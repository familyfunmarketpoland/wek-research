from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import NormalDist

import numpy as np

from research_lab.config import EXPECTED_TOTAL_TRIALS

DEFAULT_FAMILY_SIZE = EXPECTED_TOTAL_TRIALS
DEFAULT_DSR_THRESHOLD = 0.95
DEFAULT_PERMUTATIONS = 500
DEFAULT_PERMUTATION_SEED = 42
DEFAULT_COST_RATE_PER_SIDE = 0.0015
_EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio diagnostics for one observed stitched path."""

    sharpe: float
    sr0: float
    dsr: float
    passes: bool
    sample_size: int
    sample_skew: float
    pearson_kurtosis: float
    sigma_sr: float


@dataclass(frozen=True)
class PermutationTestResult:
    """Familywise best-of-all-candidates signal-path permutation diagnostics."""

    observed_sharpe: float
    null_sharpes: np.ndarray
    q95: float
    empirical_p: float
    passes: bool


@dataclass(frozen=True)
class CandidatePath:
    """One candidate's executed signal path for ragged family permutation tests."""

    candidate_id: int
    positions: Sequence[float] | np.ndarray
    market_returns: Sequence[float] | np.ndarray
    fold_ids: Sequence[int] | np.ndarray


@dataclass(frozen=True)
class ExecutionGroup:
    """Candidates sharing one exact open/close execution-price path."""

    opens: Sequence[float] | np.ndarray
    closes: Sequence[float] | np.ndarray
    fold_ids: Sequence[int] | np.ndarray
    candidate_ids: Sequence[int] | np.ndarray
    positions: Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class _NormalizedExecutionGroup:
    opens: np.ndarray
    closes: np.ndarray
    fold_ids: np.ndarray
    candidate_ids: np.ndarray
    positions: np.ndarray
    fold_slices: tuple[tuple[int, slice], ...]
    gap_returns: np.ndarray
    intrabar_returns: np.ndarray


PermutationCallback = Callable[..., Sequence[float] | np.ndarray]


def _as_float_1d(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _as_float_2d(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return array


def _broadcast_rows(values: Sequence[Sequence[float]] | np.ndarray, rows: int, *, name: str) -> np.ndarray:
    array = _as_float_2d(values, name=name)
    if array.shape[0] == 1 and rows != 1:
        array = np.repeat(array, rows, axis=0)
    if array.shape[0] != rows:
        raise ValueError(f"{name} must have either 1 row or {rows} rows.")
    return array


def _broadcast_int_rows(values: Sequence[Sequence[int]] | np.ndarray, rows: int, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer fold labels.")
    if array.shape[0] == 1 and rows != 1:
        array = np.repeat(array, rows, axis=0)
    if array.shape[0] != rows:
        raise ValueError(f"{name} must have either 1 row or {rows} rows.")
    return array.astype(int, copy=False)


def per_bar_sharpe(returns: Sequence[float] | np.ndarray) -> float:
    """Return nonannualized per-bar Sharpe, retaining zero/flat bars in T."""

    values = _as_float_1d(returns, name="returns")
    if values.size < 2 or not np.isfinite(values).all():
        return math.nan
    std = float(np.std(values, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return math.nan
    sharpe = float(np.mean(values) / std)
    return sharpe if math.isfinite(sharpe) else math.nan


def family_sharpe_for_multiple_testing(returns: Sequence[float] | np.ndarray) -> float:
    """Return Sharpe for family/null estimation, mapping undefined trials to 0.0."""

    sharpe = per_bar_sharpe(returns)
    return sharpe if math.isfinite(sharpe) else 0.0


def sample_skewness(returns: Sequence[float] | np.ndarray) -> float:
    """Return moment skewness m3 / m2**1.5 for stitched bar returns."""

    values = _as_float_1d(returns, name="returns")
    if values.size < 2 or not np.isfinite(values).all():
        return math.nan
    centered = values - float(np.mean(values))
    m2 = float(np.mean(centered**2))
    if not math.isfinite(m2) or m2 <= 0.0:
        return math.nan
    skew = float(np.mean(centered**3) / (m2**1.5))
    return skew if math.isfinite(skew) else math.nan


def sample_pearson_kurtosis(returns: Sequence[float] | np.ndarray) -> float:
    """Return Pearson kurtosis m4 / m2**2; the normal benchmark is 3."""

    values = _as_float_1d(returns, name="returns")
    if values.size < 2 or not np.isfinite(values).all():
        return math.nan
    centered = values - float(np.mean(values))
    m2 = float(np.mean(centered**2))
    if not math.isfinite(m2) or m2 <= 0.0:
        return math.nan
    kurtosis = float(np.mean(centered**4) / (m2**2))
    return kurtosis if math.isfinite(kurtosis) else math.nan


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    observed_family_sharpes: Sequence[float] | np.ndarray,
    *,
    family_size: int = DEFAULT_FAMILY_SIZE,
    threshold: float = DEFAULT_DSR_THRESHOLD,
) -> DeflatedSharpeResult:
    """Compute DSR for one observed positive Sharpe against the frozen family."""

    values = _as_float_1d(returns, name="returns")
    family = _as_float_1d(observed_family_sharpes, name="observed_family_sharpes")
    if family_size != DEFAULT_FAMILY_SIZE:
        raise ValueError(f"family_size must remain frozen at {DEFAULT_FAMILY_SIZE}.")
    if family.size != family_size:
        raise ValueError(f"observed_family_sharpes must contain exactly {family_size} values.")

    sample_size = int(values.size)
    sharpe = per_bar_sharpe(values)
    skew = sample_skewness(values)
    kurtosis = sample_pearson_kurtosis(values)
    family_for_sigma = np.where(np.isfinite(family), family, 0.0)
    sigma_sr = float(np.std(family_for_sigma, ddof=1)) if family_for_sigma.size >= 2 else math.nan
    sr0 = _deflated_sharpe_benchmark(sigma_sr, family_size)

    if (
        sample_size < 2
        or not math.isfinite(sharpe)
        or sharpe <= 0.0
        or not math.isfinite(skew)
        or not math.isfinite(kurtosis)
        or not math.isfinite(sigma_sr)
        or sigma_sr <= 0.0
        or not math.isfinite(sr0)
    ):
        return DeflatedSharpeResult(sharpe, sr0, math.nan, False, sample_size, skew, kurtosis, sigma_sr)

    variance_adjustment = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if not math.isfinite(variance_adjustment) or variance_adjustment <= 0.0:
        return DeflatedSharpeResult(sharpe, sr0, math.nan, False, sample_size, skew, kurtosis, sigma_sr)

    z_score = ((sharpe - sr0) * math.sqrt(sample_size - 1.0)) / math.sqrt(variance_adjustment)
    dsr = float(_NORMAL.cdf(z_score)) if math.isfinite(z_score) else math.nan
    passes = bool(math.isfinite(dsr) and dsr > threshold)
    return DeflatedSharpeResult(sharpe, sr0, dsr, passes, sample_size, skew, kurtosis, sigma_sr)


def _deflated_sharpe_benchmark(sigma_sr: float, family_size: int) -> float:
    if not math.isfinite(sigma_sr) or sigma_sr <= 0.0 or family_size <= 1:
        return math.nan
    first = _NORMAL.inv_cdf(1.0 - 1.0 / family_size)
    second = _NORMAL.inv_cdf(1.0 - math.exp(-1.0) / family_size)
    return float(sigma_sr * ((1.0 - _EULER_GAMMA) * first + _EULER_GAMMA * second))


def net_returns_from_position_path(
    market_returns: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    *,
    fold_ids: Sequence[int] | np.ndarray | None = None,
    cost_rate_per_side: float = DEFAULT_COST_RATE_PER_SIDE,
    initial_position: float = 0.0,
    terminal_liquidation: bool = True,
) -> np.ndarray:
    """Recompute bar PnL and costs, starting flat and liquidating at each fold end."""

    market = _as_float_1d(market_returns, name="market_returns")
    position = _as_float_1d(positions, name="positions")
    if market.shape != position.shape:
        raise ValueError("market_returns and positions must have the same shape.")
    if not np.isfinite(market).all() or not np.isfinite(position).all():
        return np.full_like(market, math.nan, dtype=float)
    costs = np.zeros_like(position, dtype=float)
    if fold_ids is None:
        previous = np.concatenate(([float(initial_position)], position[:-1]))
        costs += float(cost_rate_per_side) * np.abs(position - previous)
        if terminal_liquidation and position.size:
            costs[-1] += float(cost_rate_per_side) * abs(float(position[-1]))
        return position * market - costs

    folds = np.asarray(fold_ids)
    if folds.ndim != 1:
        raise ValueError("fold_ids must be one-dimensional.")
    if folds.shape[0] != position.shape[0]:
        raise ValueError("fold_ids and positions must have the same length.")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must contain integer fold labels.")
    for raw_fold_id in np.unique(folds):
        indices = np.flatnonzero(folds == raw_fold_id)
        if indices.size == 0:
            continue
        fold_position = position[indices]
        previous = np.concatenate(([0.0], fold_position[:-1]))
        costs[indices] += float(cost_rate_per_side) * np.abs(fold_position - previous)
        if terminal_liquidation:
            costs[indices[-1]] += float(cost_rate_per_side) * abs(float(fold_position[-1]))
    return position * market - costs


def circular_shift_by_fold(
    values: Sequence[float] | np.ndarray,
    fold_ids: Sequence[int] | np.ndarray,
    *,
    seed: int = DEFAULT_PERMUTATION_SEED,
    iteration: int,
    candidate_id: int,
) -> np.ndarray:
    """Circular-shift each fold independently with non-zero offsets when possible."""

    array = _as_float_1d(values, name="values")
    folds = np.asarray(fold_ids)
    if folds.ndim != 1:
        raise ValueError("fold_ids must be one-dimensional.")
    if folds.shape[0] != array.shape[0]:
        raise ValueError("fold_ids and values must have the same length.")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must be integer labels for SeedSequence reproducibility.")

    shifted = array.copy()
    for raw_fold_id in np.unique(folds):
        mask = folds == raw_fold_id
        fold_length = int(mask.sum())
        if fold_length < 2:
            continue
        seed_sequence = np.random.SeedSequence([int(seed), int(iteration), int(candidate_id), int(raw_fold_id)])
        rng = np.random.default_rng(seed_sequence)
        offset = int(rng.integers(1, fold_length))
        shifted[mask] = np.roll(array[mask], offset)
    return shifted


def signal_path_family_null(
    positions_by_candidate: Sequence[Sequence[float]] | np.ndarray,
    market_returns_by_candidate: Sequence[Sequence[float]] | np.ndarray,
    fold_ids_by_candidate: Sequence[Sequence[int]] | np.ndarray,
    *,
    candidate_ids: Sequence[int] | None = None,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    family_size: int = DEFAULT_FAMILY_SIZE,
    cost_rate_per_side: float = DEFAULT_COST_RATE_PER_SIDE,
    recompute_callback: PermutationCallback | None = None,
) -> np.ndarray:
    """Build the best-of-family null Sharpe distribution from dense candidate arrays."""

    if family_size != DEFAULT_FAMILY_SIZE:
        raise ValueError(f"family_size must remain frozen at {DEFAULT_FAMILY_SIZE}.")
    if permutations != DEFAULT_PERMUTATIONS:
        raise ValueError(f"permutations must remain frozen at {DEFAULT_PERMUTATIONS}.")
    if seed != DEFAULT_PERMUTATION_SEED:
        raise ValueError(f"seed must remain frozen at {DEFAULT_PERMUTATION_SEED}.")

    positions = _as_float_2d(positions_by_candidate, name="positions_by_candidate")
    if positions.shape[0] != family_size:
        raise ValueError(f"positions_by_candidate must contain exactly {family_size} candidates.")
    market_returns = _broadcast_rows(market_returns_by_candidate, family_size, name="market_returns_by_candidate")
    folds = _broadcast_int_rows(fold_ids_by_candidate, family_size, name="fold_ids_by_candidate")
    if market_returns.shape != positions.shape or folds.shape != positions.shape:
        raise ValueError("positions, market_returns, and fold_ids must have matching shapes.")
    if candidate_ids is None:
        candidate_ids_array = np.arange(family_size, dtype=int)
    else:
        candidate_ids_array = np.asarray(candidate_ids, dtype=int)
        if candidate_ids_array.shape != (family_size,):
            raise ValueError(f"candidate_ids must contain exactly {family_size} ids.")

    candidate_paths = [
        CandidatePath(
            candidate_id=int(candidate_ids_array[row_index]),
            positions=positions[row_index],
            market_returns=market_returns[row_index],
            fold_ids=folds[row_index],
        )
        for row_index in range(family_size)
    ]
    return signal_path_family_null_ragged(
        candidate_paths,
        permutations=permutations,
        seed=seed,
        family_size=family_size,
        cost_rate_per_side=cost_rate_per_side,
        recompute_callback=recompute_callback,
    )


def signal_path_family_null_ragged(
    candidate_paths: Sequence[CandidatePath],
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    family_size: int = DEFAULT_FAMILY_SIZE,
    cost_rate_per_side: float = DEFAULT_COST_RATE_PER_SIDE,
    recompute_callback: PermutationCallback | None = None,
) -> np.ndarray:
    """Build the best-of-family null distribution for variable-length candidates.

    Undefined/no-variance trial Sharpes are mapped to 0.0 inside the family max,
    so a no-trade candidate cannot make the whole permutation draw undefined.
    """

    if family_size != DEFAULT_FAMILY_SIZE:
        raise ValueError(f"family_size must remain frozen at {DEFAULT_FAMILY_SIZE}.")
    if permutations != DEFAULT_PERMUTATIONS:
        raise ValueError(f"permutations must remain frozen at {DEFAULT_PERMUTATIONS}.")
    if seed != DEFAULT_PERMUTATION_SEED:
        raise ValueError(f"seed must remain frozen at {DEFAULT_PERMUTATION_SEED}.")
    if len(candidate_paths) != family_size:
        raise ValueError(f"candidate_paths must contain exactly {family_size} candidates.")

    normalized = [_normalize_candidate_path(path) for path in candidate_paths]
    null_sharpes = np.empty(permutations, dtype=float)
    for iteration in range(permutations):
        iteration_sharpes = np.empty(family_size, dtype=float)
        for row_index, (path, positions, market_returns, folds) in enumerate(normalized):
            shifted_positions = circular_shift_by_fold(
                positions,
                folds,
                seed=seed,
                iteration=iteration,
                candidate_id=int(path.candidate_id),
            )
            if recompute_callback is None:
                permuted_returns = net_returns_from_position_path(
                    market_returns,
                    shifted_positions,
                    fold_ids=folds,
                    cost_rate_per_side=cost_rate_per_side,
                )
            else:
                permuted_returns = recompute_callback(
                    candidate_id=int(path.candidate_id),
                    iteration=int(iteration),
                    positions=shifted_positions,
                    market_returns=market_returns,
                    fold_ids=folds,
                )
            iteration_sharpes[row_index] = family_sharpe_for_multiple_testing(permuted_returns)
        null_sharpes[iteration] = float(np.max(iteration_sharpes))
    return null_sharpes


def _normalize_candidate_path(path: CandidatePath) -> tuple[CandidatePath, np.ndarray, np.ndarray, np.ndarray]:
    positions = _as_float_1d(path.positions, name="positions")
    market_returns = _as_float_1d(path.market_returns, name="market_returns")
    folds = np.asarray(path.fold_ids)
    if folds.ndim != 1:
        raise ValueError("fold_ids must be one-dimensional.")
    if market_returns.shape != positions.shape or folds.shape != positions.shape:
        raise ValueError("candidate positions, market_returns, and fold_ids must have matching lengths.")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must contain integer fold labels.")
    return path, positions, market_returns, folds.astype(int, copy=False)


def signal_path_family_null_execution_groups(
    groups: Sequence[ExecutionGroup],
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    chunk_size: int = 16,
    family_size: int = DEFAULT_FAMILY_SIZE,
    cost: float = DEFAULT_COST_RATE_PER_SIDE,
) -> np.ndarray:
    """Build the exact best-of-family null from shared execution-price groups.

    Candidate rows are processed in bounded chunks. Within a chunk, each path
    is shifted independently under the frozen SeedSequence contract, then each
    fold's exact open-gap, transition-cost, intrabar, and terminal-liquidation
    multipliers are reduced directly to return moments.
    """

    if family_size != DEFAULT_FAMILY_SIZE:
        raise ValueError(f"family_size must remain frozen at {DEFAULT_FAMILY_SIZE}.")
    if permutations != DEFAULT_PERMUTATIONS:
        raise ValueError(f"permutations must remain frozen at {DEFAULT_PERMUTATIONS}.")
    if seed != DEFAULT_PERMUTATION_SEED:
        raise ValueError(f"seed must remain frozen at {DEFAULT_PERMUTATION_SEED}.")
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)):
        raise ValueError("chunk_size must be a positive integer.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not math.isfinite(float(cost)) or float(cost) < 0.0:
        raise ValueError("cost must be finite and non-negative.")

    normalized = [_normalize_execution_group(group) for group in groups]
    candidate_count = sum(group.positions.shape[0] for group in normalized)
    if candidate_count != family_size:
        raise ValueError(f"groups must contain exactly {family_size} candidates in total.")
    all_candidate_ids = np.concatenate([group.candidate_ids for group in normalized])
    if np.unique(all_candidate_ids).size != family_size:
        raise ValueError("candidate_ids must be unique across all execution groups.")

    chunk_size = int(chunk_size)
    cost = float(cost)
    null_sharpes = np.full(permutations, -math.inf, dtype=float)
    for group in normalized:
        offsets_by_fold = _execution_group_offsets(group, permutations=permutations, seed=seed)
        candidate_count_in_group = int(group.positions.shape[0])
        for row_start in range(0, candidate_count_in_group, chunk_size):
            row_stop = min(row_start + chunk_size, candidate_count_in_group)
            row_slice = slice(row_start, row_stop)
            rows = row_stop - row_start
            for iteration in range(permutations):
                return_sum = np.zeros(rows, dtype=float)
                return_sum_squares = np.zeros(rows, dtype=float)
                sample_size = 0
                for fold_index, (_, fold_slice) in enumerate(group.fold_slices):
                    fold_positions = group.positions[row_slice, fold_slice]
                    fold_length = int(fold_positions.shape[1])
                    if fold_length < 2:
                        shifted_positions = fold_positions
                    else:
                        offsets = offsets_by_fold[fold_index][row_slice, iteration]
                        source_columns = (
                            np.arange(fold_length, dtype=np.intp)[None, :] - offsets[:, None]
                        ) % fold_length
                        shifted_positions = np.take_along_axis(fold_positions, source_columns, axis=1)

                    previous_positions = np.empty_like(shifted_positions)
                    previous_positions[:, 0] = 0
                    if fold_length > 1:
                        previous_positions[:, 1:] = shifted_positions[:, :-1]

                    multipliers = 1.0 + previous_positions * group.gap_returns[fold_slice]
                    multipliers *= 1.0 - cost * np.abs(shifted_positions - previous_positions)
                    multipliers *= 1.0 + shifted_positions * group.intrabar_returns[fold_slice]
                    multipliers[:, -1] *= 1.0 - cost * np.abs(shifted_positions[:, -1])
                    multipliers -= 1.0

                    return_sum += np.sum(multipliers, axis=1, dtype=float)
                    return_sum_squares += np.einsum(
                        "ij,ij->i", multipliers, multipliers, optimize=False
                    )
                    sample_size += fold_length

                sharpes = _family_sharpes_from_moments(
                    return_sum,
                    return_sum_squares,
                    sample_size,
                )
                null_sharpes[iteration] = max(
                    null_sharpes[iteration],
                    float(np.max(sharpes)),
                )
    return null_sharpes


def _normalize_execution_group(group: ExecutionGroup) -> _NormalizedExecutionGroup:
    opens = _as_float_1d(group.opens, name="opens")
    closes = _as_float_1d(group.closes, name="closes")
    if opens.size == 0:
        raise ValueError("execution groups must contain at least one bar.")
    if opens.shape != closes.shape:
        raise ValueError("opens and closes must have matching lengths.")
    if (
        not np.isfinite(opens).all()
        or not np.isfinite(closes).all()
        or np.any(opens <= 0.0)
        or np.any(closes <= 0.0)
    ):
        raise ValueError("opens and closes must be positive and finite.")

    folds = np.asarray(group.fold_ids)
    if folds.ndim != 1 or folds.shape != opens.shape:
        raise ValueError("fold_ids must be one-dimensional and match the price path length.")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must contain integer labels.")
    folds = folds.astype(int, copy=False)
    if np.any(folds < 0):
        raise ValueError("fold_ids must be non-negative for SeedSequence reproducibility.")

    candidate_ids = np.asarray(group.candidate_ids)
    if candidate_ids.ndim != 1 or not np.issubdtype(candidate_ids.dtype, np.integer):
        raise ValueError("candidate_ids must be a one-dimensional integer array.")
    candidate_ids = candidate_ids.astype(int, copy=False)
    if candidate_ids.size == 0:
        raise ValueError("execution groups must contain at least one candidate.")
    if np.any(candidate_ids < 0):
        raise ValueError("candidate_ids must be non-negative for SeedSequence reproducibility.")
    if np.unique(candidate_ids).size != candidate_ids.size:
        raise ValueError("candidate_ids must be unique within each execution group.")

    positions = _as_float_2d(group.positions, name="positions")
    if positions.shape != (candidate_ids.size, opens.size):
        raise ValueError("positions must have one row per candidate and one column per price bar.")
    if not np.isfinite(positions).all() or not np.isin(positions, (-1.0, 0.0, 1.0)).all():
        raise ValueError("positions must contain only finite -1, 0, or 1 values.")

    fold_starts = np.concatenate(([0], np.flatnonzero(folds[1:] != folds[:-1]) + 1))
    fold_stops = np.concatenate((fold_starts[1:], [folds.size]))
    fold_labels = folds[fold_starts]
    if np.unique(fold_labels).size != fold_labels.size:
        raise ValueError("each fold label must occupy one contiguous block.")
    fold_slices = tuple(
        (int(fold_id), slice(int(start), int(stop)))
        for fold_id, start, stop in zip(fold_labels, fold_starts, fold_stops, strict=True)
    )

    gap_returns = np.zeros_like(opens, dtype=float)
    for _, fold_slice in fold_slices:
        start = int(fold_slice.start)
        stop = int(fold_slice.stop)
        if stop - start > 1:
            gap_returns[start + 1 : stop] = opens[start + 1 : stop] / closes[start : stop - 1] - 1.0
    intrabar_returns = closes / opens - 1.0
    return _NormalizedExecutionGroup(
        opens=opens,
        closes=closes,
        fold_ids=folds,
        candidate_ids=candidate_ids,
        positions=positions.astype(np.int8, copy=False),
        fold_slices=fold_slices,
        gap_returns=gap_returns,
        intrabar_returns=intrabar_returns,
    )


def _execution_group_offsets(
    group: _NormalizedExecutionGroup,
    *,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.vstack(
            [
                _cached_permutation_offsets(
                    int(seed),
                    int(candidate_id),
                    int(fold_id),
                    int(fold_slice.stop) - int(fold_slice.start),
                    int(permutations),
                )
                for candidate_id in group.candidate_ids
            ]
        )
        for fold_id, fold_slice in group.fold_slices
    )


@lru_cache(maxsize=4096)
def _cached_permutation_offsets(
    seed: int,
    candidate_id: int,
    fold_id: int,
    fold_length: int,
    permutations: int,
) -> np.ndarray:
    if fold_length < 2:
        offsets = np.zeros(permutations, dtype=np.intp)
    else:
        offsets = np.fromiter(
            (
                int(
                    np.random.default_rng(
                        np.random.SeedSequence([seed, iteration, candidate_id, fold_id])
                    ).integers(1, fold_length)
                )
                for iteration in range(permutations)
            ),
            dtype=np.intp,
            count=permutations,
        )
    offsets.setflags(write=False)
    return offsets


def _family_sharpes_from_moments(
    return_sum: np.ndarray,
    return_sum_squares: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    sharpes = np.zeros_like(return_sum, dtype=float)
    if sample_size < 2:
        return sharpes
    centered_sum_squares = return_sum_squares - return_sum**2 / sample_size
    variance = centered_sum_squares / (sample_size - 1)
    valid = np.isfinite(return_sum) & np.isfinite(variance) & (variance > 0.0)
    sharpes[valid] = (return_sum[valid] / sample_size) / np.sqrt(variance[valid])
    sharpes[~np.isfinite(sharpes)] = 0.0
    return sharpes


def familywise_permutation_result(
    observed_returns: Sequence[float] | np.ndarray,
    null_sharpes: Sequence[float] | np.ndarray,
) -> PermutationTestResult:
    """Summarize q95, empirical p-value, and pass flag for a candidate path."""

    observed_sharpe = per_bar_sharpe(observed_returns)
    return familywise_permutation_result_from_sharpe(observed_sharpe, null_sharpes)


def familywise_permutation_result_from_sharpe(
    observed_sharpe: float,
    null_sharpes: Sequence[float] | np.ndarray,
) -> PermutationTestResult:
    """Summarize q95, empirical p-value, and pass flag for an observed Sharpe."""

    null = _as_float_1d(null_sharpes, name="null_sharpes").copy()
    if null.size != DEFAULT_PERMUTATIONS:
        raise ValueError(f"null_sharpes must contain exactly {DEFAULT_PERMUTATIONS} values.")
    if not math.isfinite(float(observed_sharpe)) or not np.isfinite(null).all():
        return PermutationTestResult(float(observed_sharpe), null, math.nan, math.nan, False)
    q95 = float(np.quantile(null, 0.95))
    empirical_p = float((1 + int(np.count_nonzero(null >= float(observed_sharpe)))) / (null.size + 1))
    passes = bool(float(observed_sharpe) > q95 and empirical_p <= 0.05)
    return PermutationTestResult(float(observed_sharpe), null, q95, empirical_p, passes)


__all__ = [
    "DEFAULT_COST_RATE_PER_SIDE",
    "DEFAULT_DSR_THRESHOLD",
    "DEFAULT_FAMILY_SIZE",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_PERMUTATION_SEED",
    "DeflatedSharpeResult",
    "CandidatePath",
    "ExecutionGroup",
    "PermutationTestResult",
    "circular_shift_by_fold",
    "deflated_sharpe_ratio",
    "family_sharpe_for_multiple_testing",
    "familywise_permutation_result",
    "familywise_permutation_result_from_sharpe",
    "net_returns_from_position_path",
    "per_bar_sharpe",
    "sample_pearson_kurtosis",
    "sample_skewness",
    "signal_path_family_null",
    "signal_path_family_null_execution_groups",
    "signal_path_family_null_ragged",
]
