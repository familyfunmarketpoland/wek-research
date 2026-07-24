from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from research_lab.config import PRE_REGISTERED_CONFIG
from research_lab.features import compute_features, validate_ohlcv_frame
from research_lab.hypotheses import Candidate, SignalBundle, generate_signals


@dataclass
class SignalBacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    positions: pd.Series
    orders: pd.DataFrame


@dataclass
class CandidateWalkForwardResult:
    candidate: Candidate
    folds: pd.DataFrame
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    fold_positions: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class FixedWalkForwardResult:
    folds: pd.DataFrame
    candidate_results: pd.DataFrame
    returns: dict[str, pd.Series]
    trades: pd.DataFrame
    fold_positions: pd.DataFrame


def run_signal_backtest(
    frame: pd.DataFrame,
    signals: SignalBundle,
    *,
    hold_bars: int | None = None,
    active_start: pd.Timestamp | None = None,
    cost_rate_per_side: float = 0.0015,
    initial_capital: float = 1.0,
    bars_per_year: float | None = None,
) -> SignalBacktestResult:
    """Execute deterministic synthetic constant-1x long/short signals.

    Close-stamped signals are executed at the next open. H3-style open targets
    are applied directly at that bar's open. Position changes pay one side of
    cost for entry or exit and two sides for long/short reversals.
    """

    validate_ohlcv_frame(frame)
    if cost_rate_per_side < 0:
        raise ValueError("cost_rate_per_side must be non-negative")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    hold = signals.candidate.hold_bars if hold_bars is None else hold_bars
    if hold is None and signals.kind == "close":
        raise ValueError("hold_bars is required for close-stamped signals")
    if hold is not None and hold <= 0:
        raise ValueError("hold_bars must be positive")

    signal = signals.signal.reindex(frame.index).fillna(0).astype(int)
    exit_signal = signals.exit_signal.reindex(frame.index).fillna(False).astype(bool)
    active_start = active_start if active_start is not None else frame.index[0]

    opens = frame["open"].to_numpy(dtype=float, copy=False)
    closes = frame["close"].to_numpy(dtype=float, copy=False)
    active = np.asarray(frame.index >= active_start, dtype=bool)
    index = frame.index

    position = 0
    entry_idx: int | None = None
    pending_target: int | None = None
    pending_reason = "signal"

    position_values = np.zeros(len(index), dtype=int)
    transition_reasons = pd.Series("", index=index, dtype=object, name="reason")

    for i, timestamp in enumerate(index):
        _ = timestamp
        if signals.kind == "open_target" and active[i]:
            target = int(signal.iloc[i])
            target_changed = i == 0 or target != int(signal.iloc[i - 1])
            if target_changed and target != position:
                position = target
                entry_idx = i if target != 0 else None
                transition_reasons.iloc[i] = "session_target"
        elif pending_target is not None and active[i]:
            position = pending_target
            entry_idx = i if pending_target != 0 else None
            transition_reasons.iloc[i] = pending_reason
            pending_target = None

        position_values[i] = position if active[i] else 0

        if i == len(index) - 1:
            continue
        if not (active[i] or active[i + 1]):
            continue
        if active[i] and not active[i + 1]:
            continue

        if signals.kind == "open_target":
            continue

        next_target = position
        reason = "signal"
        if position != 0 and bool(exit_signal.iloc[i]):
            next_target = 0
            reason = "opposite_channel"
        if position != 0 and hold is not None and entry_idx is not None and (i - entry_idx + 1) >= hold:
            next_target = 0
            reason = "time"

        desired = int(signal.iloc[i])
        if desired != 0:
            if position == 0:
                next_target = desired
                reason = "signal"
            elif signals.allow_reversal and desired != position:
                next_target = desired
                reason = "reverse_signal"
            elif next_target == 0 and signals.allow_reversal:
                next_target = desired
                reason = "time_reentry"

        if next_target != position:
            pending_target = next_target
            pending_reason = reason

    positions = pd.Series(position_values, index=index, dtype=int, name="position")
    active_frame = frame
    active_reasons = transition_reasons
    if active_start != index[0]:
        active_frame = frame.loc[frame.index >= active_start]
        positions = positions.loc[positions.index >= active_start]
        active_reasons = transition_reasons.loc[transition_reasons.index >= active_start]

    returns, equity, trades_frame, orders_frame = _replay_position_path(
        active_frame,
        positions,
        reasons=active_reasons,
        cost_rate_per_side=cost_rate_per_side,
        initial_capital=initial_capital,
    )

    return SignalBacktestResult(
        equity=equity,
        returns=returns,
        trades=trades_frame,
        metrics=metric_summary(
            equity,
            returns,
            trades_frame,
            positions.ne(0),
            initial_capital=initial_capital,
            bars_per_year=bars_per_year,
        ),
        positions=positions,
        orders=orders_frame,
    )


def walk_forward_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    config: Mapping[str, object] = PRE_REGISTERED_CONFIG,
    features: pd.DataFrame | None = None,
    cost_rate_per_side: float | None = None,
    bars_per_year: float | None = None,
) -> CandidateWalkForwardResult:
    validate_ohlcv_frame(frame)
    wf = config["walk_forward"]  # type: ignore[index]
    cost_rate = (
        float(config["costs"]["cost_rate_per_side"])  # type: ignore[index]
        if cost_rate_per_side is None
        else float(cost_rate_per_side)
    )
    if features is None:
        features = compute_features(frame, timeframe=candidate.timeframe)
    signals = generate_signals(frame, candidate, features=features)

    fold_rows: list[dict[str, object]] = []
    returns_parts: list[pd.Series] = []
    trade_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    delta = infer_bar_delta(frame.index)
    train_start = frame.index[0]
    fold_id = 0
    while True:
        train_end = train_start + pd.DateOffset(months=int(wf["train_months"]))
        oos_end = train_end + pd.DateOffset(months=int(wf["oos_months"]))
        if delta is None or frame.index[-1] + delta < oos_end:
            break
        oos_mask = (frame.index >= train_end) & (frame.index < oos_end)
        if not oos_mask.any():
            break
        context_mask = oos_mask.copy()
        prior_positions = np.flatnonzero(frame.index < train_end)
        if len(prior_positions):
            context_mask[prior_positions[-1]] = True
        context = frame.loc[context_mask]
        active_start = frame.index[oos_mask][0]
        result = run_signal_backtest(
            context,
            SignalBundle(
                candidate=signals.candidate,
                kind=signals.kind,
                signal=signals.signal,
                exit_signal=signals.exit_signal,
                allow_reversal=signals.allow_reversal,
            ),
            active_start=active_start,
            cost_rate_per_side=cost_rate,
            bars_per_year=bars_per_year,
        )
        if not result.returns.empty:
            returns_parts.append(result.returns)
        if not result.trades.empty:
            trades = result.trades.copy()
            trades.insert(0, "fold", fold_id)
            trades.insert(1, "candidate_id", candidate.candidate_id)
            trade_parts.append(trades)
        positions = result.positions.to_frame()
        positions.insert(0, "fold", fold_id)
        positions.insert(1, "candidate_id", candidate.candidate_id)
        position_parts.append(positions)
        fold_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "fold": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "oos_start": active_start,
                "oos_end": frame.index[oos_mask][-1],
                **{f"oos_{key}": value for key, value in result.metrics.items()},
            }
        )
        fold_id += 1
        train_start = train_start + pd.DateOffset(months=int(wf["step_months"]))

    returns = pd.concat(returns_parts).sort_index() if returns_parts else pd.Series(dtype=float, name="returns")
    equity = (1.0 + returns).cumprod().rename("equity") if not returns.empty else pd.Series(dtype=float, name="equity")
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else _empty_trades(candidate_id=True)
    fold_positions = pd.concat(position_parts).sort_index() if position_parts else pd.DataFrame()
    metrics = metric_summary(
        equity,
        returns,
        trades,
        fold_positions["position"].ne(0) if "position" in fold_positions else pd.Series(dtype=bool),
        initial_capital=1.0,
        bars_per_year=bars_per_year,
    )
    return CandidateWalkForwardResult(
        candidate=candidate,
        folds=pd.DataFrame(fold_rows),
        returns=returns,
        equity=equity,
        trades=trades,
        fold_positions=fold_positions,
        metrics=metrics,
    )


def walk_forward_fixed_candidates(
    frame: pd.DataFrame,
    candidates: Sequence[Candidate],
    *,
    config: Mapping[str, object] = PRE_REGISTERED_CONFIG,
    cost_rate_per_side: float | None = None,
    bars_per_year: float | None = None,
) -> FixedWalkForwardResult:
    validate_ohlcv_frame(frame)
    features_by_timeframe: dict[str, pd.DataFrame] = {}
    fold_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    returns_by_candidate: dict[str, pd.Series] = {}
    trade_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.timeframe not in features_by_timeframe:
            features_by_timeframe[candidate.timeframe] = compute_features(frame, timeframe=candidate.timeframe)
        result = walk_forward_candidate(
            frame,
            candidate,
            config=config,
            features=features_by_timeframe[candidate.timeframe],
            cost_rate_per_side=cost_rate_per_side,
            bars_per_year=bars_per_year,
        )
        if not result.folds.empty:
            fold_parts.append(result.folds)
        summary_rows.append({**candidate.as_dict(), **result.metrics})
        returns_by_candidate[candidate.candidate_id] = result.returns
        if not result.trades.empty:
            trade_parts.append(result.trades)
        if not result.fold_positions.empty:
            position_parts.append(result.fold_positions)
    return FixedWalkForwardResult(
        folds=pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame(),
        candidate_results=pd.DataFrame(summary_rows),
        returns=returns_by_candidate,
        trades=pd.concat(trade_parts, ignore_index=True) if trade_parts else _empty_trades(candidate_id=True),
        fold_positions=pd.concat(position_parts).sort_index() if position_parts else pd.DataFrame(),
    )


def returns_from_position_path(
    frame: pd.DataFrame,
    positions: pd.Series | pd.DataFrame,
    *,
    fold_ids: pd.Series | Sequence[object] | None = None,
    cost_rate_per_side: float = 0.0015,
    initial_capital: float = 1.0,
) -> pd.Series:
    """Replay constant-1x PnL from an executed target position path.

    ``positions`` is the exposure held during each bar after any open trade has
    executed. Non-zero final fold positions are liquidated at that fold's final
    close for PnL, while the input path still records the exposure held during
    that final bar. Each fold starts flat with ``initial_capital``.
    """

    validate_ohlcv_frame(frame)
    if cost_rate_per_side < 0:
        raise ValueError("cost_rate_per_side must be non-negative")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    path = _coerce_position_series(positions, frame.index)
    if fold_ids is None:
        returns, _, _, _ = _replay_position_path(
            frame,
            path,
            cost_rate_per_side=cost_rate_per_side,
            initial_capital=initial_capital,
        )
        return returns

    folds = pd.Series(fold_ids, index=frame.index, name="fold")
    if len(folds) != len(frame):
        raise ValueError("fold_ids must have the same length as frame")
    if folds.isna().any():
        raise ValueError("fold_ids must not contain missing values")

    parts: list[pd.Series] = []
    start = 0
    fold_values = folds.to_numpy(copy=False)
    for i in range(1, len(fold_values) + 1):
        if i == len(fold_values) or fold_values[i] != fold_values[start]:
            segment_index = frame.index[start:i]
            parts.append(
                _replay_position_path(
                    frame.loc[segment_index],
                    path.loc[segment_index],
                    cost_rate_per_side=cost_rate_per_side,
                    initial_capital=initial_capital,
                )[0]
            )
            start = i
    return pd.concat(parts).rename("returns") if parts else pd.Series(dtype=float, name="returns")


def metric_summary(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    exposure: pd.Series,
    *,
    initial_capital: float,
    bars_per_year: float | None,
) -> dict[str, float]:
    if equity.empty:
        return {
            "total_return": np.nan,
            "CAGR": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "trades": 0.0,
            "exposure": np.nan,
        }
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    bpy = float(bars_per_year) if bars_per_year is not None else infer_bars_per_year(equity.index)
    periods = max(len(equity), 1)
    cagr = float((equity.iloc[-1] / initial_capital) ** (bpy / periods) - 1.0) if equity.iloc[-1] > 0 else np.nan
    std = returns.std(ddof=0)
    sharpe = float(returns.mean() / std * math.sqrt(bpy)) if std > 0 else np.nan
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = float(returns.mean() / downside_std * math.sqrt(bpy)) if downside_std > 0 else np.nan
    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1.0).min())
    if trades.empty:
        win_rate = np.nan
        profit_factor = np.nan
        trade_count = 0
    else:
        net = trades["net_return"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]
        win_rate = float(len(wins) / len(net))
        if len(losses) == 0:
            profit_factor = np.inf if len(wins) else np.nan
        else:
            profit_factor = float(wins.sum() / abs(losses.sum())) if len(wins) else 0.0
        trade_count = len(net)
    return {
        "total_return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": float(trade_count),
        "exposure": float(exposure.mean()) if len(exposure) else np.nan,
    }


def infer_bars_per_year(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 365.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = deltas[deltas > 0].median()
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        return 365.0
    return float(365.0 * 24.0 * 60.0 * 60.0 / median_seconds)


def infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return None
    return positive.median()


def _coerce_position_series(positions: pd.Series | pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    if isinstance(positions, pd.DataFrame):
        if "position" not in positions:
            raise ValueError("positions DataFrame must contain a 'position' column")
        series = positions["position"]
    elif isinstance(positions, pd.Series):
        series = positions
    else:
        raise TypeError("positions must be a pandas Series or DataFrame")
    if len(series) != len(index):
        raise ValueError("positions must have the same length as frame")
    series = pd.Series(series.to_numpy(), index=index, name="position")
    if series.isna().any():
        raise ValueError("positions must not contain missing values")
    if not series.isin([-1, 0, 1]).all():
        raise ValueError("positions must contain only -1, 0, or 1")
    return series.astype(int)


def _replay_position_path(
    frame: pd.DataFrame,
    positions: pd.Series,
    *,
    reasons: pd.Series | None = None,
    cost_rate_per_side: float,
    initial_capital: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return (
            pd.Series(dtype=float, name="returns"),
            pd.Series(dtype=float, name="equity"),
            _empty_trades(),
            _empty_orders(),
        )
    reasons = (
        pd.Series("", index=frame.index, dtype=object, name="reason")
        if reasons is None
        else pd.Series(reasons.reindex(frame.index).fillna("").to_numpy(), index=frame.index, dtype=object)
    )
    equity = float(initial_capital)
    previous_position = 0
    entry_idx: int | None = None
    entry_time: pd.Timestamp | None = None
    entry_price = np.nan
    entry_equity = np.nan
    equity_values = np.empty(len(frame), dtype=float)
    trades: list[dict[str, object]] = []
    orders: list[dict[str, object]] = []
    opens = frame["open"].to_numpy(dtype=float, copy=False)
    closes = frame["close"].to_numpy(dtype=float, copy=False)
    for i, timestamp in enumerate(frame.index):
        target = int(positions.iloc[i])
        open_price = float(opens[i])
        close_price = float(closes[i])
        if open_price <= 0 or close_price <= 0 or not np.isfinite(open_price) or not np.isfinite(close_price):
            raise ValueError("open and close prices must be positive and finite")

        pre_open_equity = equity
        if i > 0 and previous_position != 0:
            pre_open_equity *= 1.0 + previous_position * (open_price / float(closes[i - 1]) - 1.0)

        transition_equity = pre_open_equity
        reason = str(reasons.iloc[i] or "position_path")
        if target != previous_position:
            if previous_position != 0:
                exit_cost = pre_open_equity * cost_rate_per_side
                after_exit_equity = pre_open_equity - exit_cost
                if entry_idx is None or entry_time is None:
                    raise RuntimeError("cannot exit without an entry")
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "side": "long" if previous_position == 1 else "short",
                        "entry_price": float(entry_price),
                        "exit_price": open_price,
                        "bars": int(max(i - entry_idx, 1)),
                        "gross_return": float((open_price / entry_price - 1.0) * previous_position),
                        "net_return": float(after_exit_equity / entry_equity - 1.0),
                        "reason": reason,
                    }
                )
                orders.append(
                    {
                        "time": timestamp,
                        "from_position": previous_position,
                        "to_position": 0,
                        "price": open_price,
                        "cost": float(exit_cost),
                        "reason": reason,
                    }
                )
                transition_equity = after_exit_equity
                entry_idx = None
                entry_time = None
                entry_price = np.nan
                entry_equity = np.nan
            if target != 0:
                entry_base_equity = transition_equity
                entry_cost = pre_open_equity * cost_rate_per_side
                transition_equity -= entry_cost
                orders.append(
                    {
                        "time": timestamp,
                        "from_position": 0,
                        "to_position": target,
                        "price": open_price,
                        "cost": float(entry_cost),
                        "reason": reason,
                    }
                )
                entry_idx = i
                entry_time = timestamp
                entry_price = open_price
                entry_equity = entry_base_equity

        equity = transition_equity * (1.0 + target * (close_price / open_price - 1.0))
        if i == len(frame) - 1 and target != 0:
            liquidation_cost = equity * cost_rate_per_side
            final_equity = equity - liquidation_cost
            if entry_idx is None or entry_time is None:
                raise RuntimeError("cannot liquidate without an entry")
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "side": "long" if target == 1 else "short",
                    "entry_price": float(entry_price),
                    "exit_price": close_price,
                    "bars": int(max(i - entry_idx, 1)),
                    "gross_return": float((close_price / entry_price - 1.0) * target),
                    "net_return": float(final_equity / entry_equity - 1.0),
                    "reason": "final_liquidation",
                }
            )
            orders.append(
                {
                    "time": timestamp,
                    "from_position": target,
                    "to_position": 0,
                    "price": close_price,
                    "cost": float(liquidation_cost),
                    "reason": "final_liquidation",
                }
            )
            equity = final_equity
        equity_values[i] = equity
        previous_position = target

    equity_series = pd.Series(equity_values, index=frame.index, dtype=float, name="equity")
    returns = equity_series.pct_change().fillna(0.0)
    returns.iloc[0] = equity_series.iloc[0] / initial_capital - 1.0
    returns = returns.rename("returns")
    trades_frame = pd.DataFrame(trades) if trades else _empty_trades()
    orders_frame = pd.DataFrame(orders) if orders else _empty_orders()
    return returns, equity_series, trades_frame, orders_frame


def _empty_trades(*, candidate_id: bool = False) -> pd.DataFrame:
    columns = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "bars",
        "gross_return",
        "net_return",
        "reason",
    ]
    if candidate_id:
        columns = ["fold", "candidate_id", *columns]
    return pd.DataFrame(columns=columns)


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["time", "from_position", "to_position", "price", "cost", "reason"]
    )
