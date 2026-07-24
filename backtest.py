from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

try:  # keep compatibility with a future/alternate indicator module API
    from wek import compute_wek as _compute_wek
except ImportError:  # pragma: no cover - exercised by this repository's current API
    from wek import wek as _compute_wek


Variant = Literal["mean_reversion", "trend_filter", "long_short", "breakout"]
Side = Literal["long", "short"]

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
VARIANTS: tuple[Variant, ...] = ("mean_reversion", "trend_filter", "long_short", "breakout")
DEFAULT_LENGTHS = (14, 20, 30, 50)
DEFAULT_SMOOTHS = (3, 5, 8)
DEFAULT_THRESHOLDS = (40.0, 50.0, 60.0, 70.0)
DEFAULT_EXIT_BARS = (5, 10, 20)


@dataclass(frozen=True)
class BacktestConfig:
    variant: Variant
    threshold: float
    exit_bars: int
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    initial_capital: float = 1.0
    bars_per_year: float | None = None


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    config: BacktestConfig
    exposure: pd.Series


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    selected_train_results: pd.DataFrame
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_metrics: dict[str, float]
    oos_trades: pd.DataFrame
    fixed_oos_results: pd.DataFrame


@dataclass(frozen=True)
class _MarketFeatures:
    ema200: pd.Series
    prior_high: pd.Series
    prior_low: pd.Series


def run_backtest(
    df: pd.DataFrame,
    wek: pd.Series,
    variant: Variant,
    threshold: float,
    exit_bars: int,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    initial_capital: float = 1.0,
    bars_per_year: float | None = None,
) -> BacktestResult:
    """Run a leakage-safe event-time backtest.

    Signals are evaluated at close ``t`` and filled at open ``t+1``. Open
    positions left at the end of the sample are explicitly liquidated on the
    final close. Shorts are synthetic 1x notional shorts with no borrow or
    funding costs.
    """

    config = BacktestConfig(
        variant=variant,
        threshold=float(threshold),
        exit_bars=int(exit_bars),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        initial_capital=float(initial_capital),
        bars_per_year=bars_per_year,
    )
    return _run_backtest_engine(df=df, wek=wek, config=config)


def grid_search(
    df: pd.DataFrame,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    smooths: Sequence[int] = DEFAULT_SMOOTHS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    exit_bars_options: Sequence[int] = DEFAULT_EXIT_BARS,
    variants: Sequence[Variant] = VARIANTS,
    *,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    initial_capital: float = 1.0,
    bars_per_year: float | None = None,
    sort_by: str = "Sharpe",
) -> pd.DataFrame:
    """Evaluate a parameter grid, precomputing each ``(length, smooth)`` WEK once."""

    _validate_frame(df)
    wek_cache = {
        (int(length), int(smooth)): _compute_wek(df, length=int(length), smooth=int(smooth))
        for length, smooth in product(lengths, smooths)
    }
    return _grid_from_cache(
        df,
        wek_cache,
        features=_compute_market_features(df),
        thresholds=thresholds,
        exit_bars_options=exit_bars_options,
        variants=variants,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        initial_capital=initial_capital,
        bars_per_year=bars_per_year,
        sort_by=sort_by,
    )


def walk_forward(
    df: pd.DataFrame,
    *,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    smooths: Sequence[int] = DEFAULT_SMOOTHS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    exit_bars_options: Sequence[int] = DEFAULT_EXIT_BARS,
    variants: Sequence[Variant] = VARIANTS,
    train_months: int = 12,
    oos_months: int = 3,
    step_months: int = 3,
    objective: str = "Sharpe",
    min_trades: int = 1,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    initial_capital: float = 1.0,
    bars_per_year: float | None = None,
    include_fixed_oos: bool = False,
) -> WalkForwardResult:
    """Calendar walk-forward with train-only parameter selection.

    Indicator and market-feature values are computed on the full history,
    which is safe because they are causal; selection only sees rows strictly
    inside each train window. Only complete OOS calendar windows are emitted.
    """

    _validate_frame(df)
    if train_months <= 0 or oos_months <= 0 or step_months <= 0:
        raise ValueError("train_months, oos_months, and step_months must be positive")

    wek_cache = {
        (int(length), int(smooth)): _compute_wek(df, length=int(length), smooth=int(smooth))
        for length, smooth in product(lengths, smooths)
    }
    market_features = _compute_market_features(df)
    expected_bar_delta = _infer_bar_delta(df.index)

    fold_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    fixed_rows: list[dict[str, object]] = []
    oos_return_parts: list[pd.Series] = []
    oos_exposure_parts: list[pd.Series] = []
    oos_trade_parts: list[pd.DataFrame] = []

    train_start = df.index[0]
    fold_id = 0
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        oos_end = train_end + pd.DateOffset(months=oos_months)
        if expected_bar_delta is None or df.index[-1] + expected_bar_delta < oos_end:
            break
        train_mask = (df.index >= train_start) & (df.index < train_end)
        oos_mask = (df.index >= train_end) & (df.index < oos_end)
        if not train_mask.any() or not oos_mask.any():
            break

        train_results = _grid_from_cache(
            df.loc[train_mask],
            wek_cache,
            features=market_features,
            thresholds=thresholds,
            exit_bars_options=exit_bars_options,
            variants=variants,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            bars_per_year=bars_per_year,
            sort_by=objective,
        )
        if train_results.empty:
            train_start = train_start + pd.DateOffset(months=step_months)
            continue

        if objective not in train_results:
            raise ValueError(f"objective column not found: {objective}")
        finite_candidates = train_results[np.isfinite(train_results[objective])]
        eligible_candidates = finite_candidates[
            finite_candidates["trades"] >= int(min_trades)
        ]
        if not eligible_candidates.empty:
            selected = eligible_candidates.iloc[0]
            selection_fallback = "none"
        elif not finite_candidates.empty:
            selected = finite_candidates.iloc[0]
            selection_fallback = "min_trades_not_met"
        else:
            selected = train_results.iloc[0]
            selection_fallback = "no_finite_objective"
        selected_record = selected.to_frame().T
        selected_record.insert(0, "fold", fold_id)
        selected_record.insert(1, "selection_fallback", selection_fallback)
        selected_rows.append(selected_record)

        context_mask = oos_mask.copy()
        prev_positions = np.flatnonzero(df.index < train_end)
        if len(prev_positions):
            context_mask[prev_positions[-1]] = True
        key = (int(selected["length"]), int(selected["smooth"]))
        config = BacktestConfig(
            variant=selected["variant"],
            threshold=float(selected["threshold"]),
            exit_bars=int(selected["exit_bars"]),
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            bars_per_year=bars_per_year,
        )
        oos_result = _run_backtest_engine(
            df=df.loc[context_mask],
            wek=wek_cache[key],
            config=config,
            active_start=df.index[oos_mask][0],
            features=market_features,
        )
        fold_returns = oos_result.returns.copy()
        if not fold_returns.empty:
            oos_return_parts.append(fold_returns)
            oos_exposure_parts.append(oos_result.exposure)
        if not oos_result.trades.empty:
            trades = oos_result.trades.copy()
            trades.insert(0, "fold", fold_id)
            oos_trade_parts.append(trades)

        fold_rows.append(
            {
                "fold": fold_id,
                "selection_fallback": selection_fallback,
                "train_start": train_start,
                "train_end": train_end,
                "oos_start": df.index[oos_mask][0],
                "oos_end": df.index[oos_mask][-1],
                "length": int(selected["length"]),
                "smooth": int(selected["smooth"]),
                "variant": selected["variant"],
                "threshold": float(selected["threshold"]),
                "exit_bars": int(selected["exit_bars"]),
                f"train_{objective}": float(selected[objective]),
                "oos_total_return": oos_result.metrics["total_return"],
                "oos_Sharpe": oos_result.metrics["Sharpe"],
                "oos_trades": oos_result.metrics["trades"],
            }
        )

        if include_fixed_oos:
            fixed_rows.extend(
                _fixed_oos_rows(
                    fold_id,
                    df.loc[context_mask],
                    df.index[oos_mask][0],
                    wek_cache,
                    market_features,
                    train_results,
                    config,
                )
            )
        fold_id += 1
        train_start = train_start + pd.DateOffset(months=step_months)

    oos_returns = (
        pd.concat(oos_return_parts).sort_index()
        if oos_return_parts
        else pd.Series(dtype=float, name="returns")
    )
    if not oos_returns.empty:
        oos_equity = initial_capital * (1.0 + oos_returns).cumprod()
        oos_equity.name = "equity"
    else:
        oos_equity = pd.Series(dtype=float, name="equity")
    oos_trades = pd.concat(oos_trade_parts, ignore_index=True) if oos_trade_parts else _empty_trades()
    oos_exposure = (
        pd.concat(oos_exposure_parts).sort_index()
        if oos_exposure_parts
        else pd.Series(dtype=bool)
    )
    oos_metrics = _metrics(
        equity=oos_equity,
        returns=oos_returns,
        trades=oos_trades,
        exposure=oos_exposure,
        initial_capital=initial_capital,
        bars_per_year=bars_per_year,
    )

    return WalkForwardResult(
        folds=pd.DataFrame(fold_rows),
        selected_train_results=pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(),
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        oos_metrics=oos_metrics,
        oos_trades=oos_trades,
        fixed_oos_results=pd.DataFrame(fixed_rows),
    )


def _run_backtest_engine(
    *,
    df: pd.DataFrame,
    wek: pd.Series,
    config: BacktestConfig,
    active_start: pd.Timestamp | None = None,
    features: _MarketFeatures | None = None,
) -> BacktestResult:
    _validate_frame(df)
    _validate_config(config)
    raw_values = pd.Series(wek, index=wek.index, dtype=float, name="WEK")
    if raw_values.index.has_duplicates:
        raise ValueError("wek index must not contain duplicates")

    if not raw_values.index.is_monotonic_increasing:
        raw_values = raw_values.sort_index()
    values = raw_values.reindex(df.index)
    market_features = features if features is not None else _compute_market_features(df)

    open_values = df["open"].to_numpy(dtype=float, copy=False)
    close_values = df["close"].to_numpy(dtype=float, copy=False)
    wek_values = values.to_numpy(dtype=float, copy=False)
    ema200_values = _aligned_feature_values(market_features.ema200, df.index, "ema200")
    prior_high_values = _aligned_feature_values(market_features.prior_high, df.index, "prior_high")
    prior_low_values = _aligned_feature_values(market_features.prior_low, df.index, "prior_low")
    cross_up_values = (
        _crossover(raw_values, config.threshold)
        .reindex(df.index, fill_value=False)
        .fillna(False)
        .to_numpy(dtype=bool, copy=False)
    )
    cross_down_values = (
        _crossunder(raw_values, -config.threshold)
        .reindex(df.index, fill_value=False)
        .fillna(False)
        .to_numpy(dtype=bool, copy=False)
    )

    cash = config.initial_capital
    units = 0.0
    position: Side | None = None
    entry_idx: int | None = None
    entry_time: pd.Timestamp | None = None
    entry_price = np.nan
    entry_equity = np.nan
    pending: dict[str, object] | None = None
    equity_values = np.empty(len(df.index), dtype=float)
    exposure_values = np.empty(len(df.index), dtype=bool)
    trades: list[dict[str, object]] = []
    total_cost_rate = config.fee_rate + config.slippage_rate
    active_start = active_start if active_start is not None else df.index[0]
    active_values = np.asarray(df.index >= active_start, dtype=bool)

    for i, timestamp in enumerate(df.index):
        open_price = open_values[i]
        close_price = close_values[i]

        if pending is not None:
            action = str(pending["action"])
            reason = str(pending["reason"])
            if action in {"exit", "reverse_long", "reverse_short"} and position is not None:
                cash, units, trade = _exit_position(
                    cash=cash,
                    units=units,
                    price=open_price,
                    cost_rate=total_cost_rate,
                    timestamp=timestamp,
                    side=position,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    entry_equity=entry_equity,
                    entry_idx=entry_idx,
                    exit_idx=i,
                    reason=reason,
                )
                trades.append(trade)
                position = None
                entry_idx = None
                entry_time = None
            if action in {"enter_long", "reverse_long"}:
                cash, units, position, entry_idx, entry_time, entry_price, entry_equity = _enter_position(
                    cash=cash,
                    units=units,
                    side="long",
                    price=open_price,
                    cost_rate=total_cost_rate,
                    timestamp=timestamp,
                    index_position=i,
                )
            elif action in {"enter_short", "reverse_short"}:
                cash, units, position, entry_idx, entry_time, entry_price, entry_equity = _enter_position(
                    cash=cash,
                    units=units,
                    side="short",
                    price=open_price,
                    cost_rate=total_cost_rate,
                    timestamp=timestamp,
                    index_position=i,
                )
            pending = None

        equity_at_close = cash + units * close_price
        equity_values[i] = equity_at_close
        exposure_values[i] = position is not None and active_values[i]

        if i == len(df.index) - 1 or not active_values[i + 1]:
            continue

        held_bars = (i - entry_idx + 1) if entry_idx is not None else 0
        pending = _next_order(
            variant=config.variant,
            position=position,
            held_bars=held_bars,
            exit_bars=config.exit_bars,
            wek_value=wek_values[i],
            close_value=close_price,
            ema200_value=ema200_values[i],
            prior_high_value=prior_high_values[i],
            prior_low_value=prior_low_values[i],
            cross_up=cross_up_values[i],
            cross_down=cross_down_values[i],
            threshold=config.threshold,
        )

    if position is not None:
        final_timestamp = df.index[-1]
        final_price = close_values[-1]
        cash, units, trade = _exit_position(
            cash=cash,
            units=units,
            price=final_price,
            cost_rate=total_cost_rate,
            timestamp=final_timestamp,
            side=position,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_equity=entry_equity,
            entry_idx=entry_idx,
            exit_idx=len(df.index) - 1,
            reason="final_liquidation",
        )
        trades.append(trade)
        equity_values[-1] = cash

    equity = pd.Series(equity_values, index=df.index, dtype=float, name="equity")
    exposure = pd.Series(exposure_values, index=df.index, dtype=bool)
    trades_frame = pd.DataFrame(trades) if trades else _empty_trades()

    if active_start != df.index[0]:
        equity = equity.loc[equity.index >= active_start]
        exposure = exposure.loc[exposure.index >= active_start]
        if not trades_frame.empty:
            trades_frame = trades_frame[trades_frame["entry_time"] >= active_start].reset_index(drop=True)

    returns = equity.pct_change().fillna(0.0)
    if not returns.empty:
        returns.iloc[0] = equity.iloc[0] / config.initial_capital - 1.0
    returns.name = "returns"
    metrics = _metrics(
        equity=equity,
        returns=returns,
        trades=trades_frame,
        exposure=exposure,
        initial_capital=config.initial_capital,
        bars_per_year=config.bars_per_year,
    )
    return BacktestResult(
        equity=equity,
        returns=returns,
        trades=trades_frame,
        metrics=metrics,
        config=config,
        exposure=exposure,
    )


def _next_order(
    *,
    variant: Variant,
    position: Side | None,
    held_bars: int,
    exit_bars: int,
    wek_value: float,
    close_value: float,
    ema200_value: float,
    prior_high_value: float,
    prior_low_value: float,
    cross_up: bool,
    cross_down: bool,
    threshold: float,
) -> dict[str, object] | None:
    time_exit = position is not None and held_bars >= exit_bars
    if variant == "mean_reversion":
        if position == "long":
            if pd.notna(wek_value) and wek_value <= 0:
                return {"action": "exit", "reason": "wek_zero"}
            if time_exit:
                return {"action": "exit", "reason": "time"}
        elif cross_up:
            return {"action": "enter_long", "reason": "cross_above_threshold"}
        return None

    if variant == "trend_filter":
        if position == "long":
            if pd.notna(wek_value) and wek_value <= 0:
                return {"action": "exit", "reason": "wek_zero"}
            if time_exit:
                return {"action": "exit", "reason": "time"}
        elif cross_up and pd.notna(ema200_value) and close_value > ema200_value:
            return {"action": "enter_long", "reason": "cross_above_threshold_ema200"}
        return None

    if variant == "long_short":
        if position == "long":
            if cross_down:
                return {"action": "reverse_short", "reason": "reverse_short"}
            if pd.notna(wek_value) and wek_value <= 0:
                return {"action": "exit", "reason": "wek_zero"}
            if time_exit:
                return {"action": "exit", "reason": "time"}
        elif position == "short":
            if cross_up:
                return {"action": "reverse_long", "reason": "reverse_long"}
            if pd.notna(wek_value) and wek_value >= 0:
                return {"action": "exit", "reason": "wek_zero"}
            if time_exit:
                return {"action": "exit", "reason": "time"}
        else:
            if cross_up:
                return {"action": "enter_long", "reason": "cross_above_threshold"}
            if cross_down:
                return {"action": "enter_short", "reason": "cross_below_threshold"}
        return None

    if variant == "breakout":
        if position == "long":
            if pd.notna(prior_low_value) and close_value < prior_low_value:
                return {"action": "exit", "reason": "donchian_exit"}
            if time_exit:
                return {"action": "exit", "reason": "time"}
        elif (
            pd.notna(prior_high_value)
            and pd.notna(wek_value)
            and close_value > prior_high_value
            and wek_value > threshold
        ):
            return {"action": "enter_long", "reason": "donchian_breakout"}
        return None

    raise ValueError(f"unknown variant: {variant}")


def _enter_position(
    *,
    cash: float,
    units: float,
    side: Side,
    price: float,
    cost_rate: float,
    timestamp: pd.Timestamp,
    index_position: int,
) -> tuple[float, float, Side, int, pd.Timestamp, float, float]:
    if price <= 0 or not np.isfinite(price):
        raise ValueError("execution price must be positive and finite")
    equity_before = cash + units * price
    notional = equity_before
    cost = abs(notional) * cost_rate
    if side == "long":
        units += notional / price
        cash -= notional + cost
    else:
        units -= notional / price
        cash += notional - cost
    return cash, units, side, index_position, timestamp, price, equity_before


def _exit_position(
    *,
    cash: float,
    units: float,
    price: float,
    cost_rate: float,
    timestamp: pd.Timestamp,
    side: Side,
    entry_time: pd.Timestamp | None,
    entry_price: float,
    entry_equity: float,
    entry_idx: int | None,
    exit_idx: int,
    reason: str,
) -> tuple[float, float, dict[str, object]]:
    if price <= 0 or not np.isfinite(price):
        raise ValueError("execution price must be positive and finite")
    if entry_time is None or entry_idx is None:
        raise RuntimeError("cannot exit without an entry")
    notional = abs(units) * price
    cost = notional * cost_rate
    cash += units * price - cost
    equity_after = cash
    gross_return = (price / entry_price - 1.0) if side == "long" else -(price / entry_price - 1.0)
    net_return = equity_after / entry_equity - 1.0
    trade = {
        "entry_time": entry_time,
        "exit_time": timestamp,
        "side": side,
        "entry_price": float(entry_price),
        "exit_price": float(price),
        "bars": int(max(exit_idx - entry_idx, 1)),
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "reason": reason,
    }
    return cash, 0.0, trade


def _metrics(
    *,
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    exposure: pd.Series,
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
    bpy = float(bars_per_year) if bars_per_year is not None else _infer_bars_per_year(equity.index)
    periods = max(len(equity), 1)
    cagr = np.nan
    if equity.iloc[-1] > 0 and initial_capital > 0:
        cagr = float((equity.iloc[-1] / initial_capital) ** (bpy / periods) - 1.0)

    mean_return = returns.mean()
    std_return = returns.std(ddof=0)
    sharpe = float(mean_return / std_return * np.sqrt(bpy)) if std_return > 0 else np.nan
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = float(mean_return / downside_std * np.sqrt(bpy)) if downside_std > 0 else np.nan
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
        trade_count = int(len(net))

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


def _grid_from_cache(
    df: pd.DataFrame,
    wek_cache: dict[tuple[int, int], pd.Series],
    *,
    features: _MarketFeatures | None = None,
    thresholds: Sequence[float],
    exit_bars_options: Sequence[int],
    variants: Sequence[Variant],
    fee_rate: float,
    slippage_rate: float,
    initial_capital: float,
    bars_per_year: float | None,
    sort_by: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    market_features = features if features is not None else _compute_market_features(df)
    for (length, smooth), wek_values in wek_cache.items():
        for variant, threshold, exit_bars in product(variants, thresholds, exit_bars_options):
            config = BacktestConfig(
                variant=variant,
                threshold=float(threshold),
                exit_bars=int(exit_bars),
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                initial_capital=initial_capital,
                bars_per_year=bars_per_year,
            )
            result = _run_backtest_engine(
                df=df,
                wek=wek_values,
                config=config,
                features=market_features,
            )
            rows.append(
                {
                    "length": length,
                    "smooth": smooth,
                    "variant": variant,
                    "threshold": float(threshold),
                    "exit_bars": int(exit_bars),
                    **result.metrics,
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty and sort_by in output:
        output = output.sort_values(
            sort_by,
            ascending=False,
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return output


def _fixed_oos_rows(
    fold_id: int,
    context_df: pd.DataFrame,
    active_start: pd.Timestamp,
    wek_cache: dict[tuple[int, int], pd.Series],
    features: _MarketFeatures,
    train_results: pd.DataFrame,
    selected_config: BacktestConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if context_df.empty:
        return rows
    for _, candidate in train_results.iterrows():
        key = (int(candidate["length"]), int(candidate["smooth"]))
        config = BacktestConfig(
            variant=candidate["variant"],
            threshold=float(candidate["threshold"]),
            exit_bars=int(candidate["exit_bars"]),
            fee_rate=selected_config.fee_rate,
            slippage_rate=selected_config.slippage_rate,
            initial_capital=selected_config.initial_capital,
            bars_per_year=selected_config.bars_per_year,
        )
        result = _run_backtest_engine(
            df=context_df,
            wek=wek_cache[key],
            config=config,
            active_start=active_start,
            features=features,
        )
        rows.append(
            {
                "fold": fold_id,
                "length": key[0],
                "smooth": key[1],
                "variant": config.variant,
                "threshold": config.threshold,
                "exit_bars": config.exit_bars,
                **{f"oos_{k}": v for k, v in result.metrics.items()},
            }
        )
    return rows


def _compute_market_features(df: pd.DataFrame) -> _MarketFeatures:
    return _MarketFeatures(
        ema200=df["close"].ewm(span=200, min_periods=200, adjust=False).mean(),
        prior_high=df["high"].shift(1).rolling(20, min_periods=20).max(),
        prior_low=df["low"].shift(1).rolling(20, min_periods=20).min(),
    )


def _aligned_feature_values(series: pd.Series, index: pd.DatetimeIndex, name: str) -> np.ndarray:
    if series.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")
    missing = index.difference(series.index)
    if len(missing):
        raise ValueError(f"{name} is missing {len(missing)} required rows")
    return series.reindex(index).to_numpy(dtype=float, copy=False)


def _validate_frame(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must use a UTC DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("df must use a timezone-aware UTC DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be monotonic increasing")
    if df.index.has_duplicates:
        raise ValueError("df index must not contain duplicates")


def _validate_config(config: BacktestConfig) -> None:
    if config.variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if config.threshold <= 0:
        raise ValueError("threshold must be positive")
    if config.exit_bars <= 0:
        raise ValueError("exit_bars must be positive")
    if config.fee_rate < 0 or config.slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if config.bars_per_year is not None and config.bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")


def _crossover(series: pd.Series, threshold: float) -> pd.Series:
    return (series > threshold) & (series.shift(1) <= threshold)


def _crossunder(series: pd.Series, threshold: float) -> pd.Series:
    return (series < threshold) & (series.shift(1) >= threshold)


def _infer_bars_per_year(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 365.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = deltas[deltas > 0].median()
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        return 365.0
    return float(365.0 * 24.0 * 60.0 * 60.0 / median_seconds)


def _infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return None
    return positive.median()


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


__all__: Iterable[str] = (
    "BacktestConfig",
    "BacktestResult",
    "WalkForwardResult",
    "run_backtest",
    "grid_search",
    "walk_forward",
)
