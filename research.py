from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
from data_pipeline import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, load_cached_data
from wek import pine_ema, wek as compute_wek


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = PROJECT_ROOT / "charts"
REPORT_PATH = PROJECT_ROOT / "report.md"
SEED = 42
TOTAL_COST_PER_SIDE = 0.0015
OBJECTIVE = "Sharpe"


@dataclass(frozen=True)
class ResearchConfig:
    lengths: tuple[int, ...] = (14, 20, 30, 50)
    smooths: tuple[int, ...] = (3, 5, 8)
    thresholds: tuple[float, ...] = (40.0, 50.0, 60.0, 70.0)
    exit_bars_options: tuple[int, ...] = (5, 10, 20)
    variants: tuple[str, ...] = ("mean_reversion", "trend_filter", "long_short", "breakout")
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    train_months: int = 12
    oos_months: int = 3
    step_months: int = 3
    seed: int = SEED
    mc_permutations: int = 1000
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES
    quick: bool = False
    max_datasets: int | None = None

    @classmethod
    def quick_config(cls) -> "ResearchConfig":
        return cls(
            lengths=(14, 20),
            smooths=(3,),
            thresholds=(50.0,),
            exit_bars_options=(5,),
            variants=("mean_reversion", "long_short"),
            mc_permutations=40,
            max_datasets=1,
            quick=True,
        )


@dataclass
class DatasetResult:
    dataset: str
    symbol: str
    timeframe: str
    rows: int
    start: str
    end: str
    min_trades_required: int
    fallback_folds: int
    fold_count: int
    power_warning: str
    credible_edge: bool
    wfo_metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    best_fixed: dict[str, object] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def dataset_id(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '_').lower()}_{timeframe}"


def bars_per_year_for_timeframe(timeframe: str) -> float:
    if timeframe == "1h":
        return 365.0 * 24.0
    if timeframe == "4h":
        return 365.0 * 6.0
    if timeframe == "1d":
        return 365.0
    raise ValueError(f"unsupported timeframe: {timeframe}")


def min_trades_for_timeframe(timeframe: str) -> int:
    return 8 if timeframe == "1d" else 20


def run_study(config: ResearchConfig | None = None) -> dict[str, object]:
    config = config or ResearchConfig()
    RESULTS_DIR.mkdir(exist_ok=True)
    CHARTS_DIR.mkdir(exist_ok=True)

    manifest: dict[str, object] = {
        "seed": config.seed,
        "objective": OBJECTIVE,
        "costs": {
            "fee_rate": config.fee_rate,
            "slippage_rate": config.slippage_rate,
            "per_side": config.fee_rate + config.slippage_rate,
        },
        "walk_forward": {
            "train_months": config.train_months,
            "oos_months": config.oos_months,
            "step_months": config.step_months,
            "min_train_trades": "20 for 1h/4h, 8 for 1d; fallback recorded per fold",
        },
        "grid": _jsonable(asdict(config)),
        "datasets": [],
    }

    dataset_results: list[DatasetResult] = []
    fixed_all: list[pd.DataFrame] = []
    equity_all: list[pd.DataFrame] = []
    selection_all: list[pd.DataFrame] = []

    tasks = list(product(config.symbols, config.timeframes))
    if config.max_datasets is not None:
        tasks = tasks[: config.max_datasets]

    for symbol, timeframe in tasks:
        result, frames = run_dataset(symbol, timeframe, config=config)
        dataset_results.append(result)
        manifest["datasets"].append(asdict(result))
        if not frames["fixed"].empty:
            fixed_all.append(frames["fixed"])
        if not frames["equity"].empty:
            equity_all.append(frames["equity"])
        if not frames["selected"].empty:
            selection_all.append(frames["selected"])
        _write_json(RESULTS_DIR / f"checkpoint_{result.dataset}.json", asdict(result))

    leaderboard = build_leaderboard(dataset_results)
    leaderboard.to_csv(RESULTS_DIR / "aggregate_leaderboard.csv", index=False)
    fixed_combined = pd.concat(fixed_all, ignore_index=True) if fixed_all else pd.DataFrame()
    final_config = select_final_config(fixed_combined, config=config)
    final_outputs = postprocess_final_strategy(final_config, config) if final_config else _empty_final_outputs(config)
    final_config = final_outputs["final_config"]
    final_result = final_outputs["result"]
    mc = final_outputs["monte_carlo"]
    mc_summary = final_outputs["monte_carlo_summary"]

    ablation = run_ablation(final_config, config, full_result=final_result) if final_config else pd.DataFrame()
    ablation.to_csv(RESULTS_DIR / "ablation.csv", index=False)
    selected_summary = build_is_summary(pd.concat(selection_all, ignore_index=True) if selection_all else pd.DataFrame())
    selected_summary.to_csv(RESULTS_DIR / "selected_is_summary.csv", index=False)

    _write_json(RESULTS_DIR / "assumptions_manifest.json", manifest)
    create_charts(
        equity=pd.concat(equity_all, ignore_index=True) if equity_all else pd.DataFrame(),
        fixed=fixed_combined,
        selected=pd.concat(selection_all, ignore_index=True) if selection_all else pd.DataFrame(),
        monte_carlo=mc,
        final_config=final_config,
    )
    write_report(leaderboard, dataset_results, final_config, ablation, mc, selected_summary, mc_summary)
    return {"leaderboard": leaderboard, "final_config": final_config, "manifest": manifest}


def run_dataset(symbol: str, timeframe: str, *, config: ResearchConfig) -> tuple[DatasetResult, dict[str, pd.DataFrame]]:
    did = dataset_id(symbol, timeframe)
    bars_per_year = bars_per_year_for_timeframe(timeframe)
    min_trades = min_trades_for_timeframe(timeframe)
    frame = load_cached_data(symbol, timeframe, validate_coverage=False)

    grid = backtest.grid_search(
        frame,
        lengths=config.lengths,
        smooths=config.smooths,
        thresholds=config.thresholds,
        exit_bars_options=config.exit_bars_options,
        variants=config.variants,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        bars_per_year=bars_per_year,
        sort_by=OBJECTIVE,
    )
    grid.insert(0, "dataset", did)
    grid.insert(1, "symbol", symbol)
    grid.insert(2, "timeframe", timeframe)
    grid_path = RESULTS_DIR / f"{did}_full_grid.csv"
    grid.to_csv(grid_path, index=False)

    wfo = backtest.walk_forward(
        frame,
        lengths=config.lengths,
        smooths=config.smooths,
        thresholds=config.thresholds,
        exit_bars_options=config.exit_bars_options,
        variants=config.variants,
        train_months=config.train_months,
        oos_months=config.oos_months,
        step_months=config.step_months,
        objective=OBJECTIVE,
        min_trades=min_trades,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        bars_per_year=bars_per_year,
        include_fixed_oos=True,
    )
    folds = wfo.folds.copy()
    selected = wfo.selected_train_results.copy()
    fixed = wfo.fixed_oos_results.copy()
    for table in (folds, selected, fixed):
        if not table.empty:
            table.insert(0, "dataset", did)
            table.insert(1, "symbol", symbol)
            table.insert(2, "timeframe", timeframe)

    if not folds.empty:
        folds["min_trades_required"] = min_trades
        selected_trade_counts = selected["trades"].astype(float).to_numpy() if not selected.empty else np.array([])
        folds["min_trade_fallback"] = False
        if len(selected_trade_counts) == len(folds):
            folds["min_trade_fallback"] = selected_trade_counts < min_trades
    if not selected.empty:
        selected["min_trades_required"] = min_trades
        selected["min_trade_fallback"] = selected["trades"].astype(float) < min_trades
    if not fixed.empty:
        fixed["min_trades_required"] = min_trades

    cost_rate = config.fee_rate + config.slippage_rate
    benchmark_equity = buy_and_hold_benchmark(frame, wfo.oos_equity.index, cost_rate=cost_rate)
    benchmark_returns = equity_returns(benchmark_equity)
    benchmark_metrics = metric_summary(benchmark_equity, benchmark_returns, bars_per_year=bars_per_year)
    oos_equity = pd.DataFrame(
        {
            "timestamp": wfo.oos_equity.index,
            "dataset": did,
            "symbol": symbol,
            "timeframe": timeframe,
            "selected_equity": wfo.oos_equity.to_numpy(dtype=float, copy=False),
            "selected_returns": wfo.oos_returns.reindex(wfo.oos_equity.index).to_numpy(dtype=float, copy=False),
            "benchmark_equity": benchmark_equity.reindex(wfo.oos_equity.index).to_numpy(dtype=float, copy=False),
            "benchmark_returns": benchmark_returns.reindex(wfo.oos_equity.index).to_numpy(dtype=float, copy=False),
        }
    )

    paths = {
        "grid": str(grid_path.relative_to(PROJECT_ROOT)),
        "folds": str((RESULTS_DIR / f"{did}_folds.csv").relative_to(PROJECT_ROOT)),
        "selected_is": str((RESULTS_DIR / f"{did}_selected_is.csv").relative_to(PROJECT_ROOT)),
        "oos_trades": str((RESULTS_DIR / f"{did}_oos_trades.csv").relative_to(PROJECT_ROOT)),
        "oos_equity": str((RESULTS_DIR / f"{did}_oos_equity.csv").relative_to(PROJECT_ROOT)),
        "fixed_oos": str((RESULTS_DIR / f"{did}_fixed_oos_stability.csv").relative_to(PROJECT_ROOT)),
    }
    folds.to_csv(PROJECT_ROOT / paths["folds"], index=False)
    selected.to_csv(PROJECT_ROOT / paths["selected_is"], index=False)
    trades = wfo.oos_trades.copy()
    if not trades.empty:
        trades.insert(0, "dataset", did)
        trades.insert(1, "symbol", symbol)
        trades.insert(2, "timeframe", timeframe)
    trades.to_csv(PROJECT_ROOT / paths["oos_trades"], index=False)
    oos_equity.to_csv(PROJECT_ROOT / paths["oos_equity"], index=False)
    fixed.to_csv(PROJECT_ROOT / paths["fixed_oos"], index=False)

    oos_metrics = _float_dict(wfo.oos_metrics)
    benchmark_metrics = _float_dict(benchmark_metrics)
    credible_edge = has_credible_edge(oos_metrics, benchmark_metrics, min_trades)
    power_warning = power_warning_for_dataset(timeframe, oos_metrics, selected, min_trades)

    dataset_result = DatasetResult(
        dataset=did,
        symbol=symbol,
        timeframe=timeframe,
        rows=len(frame),
        start=str(frame.index.min()),
        end=str(frame.index.max()),
        min_trades_required=min_trades,
        fallback_folds=int(selected["min_trade_fallback"].sum()) if "min_trade_fallback" in selected else 0,
        fold_count=int(len(folds)),
        power_warning=power_warning,
        credible_edge=credible_edge,
        wfo_metrics=oos_metrics,
        benchmark_metrics=benchmark_metrics,
        best_fixed=aggregate_fixed_candidates(fixed).iloc[0].to_dict() if not fixed.empty else {},
        files=paths,
    )
    return dataset_result, {"fixed": fixed, "equity": oos_equity, "selected": selected, "trades": trades}


def buy_and_hold_benchmark(df: pd.DataFrame, timestamps: pd.Index, *, cost_rate: float) -> pd.Series:
    if len(timestamps) == 0:
        return pd.Series(dtype=float, name="benchmark_equity")
    prices = df.reindex(timestamps)
    if prices[["open", "close"]].isna().any().any():
        prices = df.loc[df.index.intersection(timestamps)]
    if prices.empty:
        return pd.Series(dtype=float, name="benchmark_equity")
    entry_open = float(prices["open"].iloc[0])
    units = (1.0 * (1.0 - cost_rate)) / entry_open
    equity = units * prices["close"].astype(float)
    equity.iloc[-1] = equity.iloc[-1] * (1.0 - cost_rate)
    equity.name = "benchmark_equity"
    return equity


def equity_returns(equity: pd.Series) -> pd.Series:
    returns = equity.pct_change().fillna(0.0)
    if not equity.empty:
        returns.iloc[0] = float(equity.iloc[0]) - 1.0
    returns.name = "returns"
    return returns


def metric_summary(equity: pd.Series, returns: pd.Series, *, bars_per_year: float) -> dict[str, float]:
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
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    periods = max(len(equity), 1)
    cagr = float(equity.iloc[-1] ** (bars_per_year / periods) - 1.0) if equity.iloc[-1] > 0 else np.nan
    std = returns.std(ddof=0)
    sharpe = float(returns.mean() / std * math.sqrt(bars_per_year)) if std > 0 else np.nan
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = float(returns.mean() / downside_std * math.sqrt(bars_per_year)) if downside_std > 0 else np.nan
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "trades": 1.0,
        "exposure": 1.0,
    }


def has_credible_edge(oos_metrics: dict[str, float], benchmark_metrics: dict[str, float], min_trades: int) -> bool:
    total_return = oos_metrics.get("total_return", np.nan)
    sharpe = oos_metrics.get("Sharpe", np.nan)
    trades = oos_metrics.get("trades", 0.0)
    benchmark_return = benchmark_metrics.get("total_return", np.nan)
    benchmark_sharpe = benchmark_metrics.get("Sharpe", np.nan)
    return bool(
        np.isfinite(total_return)
        and np.isfinite(sharpe)
        and np.isfinite(benchmark_return)
        and np.isfinite(benchmark_sharpe)
        and total_return > 0
        and trades >= min_trades
        and total_return > benchmark_return
        and sharpe > benchmark_sharpe
    )


def power_warning_for_dataset(
    timeframe: str,
    oos_metrics: dict[str, float],
    selected: pd.DataFrame,
    min_trades: int,
) -> str:
    trades = float(oos_metrics.get("trades", 0.0) or 0.0)
    fallback_folds = int(selected["min_trade_fallback"].sum()) if "min_trade_fallback" in selected else 0
    warnings = []
    if trades < min_trades:
        warnings.append(f"underpowered stitched OOS: {trades:.0f} trades < required {min_trades}")
    if fallback_folds:
        warnings.append(f"{fallback_folds} folds used min-trade fallback")
    if timeframe == "1d" and trades < min_trades:
        warnings.append("1d/no-signal underpower")
    return "; ".join(warnings) if warnings else "adequate by predeclared trade-count rule"


def aggregate_fixed_candidates(fixed: pd.DataFrame) -> pd.DataFrame:
    if fixed.empty:
        return pd.DataFrame()
    keys = ["dataset", "symbol", "timeframe", "length", "smooth", "variant", "threshold", "exit_bars"]
    grouped = (
        fixed.groupby(keys, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mean_oos_Sharpe=("oos_Sharpe", "mean"),
            median_oos_Sharpe=("oos_Sharpe", "median"),
            positive_fold_share=("oos_Sharpe", lambda s: float((s.astype(float) > 0).mean())),
            mean_oos_return=("oos_total_return", "mean"),
            total_oos_trades=("oos_trades", "sum"),
            worst_oos_drawdown=("oos_max_drawdown", "min"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["mean_oos_Sharpe", "positive_fold_share", "median_oos_Sharpe"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def stability_neighbor_diagnostic(grid: pd.DataFrame, candidate: dict[str, object]) -> dict[str, object]:
    if grid.empty:
        return {"neighbor_count": 0, "positive_sharpe_share": np.nan, "median_neighbor_sharpe": np.nan, "overfit": True}
    metric = "mean_oos_Sharpe" if "mean_oos_Sharpe" in grid.columns else "oos_Sharpe" if "oos_Sharpe" in grid.columns else "Sharpe"
    dimensions = ["length", "smooth", "threshold", "exit_bars"]
    same_variant = grid[grid["variant"] == candidate["variant"]].copy()
    if "dataset" in same_variant and "dataset" in candidate:
        same_variant = same_variant[same_variant["dataset"] == candidate["dataset"]]
    masks = []
    for dim in dimensions:
        values = sorted(same_variant[dim].dropna().unique())
        if candidate[dim] not in values:
            continue
        idx = values.index(candidate[dim])
        adjacent = set()
        if idx > 0:
            adjacent.add(values[idx - 1])
        if idx < len(values) - 1:
            adjacent.add(values[idx + 1])
        if not adjacent:
            continue
        mask = same_variant[dim].isin(adjacent)
        for other in dimensions:
            if other != dim:
                mask &= same_variant[other] == candidate[other]
        masks.append(mask)
    neighbors = same_variant[np.logical_or.reduce(masks)] if masks else same_variant.iloc[0:0]
    sharpe = pd.to_numeric(neighbors[metric], errors="coerce").dropna()
    share = float((sharpe > 0).mean()) if len(sharpe) else np.nan
    median = float(sharpe.median()) if len(sharpe) else np.nan
    overfit = bool(len(sharpe) == 0 or share < 0.5 or median <= 0.0)
    return {
        "neighbor_count": int(len(sharpe)),
        "positive_sharpe_share": share,
        "median_neighbor_sharpe": median,
        "overfit": overfit,
    }


def select_final_config(fixed: pd.DataFrame, config: ResearchConfig | None = None) -> dict[str, object]:
    aggregate = aggregate_fixed_candidates(fixed)
    if aggregate.empty:
        return {}
    aggregate["min_total_oos_trades_required"] = aggregate["timeframe"].map(min_trades_for_timeframe).astype(float)
    qualified = aggregate[aggregate["total_oos_trades"].astype(float) >= aggregate["min_total_oos_trades_required"]]
    underpowered_fallback = qualified.empty
    selection_pool = qualified if not qualified.empty else aggregate
    candidates = selection_pool.head(min(len(selection_pool), 40)).copy()
    diagnostics = [stability_neighbor_diagnostic(aggregate, row.to_dict()) for _, row in candidates.iterrows()]
    diag = pd.DataFrame(diagnostics)
    candidates = pd.concat([candidates.reset_index(drop=True), diag], axis=1)
    candidates["robust_score"] = (
        candidates["mean_oos_Sharpe"].fillna(-999.0)
        + candidates["positive_sharpe_share"].fillna(0.0)
        + candidates["median_neighbor_sharpe"].fillna(-1.0).clip(lower=-1.0, upper=1.0)
    )
    robust = candidates[(~candidates["overfit"]) & (candidates["mean_oos_Sharpe"] > 0)]
    selected = (robust if not robust.empty else candidates).sort_values(
        ["robust_score", "mean_oos_Sharpe"], ascending=False
    ).iloc[0]
    output = _jsonable(selected.to_dict())
    output["fee_rate"] = config.fee_rate if config is not None else 0.001
    output["slippage_rate"] = config.slippage_rate if config is not None else 0.0005
    output["tick_size"] = 1e-10
    output["underpowered_fallback"] = underpowered_fallback
    output["flat_bar_tick_note"] = "tick_size=1e-10 matches WEK floor; cached exchange data has no special flat-bar tick adjustment, so exchange tick is inconsequential here."
    output["selection_note"] = "Wybrane przez diagnostyke OOS/meta-selection; nie jest to nietkniete potwierdzenie."
    return output


def construct_ablated_wek(
    frame: pd.DataFrame,
    *,
    length: int,
    smooth: int,
    remove_entropy: bool = False,
    remove_conviction: bool = False,
    remove_wick: bool = False,
) -> pd.Series:
    _, components = compute_wek(frame, length=length, smooth=smooth, return_components=True)
    wick_signal = components["wick_sig"] * (0.0 if remove_wick else 1.0)
    conviction_multiplier = 1.0 if remove_conviction else (0.5 + components["conv_sig"])
    entropy_multiplier = 1.0 if remove_entropy else (0.3 + components["event_factor"] * 1.4)
    raw = wick_signal * conviction_multiplier * entropy_multiplier
    ablated = pine_ema(raw * 100.0, smooth).clip(lower=-100.0, upper=100.0)
    ablated.name = "WEK"
    return ablated


def run_final_strategy(
    final_config: dict[str, object],
    config: ResearchConfig,
    *,
    frame: pd.DataFrame | None = None,
) -> backtest.BacktestResult:
    """Evaluate the exact fixed final candidate from the common OOS start."""
    if not final_config:
        raise ValueError("final_config must not be empty")
    symbol = str(final_config["symbol"])
    timeframe = str(final_config["timeframe"])
    evaluation_frame = frame if frame is not None else load_cached_data(symbol, timeframe, validate_coverage=False)
    if evaluation_frame.empty:
        raise ValueError("cannot evaluate final strategy on an empty frame")
    oos_start = evaluation_frame.index[0] + pd.DateOffset(months=config.train_months)
    wek_series = compute_wek(
        evaluation_frame,
        length=int(final_config["length"]),
        smooth=int(final_config["smooth"]),
        tick_size=float(final_config.get("tick_size", 1e-10)),
    )
    return run_backtest_with_active_start(
        evaluation_frame,
        wek_series,
        active_start=oos_start,
        variant=str(final_config["variant"]),
        threshold=float(final_config["threshold"]),
        exit_bars=int(final_config["exit_bars"]),
        fee_rate=float(final_config.get("fee_rate", config.fee_rate)),
        slippage_rate=float(final_config.get("slippage_rate", config.slippage_rate)),
        bars_per_year=bars_per_year_for_timeframe(timeframe),
    )


def postprocess_final_strategy(
    final_config: dict[str, object],
    config: ResearchConfig,
    *,
    result: backtest.BacktestResult | None = None,
) -> dict[str, object]:
    """Persist one fixed evaluation and derive Monte Carlo only from its trades."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not final_config:
        return _empty_final_outputs(config)
    fixed_result = result if result is not None else run_final_strategy(final_config, config)
    trades = fixed_result.trades.copy()
    metadata = {
        "dataset": str(final_config["dataset"]),
        "symbol": str(final_config["symbol"]),
        "timeframe": str(final_config["timeframe"]),
    }
    for column, value in reversed(tuple(metadata.items())):
        if column not in trades:
            trades.insert(0, column, value)

    equity = pd.DataFrame(
        {
            "timestamp": fixed_result.equity.index,
            "dataset": [metadata["dataset"]] * len(fixed_result.equity),
            "symbol": [metadata["symbol"]] * len(fixed_result.equity),
            "timeframe": [metadata["timeframe"]] * len(fixed_result.equity),
            "returns": fixed_result.returns.reindex(fixed_result.equity.index).to_numpy(dtype=float),
            "equity": fixed_result.equity.to_numpy(dtype=float),
        }
    )
    trades_path = RESULTS_DIR / "final_strategy_trades.csv"
    equity_path = RESULTS_DIR / "final_strategy_equity.csv"
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)

    trade_returns = pd.to_numeric(trades.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
    mc = monte_carlo_permutations(trade_returns, n=config.mc_permutations, seed=config.seed)
    mc_summary = monte_carlo_summary(mc, trade_returns)
    mc.to_csv(RESULTS_DIR / "monte_carlo_1000_permutation.csv", index=False)
    mc_summary.to_csv(RESULTS_DIR / "monte_carlo_summary.csv", index=False)

    oos_start = fixed_result.equity.index[0].isoformat() if not fixed_result.equity.empty else None
    trade_compound_return = float(np.prod(1.0 + np.asarray(trade_returns)) - 1.0) if trade_returns else 0.0
    enriched_config = dict(final_config)
    enriched_config["fixed_evaluation"] = {
        "oos_start": oos_start,
        "metrics": _jsonable(fixed_result.metrics),
        "trade_count": int(len(trade_returns)),
        "trade_compound_return": trade_compound_return,
        "trades_path": _display_path(trades_path),
        "equity_path": _display_path(equity_path),
    }
    enriched_config["monte_carlo"] = {
        "source": _display_path(trades_path),
        "source_column": "net_return",
        "method": "fixed-final-candidate trade-order permutation",
        "permutations": int(config.mc_permutations),
        "seed": int(config.seed),
    }
    _write_json(RESULTS_DIR / "final_config.json", enriched_config)
    return {
        "final_config": enriched_config,
        "result": fixed_result,
        "trades": trades,
        "equity": equity,
        "monte_carlo": mc,
        "monte_carlo_summary": mc_summary,
    }


def run_ablation(
    final_config: dict[str, object],
    config: ResearchConfig,
    *,
    full_result: backtest.BacktestResult | None = None,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not final_config:
        return pd.DataFrame()
    symbol = str(final_config["symbol"])
    timeframe = str(final_config["timeframe"])
    did = str(final_config["dataset"])
    evaluation_frame = frame if frame is not None else load_cached_data(symbol, timeframe, validate_coverage=False)
    oos_start = evaluation_frame.index[0] + pd.DateOffset(months=config.train_months)
    rows = []
    variants = [
        ("full", {}),
        ("entropy_neutral_1", {"remove_entropy": True}),
        ("conviction_neutral_1", {"remove_conviction": True}),
        ("wick_zero", {"remove_wick": True}),
    ]
    for label, kwargs in variants:
        if label == "full":
            result = full_result if full_result is not None else run_final_strategy(
                final_config,
                config,
                frame=evaluation_frame,
            )
            rows.append({"dataset": did, "ablation": label, **result.metrics})
            continue
        series = construct_ablated_wek(
            evaluation_frame,
            length=int(final_config["length"]),
            smooth=int(final_config["smooth"]),
            **kwargs,
        )
        result = run_backtest_with_active_start(
            evaluation_frame,
            series,
            active_start=oos_start,
            variant=str(final_config["variant"]),
            threshold=float(final_config["threshold"]),
            exit_bars=int(final_config["exit_bars"]),
            fee_rate=float(final_config.get("fee_rate", config.fee_rate)),
            slippage_rate=float(final_config.get("slippage_rate", config.slippage_rate)),
            bars_per_year=bars_per_year_for_timeframe(timeframe),
        )
        rows.append({"dataset": did, "ablation": label, **result.metrics})
    diagnostic = component_forward_return_diagnostic(
        evaluation_frame,
        int(final_config["length"]),
        int(final_config["smooth"]),
        active_start=oos_start,
    )
    out = pd.DataFrame(rows)
    if not out.empty:
        full_row = out[out["ablation"] == "full"].iloc[0]
        out["total_return_degradation"] = float(full_row["total_return"]) - out["total_return"].astype(float)
        out["Sharpe_degradation"] = float(full_row["Sharpe"]) - out["Sharpe"].astype(float)
    for key, value in diagnostic.items():
        out[key] = value
    return out


def run_backtest_with_active_start(
    frame: pd.DataFrame,
    wek_series: pd.Series,
    *,
    active_start: pd.Timestamp,
    variant: str,
    threshold: float,
    exit_bars: int,
    fee_rate: float,
    slippage_rate: float,
    bars_per_year: float,
) -> backtest.BacktestResult:
    if hasattr(backtest, "_run_backtest_engine") and hasattr(backtest, "BacktestConfig"):
        config = backtest.BacktestConfig(
            variant=variant,
            threshold=float(threshold),
            exit_bars=int(exit_bars),
            fee_rate=float(fee_rate),
            slippage_rate=float(slippage_rate),
            bars_per_year=bars_per_year,
        )
        return backtest._run_backtest_engine(df=frame, wek=wek_series, config=config, active_start=active_start)
    oos = frame.loc[frame.index >= active_start]
    return backtest.run_backtest(
        oos,
        wek_series.loc[oos.index],
        variant=variant,
        threshold=threshold,
        exit_bars=exit_bars,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        bars_per_year=bars_per_year,
    )


def component_forward_return_diagnostic(
    frame: pd.DataFrame,
    length: int,
    smooth: int,
    *,
    active_start: pd.Timestamp | None = None,
) -> dict[str, float]:
    _, components = compute_wek(frame, length=length, smooth=smooth, return_components=True)
    forward = frame["close"].pct_change().shift(-1)
    if active_start is not None:
        components = components.loc[components.index >= active_start]
        forward = forward.loc[forward.index >= active_start]
    out: dict[str, float] = {}
    for column in ("event_factor", "wick_sig", "conv_sig", "WEK"):
        joined = pd.concat([components[column], forward], axis=1).dropna()
        out[f"{column}_spearman_fwd1"] = float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")) if len(joined) > 3 else np.nan
    return out


def monte_carlo_permutations(trade_returns: Sequence[float], *, n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    returns = np.asarray(list(trade_returns), dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    if returns.size == 0:
        return pd.DataFrame(columns=["iteration", "max_drawdown", "terminal_return", "trades"])
    terminal_return = float(np.prod(1.0 + returns) - 1.0)
    for iteration in range(int(n)):
        shuffled = rng.permutation(returns)
        equity = np.cumprod(1.0 + shuffled)
        running_max = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        drawdown = equity / running_max - 1.0
        rows.append(
            {
                "iteration": iteration,
                "max_drawdown": float(drawdown.min()),
                "terminal_return": terminal_return,
                "trades": int(returns.size),
            }
        )
    return pd.DataFrame(rows)


def chronological_trade_drawdown(trade_returns: Sequence[float]) -> float:
    returns = np.asarray(list(trade_returns), dtype=float)
    if returns.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float((equity / running_max - 1.0).min())


def monte_carlo_summary(mc: pd.DataFrame, trade_returns: Sequence[float]) -> pd.DataFrame:
    if mc.empty:
        return pd.DataFrame(
            columns=[
                "permutations",
                "trades",
                "terminal_return",
                "mc_drawdown_q05",
                "mc_drawdown_q50",
                "mc_drawdown_q95",
                "chronological_trade_max_drawdown",
            ]
        )
    return pd.DataFrame(
        [
            {
                "permutations": int(len(mc)),
                "trades": int(mc["trades"].iloc[0]),
                "terminal_return": float(mc["terminal_return"].iloc[0]),
                "mc_drawdown_q05": float(mc["max_drawdown"].quantile(0.05)),
                "mc_drawdown_q50": float(mc["max_drawdown"].quantile(0.50)),
                "mc_drawdown_q95": float(mc["max_drawdown"].quantile(0.95)),
                "chronological_trade_max_drawdown": chronological_trade_drawdown(trade_returns),
            }
        ]
    )


def build_is_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    metric_columns = [
        "total_return",
        "CAGR",
        "Sharpe",
        "Sortino",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "trades",
        "exposure",
    ]
    rows = []
    for dataset, group in selected.groupby("dataset"):
        row: dict[str, object] = {
            "dataset": dataset,
            "folds": int(group["fold"].nunique()) if "fold" in group else int(len(group)),
            "fallback_folds": int(group["min_trade_fallback"].sum()) if "min_trade_fallback" in group else 0,
        }
        for metric in metric_columns:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"is_mean_{metric}"] = float(values.mean())
                row[f"is_median_{metric}"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_leaderboard(results: Sequence[DatasetResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "dataset": result.dataset,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "rows": result.rows,
            "start": result.start,
            "end": result.end,
            "folds": result.fold_count,
            "fallback_folds": result.fallback_folds,
            "min_trades_required": result.min_trades_required,
            "credible_edge": result.credible_edge,
            "power_warning": result.power_warning,
            "total_return": result.wfo_metrics.get("total_return"),
            "CAGR": result.wfo_metrics.get("CAGR"),
            "Sharpe": result.wfo_metrics.get("Sharpe"),
            "Sortino": result.wfo_metrics.get("Sortino"),
            "max_drawdown": result.wfo_metrics.get("max_drawdown"),
            "win_rate": result.wfo_metrics.get("win_rate"),
            "profit_factor": result.wfo_metrics.get("profit_factor"),
            "trades": result.wfo_metrics.get("trades"),
            "exposure": result.wfo_metrics.get("exposure"),
            "benchmark_total_return": result.benchmark_metrics.get("total_return"),
            "benchmark_CAGR": result.benchmark_metrics.get("CAGR"),
            "benchmark_Sharpe": result.benchmark_metrics.get("Sharpe"),
            "benchmark_Sortino": result.benchmark_metrics.get("Sortino"),
            "benchmark_max_drawdown": result.benchmark_metrics.get("max_drawdown"),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False, na_position="last").reset_index(drop=True)


def create_charts(
    *,
    equity: pd.DataFrame,
    fixed: pd.DataFrame,
    selected: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    final_config: dict[str, object],
) -> None:
    if not equity.empty:
        plt.figure(figsize=(12, 7))
        for did, group in equity.groupby("dataset"):
            group = group.sort_values("timestamp")
            plt.plot(group["timestamp"], group["selected_equity"], label=f"{did} WEK", linewidth=1.0)
            plt.plot(group["timestamp"], group["benchmark_equity"], label=f"{did} B&H", linewidth=0.8, linestyle="--")
        plt.title("Selected OOS equity vs cost-matched buy&hold")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "oos_selected_equity_vs_benchmark.png", dpi=150)
        plt.close()

    aggregate = aggregate_fixed_candidates(fixed)
    if final_config and not aggregate.empty:
        subset = aggregate[(aggregate["dataset"] == final_config["dataset"]) & (aggregate["variant"] == final_config["variant"])]
        lengths = sorted(subset["length"].unique())
        smooths = sorted(subset["smooth"].unique())
        fig, axes = plt.subplots(len(lengths), len(smooths), figsize=(4 * len(smooths), 3.2 * len(lengths)), squeeze=False)
        for i, length in enumerate(lengths):
            for j, smooth in enumerate(smooths):
                ax = axes[i][j]
                facet = subset[(subset["length"] == length) & (subset["smooth"] == smooth)]
                pivot = facet.pivot_table(index="threshold", columns="exit_bars", values="mean_oos_Sharpe", aggfunc="mean")
                im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="RdYlGn")
                ax.set_title(f"length={length}, smooth={smooth}")
                ax.set_xticks(range(len(pivot.columns)), labels=[str(x) for x in pivot.columns])
                ax.set_yticks(range(len(pivot.index)), labels=[str(x) for x in pivot.index])
                ax.set_xlabel("exit")
                ax.set_ylabel("threshold")
                fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(f"Fixed OOS Sharpe heatmaps: {final_config['dataset']} / {final_config['variant']}")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "best_result_parameter_heatmaps.png", dpi=150)
        plt.close(fig)

    if not monte_carlo.empty:
        plt.figure(figsize=(9, 5))
        plt.hist(monte_carlo["max_drawdown"], bins=40, color="#4d7c8a", edgecolor="white")
        plt.title("Monte Carlo drawdown histogram")
        plt.xlabel("max drawdown")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "mc_drawdown_histogram.png", dpi=150)
        plt.close()

    if not selected.empty:
        freq = selected.groupby(["dataset", "variant"]).size().reset_index(name="folds")
        labels = freq["dataset"] + "\n" + freq["variant"]
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(freq)), freq["folds"], color="#7d6b91")
        plt.xticks(range(len(freq)), labels, rotation=70, ha="right", fontsize=7)
        plt.title("Selection frequency by dataset and variant")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "selection_frequency.png", dpi=150)
        plt.close()


def write_report(
    leaderboard: pd.DataFrame,
    dataset_results: Sequence[DatasetResult],
    final_config: dict[str, object],
    ablation: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    selected_is_summary: pd.DataFrame | None = None,
    mc_summary: pd.DataFrame | None = None,
) -> None:
    selected_is_summary = selected_is_summary if selected_is_summary is not None else pd.DataFrame()
    mc_summary = mc_summary if mc_summary is not None else pd.DataFrame()
    no_edge = leaderboard.empty or not bool(leaderboard["credible_edge"].fillna(False).any())
    degradation_text = ablation_degradation_text(ablation)
    fixed_evaluation = final_config.get("fixed_evaluation", {}) if final_config else {}
    fixed_evaluation = fixed_evaluation if isinstance(fixed_evaluation, dict) else {}
    fixed_metrics = fixed_evaluation.get("metrics", {})
    fixed_metrics = fixed_metrics if isinstance(fixed_metrics, dict) else {}
    mc_metadata = final_config.get("monte_carlo", {}) if final_config else {}
    mc_metadata = mc_metadata if isinstance(mc_metadata, dict) else {}
    fixed_trade_count = fixed_evaluation.get("trade_count", "brak")
    trade_terminal = fixed_evaluation.get("trade_compound_return")
    fixed_total_return = fixed_metrics.get("total_return")
    trade_terminal_text = f"{float(trade_terminal):.4%}" if trade_terminal is not None else "brak"
    fixed_total_return_text = f"{float(fixed_total_return):.4%}" if fixed_total_return is not None else "brak"
    mc_source = mc_metadata.get("source", "results/final_strategy_trades.csv")
    lines = [
        "# Raport badania WEK",
        "",
        "## Zakres danych",
        _markdown_table(pd.DataFrame([{"dataset": r.dataset, "rows": r.rows, "start": r.start, "end": r.end} for r in dataset_results])),
        "",
        "## Założenia i wykonanie",
        "Sygnał jest liczony przy zamknięciu świecy, egzekucja odbywa się na następnym otwarciu, bez look-ahead. Koszt wynosi 0.001 fee + 0.0005 slippage na stronę. tick_size=1e-10 jest tylko podłogą numeryczną WEK; brak specjalnej korekty flat-barów, więc tick giełdowy jest tu praktycznie nieistotny. Walk-forward używa 12 miesięcy IS, 3 miesięcy OOS i kroku 3 miesiące. Minimalna liczba transakcji IS to 20 dla 1h/4h i 8 dla 1d; fallback jest zapisany w tabelach foldów.",
        "",
        "## IS",
        "Poniższa tabela agreguje wyłącznie wyniki IS wybranych foldów; nie jest to dowód OOS.",
        _markdown_table(selected_is_summary),
        "",
        "## OOS",
        "Poniższy leaderboard pokazuje zszyte OOS wybranych konfiguracji, czyli główny materiał dowodowy. Reguła wiarygodnej przewagi jest predefiniowana: dodatni net OOS, adekwatna liczba transakcji oraz lepszy total return i Sharpe niż kosztowo dopasowany buy&hold.",
        _markdown_table(leaderboard.head(12)),
        "",
        "## Benchmark buy&hold",
        "Benchmark to long buy&hold na identycznych znacznikach czasu OOS, z kosztem wejścia i wyjścia po 0.0015. Pierwszy zwrot benchmarku zawiera koszt wejścia i intrabar open-to-close, aby Sharpe był porównywalny czasowo z OOS. Porównanie jest w leaderboardzie oraz na wykresie equity.",
        "",
        "## Moc testu",
        _markdown_table(pd.DataFrame([{"dataset": r.dataset, "folds": r.fold_count, "fallback_folds": r.fallback_folds, "warning": r.power_warning} for r in dataset_results])),
        "",
        "## Stabilność i przeoptymalizowanie",
        "Wszystkie wyniki należy czytać z caveat wielokrotnego testowania i OOS meta-selection; fixed-candidate diagnostics nie są nietkniętym potwierdzeniem.",
    ]
    if final_config:
        lines.append(
            f"Wybrana konfiguracja końcowa: `{final_config.get('dataset')}` `{final_config.get('variant')}` length={final_config.get('length')}, smooth={final_config.get('smooth')}, threshold={final_config.get('threshold')}, exit={final_config.get('exit_bars')}. "
            f"Udział dodatnich sąsiadów Sharpe: {final_config.get('positive_sharpe_share')}, mediana sąsiadów: {final_config.get('median_neighbor_sharpe')}, flaga overfit={final_config.get('overfit')}, underpowered_fallback={final_config.get('underpowered_fallback')}. "
            "To jest OOS meta-selection, a nie nietknięte potwierdzenie. Reguła: izolowany szczyt oznacza overfit, gdy mniej niż połowa bezpośrednich sąsiadów ma dodatni Sharpe albo mediana Sharpe sąsiadów jest <= 0. Istnieje też caveat wielokrotnego testowania."
        )
    lines.extend(
        [
            "",
            "## Monte Carlo",
            "Monte Carlo permutuje zwroty netto transakcji dokładnie wybranego fixed final candidate, ocenionego od wspólnego początku OOS z pełnym kontekstem przyczynowym. To wyłącznie test ryzyka ścieżki wynikającego z kolejności transakcji (trade-order path risk); nie symuluje nowych zwrotów, kosztów ani reżimów rynku.",
            f"Źródło `{mc_source}` (`net_return`) zawiera {fixed_trade_count} transakcji. Złożony terminal return tych transakcji wynosi {trade_terminal_text}, a fixed-evaluation total_return wynosi {fixed_total_return_text}; permutacja zachowuje terminal return i liczbę transakcji.",
            _markdown_table(mc_summary),
            _markdown_table(monte_carlo.describe().reset_index() if not monte_carlo.empty else pd.DataFrame()),
            "",
            "## Ablacja",
            "Ablacja porównuje pełny WEK z neutralizacją entropii, neutralizacją conviction oraz wyzerowaniem wick signal przy fixed-final-params, bez reoptymalizacji. Używa pełnej historii dla przyczynowego WEK/EMA/Donchian, a PnL startuje od wspólnego OOS. Interpretacja ma ograniczenie skali i interakcji komponentów.",
            degradation_text,
            _markdown_table(ablation),
            "",
            "## Ścieżki artefaktów",
            "- `results/aggregate_leaderboard.csv`",
            "- `results/selected_is_summary.csv`",
            "- `results/final_config.json`",
            "- `results/final_strategy_trades.csv`",
            "- `results/final_strategy_equity.csv`",
            "- `results/ablation.csv`",
            "- `results/monte_carlo_1000_permutation.csv`",
            "- `results/monte_carlo_summary.csv`",
            "- `charts/oos_selected_equity_vs_benchmark.png`",
            "- `charts/best_result_parameter_heatmaps.png`",
            "- `charts/mc_drawdown_histogram.png`",
            "- `charts/selection_frequency.png`",
            "",
            "## Rekomendacja",
        ]
    )
    if no_edge:
        lines.append("NO EDGE: brak wystarczająco wiarygodnej przewagi według reguły benchmark-aware, do czasu potwierdzenia na nowych danych poza tą meta-selekcją OOS.")
    else:
        lines.append("Przewaga, jeśli występuje, powinna być traktowana jako hipoteza do dalszej walidacji poza badaną próbką. Produkcyjne użycie wymaga dodatkowego holdoutu i kontroli kosztów/poślizgu.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_final_outputs_from_artifacts(
    config: ResearchConfig | None = None,
    *,
    final_config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Refresh final evaluation, MC, ablation, charts, and report without grid search."""
    config = config or ResearchConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    if final_config is None:
        config_path = RESULTS_DIR / "final_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"final config not found at {config_path}")
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not loaded:
            raise ValueError("final_config.json must contain a non-empty JSON object")
        final_config = loaded

    symbol = str(final_config["symbol"])
    timeframe = str(final_config["timeframe"])
    frame = load_cached_data(symbol, timeframe, validate_coverage=False)
    fixed_result = run_final_strategy(final_config, config, frame=frame)
    outputs = postprocess_final_strategy(final_config, config, result=fixed_result)
    enriched_config = outputs["final_config"]

    ablation = run_ablation(
        enriched_config,
        config,
        full_result=fixed_result,
        frame=frame,
    )
    ablation.to_csv(RESULTS_DIR / "ablation.csv", index=False)

    leaderboard = _read_csv_if_present(RESULTS_DIR / "aggregate_leaderboard.csv")
    selected_summary = _read_csv_if_present(RESULTS_DIR / "selected_is_summary.csv")
    dataset_results = _load_checkpoint_results()
    fixed = _concat_csv_artifacts("*_fixed_oos_stability.csv")
    selected = _concat_csv_artifacts("*_selected_is.csv")
    equity = _concat_csv_artifacts("*_oos_equity.csv")
    if "timestamp" in equity:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True, errors="coerce")

    create_charts(
        equity=equity,
        fixed=fixed,
        selected=selected,
        monte_carlo=outputs["monte_carlo"],
        final_config=enriched_config,
    )
    write_report(
        leaderboard,
        dataset_results,
        enriched_config,
        ablation,
        outputs["monte_carlo"],
        selected_summary,
        outputs["monte_carlo_summary"],
    )
    return {**outputs, "ablation": ablation, "leaderboard": leaderboard}


def _empty_final_outputs(config: ResearchConfig) -> dict[str, object]:
    trades = pd.DataFrame(
        columns=[
            "dataset",
            "symbol",
            "timeframe",
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
    equity = pd.DataFrame(columns=["timestamp", "dataset", "symbol", "timeframe", "returns", "equity"])
    mc = monte_carlo_permutations([], n=config.mc_permutations, seed=config.seed)
    mc_summary = monte_carlo_summary(mc, [])
    trades.to_csv(RESULTS_DIR / "final_strategy_trades.csv", index=False)
    equity.to_csv(RESULTS_DIR / "final_strategy_equity.csv", index=False)
    mc.to_csv(RESULTS_DIR / "monte_carlo_1000_permutation.csv", index=False)
    mc_summary.to_csv(RESULTS_DIR / "monte_carlo_summary.csv", index=False)
    _write_json(RESULTS_DIR / "final_config.json", {})
    return {
        "final_config": {},
        "result": None,
        "trades": trades,
        "equity": equity,
        "monte_carlo": mc,
        "monte_carlo_summary": mc_summary,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _concat_csv_artifacts(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(RESULTS_DIR.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_checkpoint_results() -> list[DatasetResult]:
    results = []
    for path in sorted(RESULTS_DIR.glob("checkpoint_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            results.append(DatasetResult(**payload))
    return results


def ablation_degradation_text(ablation: pd.DataFrame) -> str:
    if ablation.empty or "ablation" not in ablation:
        return "Brak danych ablacji."
    non_structural = ablation[ablation["ablation"].isin(["entropy_neutral_1", "conviction_neutral_1"])].copy()
    if non_structural.empty:
        return "Brak porównania entropii i conviction."
    metric = "Sharpe_degradation" if "Sharpe_degradation" in non_structural else "total_return_degradation"
    non_structural[metric] = pd.to_numeric(non_structural[metric], errors="coerce")
    worst = non_structural.sort_values(metric, ascending=False, na_position="last").iloc[0]
    component = "entropia" if worst["ablation"] == "entropy_neutral_1" else "conviction"
    wick_note = " Wick-zero jest wynikiem strukturalnym: wyzerowanie wick signal usuwa kierunek sygnału, więc nie jest porównywalną subtelną ablację skali."
    return (
        f"Największą degradację poza structural wick-zero powoduje komponent: {component} "
        f"({metric}={worst.get(metric)})."
        + wick_note
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Brak danych._"
    trimmed = frame.copy()
    for column in trimmed.columns:
        if pd.api.types.is_float_dtype(trimmed[column]):
            trimmed[column] = trimmed[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            trimmed[column] = trimmed[column].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(column) for column in trimmed.columns]
    rows = trimmed.astype(str).values.tolist()
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), separator, *(fmt(row) for row in rows)])


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _float_dict(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) if value is not None else np.nan for key, value in metrics.items()}


__all__: Iterable[str] = (
    "ResearchConfig",
    "run_study",
    "run_dataset",
    "buy_and_hold_benchmark",
    "equity_returns",
    "construct_ablated_wek",
    "run_final_strategy",
    "postprocess_final_strategy",
    "refresh_final_outputs_from_artifacts",
    "monte_carlo_permutations",
    "monte_carlo_summary",
    "has_credible_edge",
    "stability_neighbor_diagnostic",
    "select_final_config",
)
