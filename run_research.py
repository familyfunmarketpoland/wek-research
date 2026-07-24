from __future__ import annotations

import argparse
from typing import Iterable

from research import ResearchConfig, run_study


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible WEK research study.")
    parser.add_argument("--quick", action="store_true", help="Run a tiny deterministic smoke study instead of the full grid.")
    parser.add_argument("--max-datasets", type=int, default=None, help="Limit the number of datasets; intended for local smoke runs.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = ResearchConfig.quick_config() if args.quick else ResearchConfig()
    if args.max_datasets is not None:
        config = ResearchConfig(
            lengths=config.lengths,
            smooths=config.smooths,
            thresholds=config.thresholds,
            exit_bars_options=config.exit_bars_options,
            variants=config.variants,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            train_months=config.train_months,
            oos_months=config.oos_months,
            step_months=config.step_months,
            seed=config.seed,
            mc_permutations=config.mc_permutations,
            symbols=config.symbols,
            timeframes=config.timeframes,
            quick=config.quick,
            max_datasets=args.max_datasets,
        )
    result = run_study(config)
    leaderboard = result["leaderboard"]
    final_config = result["final_config"]
    print(f"datasets: {len(leaderboard)}")
    if final_config:
        print(
            "final_config: "
            f"{final_config['dataset']} {final_config['variant']} "
            f"length={final_config['length']} smooth={final_config['smooth']} "
            f"threshold={final_config['threshold']} exit={final_config['exit_bars']}"
        )
    print("artifacts: results/, charts/, report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
