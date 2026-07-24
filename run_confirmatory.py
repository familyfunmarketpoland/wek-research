from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from research_lab.data_guard import DATA_DIR
from research_lab.runner import DEFAULT_REPORT_PATH, RESULTS_DIRNAME, run_holdout_phase, run_research_phase


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the H1-H6 confirmatory workflow.")
    parser.add_argument("phase", choices=("research", "holdout"), help="Run exactly one preregistered phase.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Guarded data directory with split manifest.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional manifest path inside data-dir.")
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIRNAME), help="Artifact directory.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Main report2.md path.")
    parser.add_argument("--config-path", type=Path, default=Path("configs/pre_registered.json"), help="Frozen config path.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "research":
        decision = run_research_phase(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
            manifest_path=args.manifest_path,
            config_path=args.config_path,
        )
        print(decision["decision"])
        print(f"artifacts: {args.output_dir}")
        return 0
    result = run_holdout_phase(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        manifest_path=args.manifest_path,
        config_path=args.config_path,
    )
    print(result["final_status"])
    print(f"artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
