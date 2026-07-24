from __future__ import annotations

import argparse
from pathlib import Path

from forward_test.core import SYMBOL, TIMEFRAME, initialize_artifacts, run_forward_test
from forward_test.data import default_since_for_warmup, fetch_binance_ohlcv_bundle, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen H2 forward paper-trading core.")
    parser.add_argument("--root", default="forward_test", help="forward-test artifact directory")
    parser.add_argument("--init", action="store_true", help="initialize empty state/ledger/head artifacts")
    parser.add_argument("--limit", type=int, default=1000, help="ccxt OHLCV limit")
    args = parser.parse_args()
    root = Path(args.root)
    if args.init:
        initialize_artifacts(root)
        print(f"initialized {root}")
        return
    run_now = utc_now()
    fetched = fetch_binance_ohlcv_bundle(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        since=default_since_for_warmup(),
        limit=args.limit,
        now=run_now,
    )
    result = run_forward_test(fetched.completed, root=root, now=run_now, current_open=fetched.current_open)
    print(
        f"status={result.status} processed={result.processed} "
        f"appended={result.appended} head={result.ledger_head}"
    )


if __name__ == "__main__":
    main()
