"""Binance OHLCV downloader and parquet cache.

CLI usage:
    ./.venv/bin/python -m data_pipeline
    ./.venv/bin/python -m data_pipeline --symbols BTC/USDT ETH/USDT --timeframes 1h 4h

The default CLI downloads BTC/USDT, ETH/USDT, and SOL/USDT for 1h, 4h, and 1d,
validates that at least 3 years of history are cached, and stores each dataset
under ``./data/{base}_{quote}_{timeframe}.parquet`` relative to this module.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1h", "4h", "1d")
DEFAULT_YEARS = 3
DEFAULT_LIMIT = 1000
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.0
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TIMEFRAME_TO_DELTA = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
TIMEFRAME_TO_FLOOR = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def _get_exchange():
    try:
        import ccxt
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError("ccxt is required to download Binance OHLCV data") from exc

    return ccxt.binance({"enableRateLimit": True})


def get_symbol_metadata(symbol: str, *, exchange=None) -> dict[str, float | str | None]:
    """Return lightweight market metadata without coupling cache reads to it."""
    _validate_symbol_timeframe(symbol, "1h")
    client = exchange if exchange is not None else _get_exchange()
    market = client.market(symbol)

    precision = market.get("precision") or {}
    tick_size = precision.get("price")
    if tick_size is None:
        limits = market.get("limits") or {}
        price_limits = limits.get("price") or {}
        tick_size = price_limits.get("min")

    return {
        "symbol": symbol,
        "base": market.get("base"),
        "quote": market.get("quote"),
        "tick_size": float(tick_size) if tick_size is not None else None,
    }


def _cache_path(symbol: str, timeframe: str, data_dir: Path | None = None) -> Path:
    base, quote = symbol.split("/")
    cache_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    return cache_dir / f"{base.lower()}_{quote.lower()}_{timeframe}.parquet"


def _validate_symbol_timeframe(symbol: str, timeframe: str) -> None:
    if "/" not in symbol:
        raise ValueError(f"symbol must be in BASE/QUOTE form, got {symbol!r}")
    if timeframe not in TIMEFRAME_TO_DELTA:
        supported = ", ".join(sorted(TIMEFRAME_TO_DELTA))
        raise ValueError(f"unsupported timeframe {timeframe!r}; expected one of: {supported}")


def _utc_now(now: pd.Timestamp | None = None) -> pd.Timestamp:
    timestamp = now if now is not None else pd.Timestamp.now(tz="UTC")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _last_completed_bar(now: pd.Timestamp | None, timeframe: str) -> pd.Timestamp:
    return _utc_now(now).floor(TIMEFRAME_TO_FLOOR[timeframe]) - TIMEFRAME_TO_DELTA[timeframe]


def _coverage_start(now: pd.Timestamp | None, years: int, timeframe: str) -> pd.Timestamp:
    return _last_completed_bar(now, timeframe) - pd.DateOffset(years=years)


def _normalize_ohlcv(rows: list[list[float | int]]) -> pd.DataFrame:
    if not rows:
        empty_index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=empty_index, dtype=float)

    frame = pd.DataFrame(rows, columns=["timestamp_ms", *OHLCV_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    frame.index.name = "timestamp"
    frame = frame.loc[:, OHLCV_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    return frame.sort_index()


def _drop_incomplete_bars(
    frame: pd.DataFrame,
    timeframe: str,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    complete_before = _last_completed_bar(now, timeframe)
    return frame.loc[frame.index <= complete_before]


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, incoming], axis=0)

    if merged.empty:
        return merged

    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    merged = merged.loc[:, OHLCV_COLUMNS].astype(float)
    merged.index = pd.DatetimeIndex(merged.index, tz="UTC", name="timestamp")
    return merged


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        empty_index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=empty_index, dtype=float)

    cached = pd.read_parquet(path)
    if not isinstance(cached.index, pd.DatetimeIndex):
        raise ValueError(f"cache at {path} must use a DatetimeIndex")
    if cached.index.tz is None:
        cached.index = cached.index.tz_localize("UTC")
    else:
        cached.index = cached.index.tz_convert("UTC")
    missing = [column for column in OHLCV_COLUMNS if column not in cached.columns]
    if missing:
        raise ValueError(f"cache at {path} is missing columns: {missing}")
    cached = cached.loc[:, OHLCV_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    cached.index.name = "timestamp"
    cached = cached[~cached.index.duplicated(keep="last")]
    return cached.sort_index()


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _validate_coverage(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    years: int,
    now: pd.Timestamp | None = None,
) -> None:
    if frame.empty:
        raise ValueError(f"no cached OHLCV data available for {symbol} {timeframe}")

    required_start = _coverage_start(now, years, timeframe)
    first_timestamp = frame.index.min()
    if first_timestamp > required_start:
        raise ValueError(
            f"insufficient {symbol} {timeframe} coverage: earliest candle {first_timestamp.isoformat()} "
            f"is later than required start {required_start.isoformat()}"
        )


def _fetch_page(exchange, symbol: str, timeframe: str, since_ms: int) -> list[list[float | int]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=DEFAULT_LIMIT)
        except Exception as exc:  # pragma: no cover - exact exception type depends on ccxt runtime
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(BACKOFF_SECONDS * attempt)

    assert last_error is not None
    raise RuntimeError(
        f"failed to fetch {symbol} {timeframe} from Binance after {MAX_RETRIES} attempts"
    ) from last_error


def _fetch_incremental(
    exchange,
    symbol: str,
    timeframe: str,
    start_since_ms: int,
) -> pd.DataFrame:
    timeframe_ms = int(TIMEFRAME_TO_DELTA[timeframe].total_seconds() * 1000)
    since_ms = start_since_ms
    pages: list[pd.DataFrame] = []

    while True:
        rows = _fetch_page(exchange, symbol, timeframe, since_ms=since_ms)
        page = _normalize_ohlcv(rows)
        if page.empty:
            break

        page = page.loc[page.index >= pd.to_datetime(since_ms, unit="ms", utc=True)]
        if page.empty:
            raise RuntimeError(
                f"pagination stalled for {symbol} {timeframe}: Binance returned no rows after since={since_ms}"
            )

        pages.append(page)
        last_timestamp = page.index.max()
        next_since_ms = int(last_timestamp.timestamp() * 1000) + timeframe_ms
        if next_since_ms <= since_ms:
            raise RuntimeError(
                f"pagination failed to advance for {symbol} {timeframe}: since={since_ms}, "
                f"last_timestamp={last_timestamp.isoformat()}"
            )
        if len(rows) < DEFAULT_LIMIT:
            break
        since_ms = next_since_ms

    if not pages:
        empty_index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=empty_index, dtype=float)

    return pd.concat(pages, axis=0)


def download_ohlcv(
    symbol: str,
    timeframe: str,
    *,
    years: int = DEFAULT_YEARS,
    data_dir: Path | None = None,
    exchange=None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    _validate_symbol_timeframe(symbol, timeframe)
    cache_path = _cache_path(symbol, timeframe, data_dir=data_dir)
    cached = _drop_incomplete_bars(_read_cache(cache_path), timeframe=timeframe, now=now)

    required_start = _coverage_start(now, years, timeframe)
    complete_before = _last_completed_bar(now, timeframe)

    if cached.empty:
        fetch_since = required_start
    elif cached.index.min() > required_start:
        # Refetching from the required boundary safely fills a missing left edge
        # while merge/deduplication preserves any already-current cached tail.
        fetch_since = required_start
    else:
        last_cached = cached.index.max()
        if last_cached >= complete_before:
            fetch_since = None
        else:
            fetch_since = max(required_start, last_cached + TIMEFRAME_TO_DELTA[timeframe])

    incoming = pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    if fetch_since is not None and fetch_since <= complete_before:
        client = exchange if exchange is not None else _get_exchange()
        incoming = _fetch_incremental(
            client,
            symbol,
            timeframe,
            start_since_ms=int(fetch_since.timestamp() * 1000),
        )

    merged = _drop_incomplete_bars(_merge_frames(cached, incoming), timeframe=timeframe, now=now)
    _validate_coverage(merged, symbol=symbol, timeframe=timeframe, years=years, now=now)

    if cache_path.exists():
        existing = _read_cache(cache_path)
        if merged.equals(existing):
            return merged

    _write_cache(cache_path, merged)
    return merged


def load_cached_data(
    symbol: str,
    timeframe: str,
    *,
    years: int = DEFAULT_YEARS,
    data_dir: Path | None = None,
    validate_coverage: bool = True,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    _validate_symbol_timeframe(symbol, timeframe)
    cache_path = _cache_path(symbol, timeframe, data_dir=data_dir)
    cached = _drop_incomplete_bars(_read_cache(cache_path), timeframe=timeframe, now=now)
    if validate_coverage:
        _validate_coverage(cached, symbol=symbol, timeframe=timeframe, years=years, now=now)
    return cached


def download_defaults(
    *,
    years: int = DEFAULT_YEARS,
    data_dir: Path | None = None,
    now: pd.Timestamp | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    client = _get_exchange()
    datasets: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in DEFAULT_SYMBOLS:
        for timeframe in DEFAULT_TIMEFRAMES:
            datasets[(symbol, timeframe)] = download_ohlcv(
                symbol,
                timeframe,
                years=years,
                data_dir=data_dir,
                exchange=client,
                now=now,
            )
    return datasets


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    client = _get_exchange()
    for symbol in args.symbols:
        for timeframe in args.timeframes:
            frame = download_ohlcv(
                symbol,
                timeframe,
                years=args.years,
                data_dir=args.data_dir,
                exchange=client,
            )
            print(f"{symbol} {timeframe}: {len(frame)} rows cached at {_cache_path(symbol, timeframe, args.data_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
