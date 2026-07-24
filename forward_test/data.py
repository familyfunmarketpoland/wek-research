from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OhlcvFetchResult:
    completed: pd.DataFrame
    current_open: dict[str, float | str] | None


def fetch_binance_ohlcv(
    *,
    symbol: str = "SOL/USDT",
    timeframe: str = "4h",
    since: pd.Timestamp | None = None,
    limit: int = 1000,
    now: pd.Timestamp | None = None,
    exchange: Any | None = None,
) -> pd.DataFrame:
    return fetch_binance_ohlcv_bundle(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=limit,
        now=now,
        exchange=exchange,
    ).completed


def fetch_binance_ohlcv_bundle(
    *,
    symbol: str = "SOL/USDT",
    timeframe: str = "4h",
    since: pd.Timestamp | None = None,
    limit: int = 1000,
    now: pd.Timestamp | None = None,
    exchange: Any | None = None,
) -> OhlcvFetchResult:
    """Fetch Binance public OHLCV through ccxt without API credentials.

    ccxt returns at most one page per call. The frozen forward filter needs more
    than 2210 4h candles of warm-up, so this adapter paginates deterministically
    from ``since`` through the last fully closed candle and excludes the current
    still-open candle from the completed frame. If the exchange page includes
    that open candle, only its timestamp/open is returned separately for pending
    next-open fills.
    """

    if exchange is None:
        import ccxt  # type: ignore[import-not-found]

        exchange = ccxt.binance({"enableRateLimit": True})
    per_page = min(int(limit), 1000)
    if per_page <= 0:
        raise ValueError("limit must be positive")
    delta = _timeframe_delta(timeframe)
    cutoff_open = _current_candle_open(now if now is not None else utc_now(), delta)
    since_ms = None
    if since is not None:
        ts = pd.Timestamp(since)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        since_ms = int(ts.timestamp() * 1000)
    rows_by_timestamp: dict[int, list[float]] = {}
    current_open: dict[str, float | str] | None = None
    next_since_ms = since_ms
    cutoff_ms = int(cutoff_open.timestamp() * 1000)

    while True:
        if next_since_ms is not None and next_since_ms > cutoff_ms:
            break
        page = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since_ms, limit=per_page)
        if not page:
            break
        max_seen_ms: int | None = None
        for row in page:
            if len(row) != 6:
                raise ValueError("exchange returned malformed OHLCV row")
            ts_ms = int(row[0])
            max_seen_ms = ts_ms if max_seen_ms is None else max(max_seen_ms, ts_ms)
            if ts_ms < cutoff_ms:
                rows_by_timestamp[ts_ms] = list(row)
            elif ts_ms == cutoff_ms:
                current_open = {"open_utc": cutoff_open.strftime("%Y-%m-%dT%H:%M:%SZ"), "open": float(row[1])}
        if max_seen_ms is None or max_seen_ms < (next_since_ms or max_seen_ms):
            raise ValueError("exchange pagination did not advance")
        candidate_next = max_seen_ms + int(delta.total_seconds() * 1000)
        if next_since_ms is not None and candidate_next <= next_since_ms:
            raise ValueError("exchange pagination stalled")
        next_since_ms = candidate_next
        if max_seen_ms >= cutoff_ms:
            break
        if len(page) < per_page:
            break

    frame = ohlcv_rows_to_frame([rows_by_timestamp[key] for key in sorted(rows_by_timestamp)])
    _validate_continuity(frame, delta)
    return OhlcvFetchResult(completed=frame, current_open=current_open)


def ohlcv_rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC"))
    index = pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True)
    out = frame.astype(float)
    out.index = pd.DatetimeIndex(index, name="open_utc")
    return out[["open", "high", "low", "close", "volume"]]


def default_since_for_warmup() -> pd.Timestamp:
    # 2210 4h bars before first eligibility, with extra margin for exchange pagination.
    return pd.Timestamp("2025-07-20T00:00:00Z")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    unit = timeframe[-1:]
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported timeframe: {timeframe}") from exc
    if value <= 0:
        raise ValueError("timeframe value must be positive")
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def _current_candle_open(now: pd.Timestamp, delta: pd.Timedelta) -> pd.Timestamp:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    epoch_ns = pd.Timestamp("1970-01-01T00:00:00Z").value
    delta_ns = delta.value
    open_ns = epoch_ns + ((ts.value - epoch_ns) // delta_ns) * delta_ns
    return pd.Timestamp(open_ns, tz="UTC")


def _validate_continuity(frame: pd.DataFrame, delta: pd.Timedelta) -> None:
    if frame.empty:
        return
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV frame must be sorted without duplicates")
    gaps = frame.index.to_series().diff().dropna()
    if not gaps.empty and not (gaps == delta).all():
        raise ValueError("OHLCV frame contains gaps after pagination")
