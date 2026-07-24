from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline import download_ohlcv, load_cached_data


class FakeExchange:
    def __init__(self, pages: list[list[list[float | int]]]) -> None:
        self.pages = list(pages)
        self.calls: list[dict[str, object]] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
        self.calls.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )
        if not self.pages:
            return []
        return self.pages.pop(0)


def _rows(
    start: str,
    periods: int,
    timeframe: str,
    *,
    price_start: float = 100.0,
) -> list[list[float | int]]:
    freq = {"1h": "1h", "4h": "4h", "1d": "1D"}[timeframe]
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    rows: list[list[float | int]] = []
    for offset, timestamp in enumerate(index):
        price = price_start + float(offset)
        rows.append(
            [
                int(timestamp.timestamp() * 1000),
                price,
                price + 1.0,
                price - 1.0,
                price + 0.5,
                1000.0 + offset,
            ]
        )
    return rows


def test_download_paginates_with_monotonic_since(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:30:00+00:00")
    page_one = _rows("2023-07-24 00:00:00+00:00", 1000, "1h")
    page_two = _rows("2023-09-03 16:00:00+00:00", 10, "1h", price_start=2000.0)
    exchange = FakeExchange([page_one, page_two])

    frame = download_ohlcv(
        "BTC/USDT",
        "1h",
        data_dir=tmp_path,
        exchange=exchange,
        now=now,
    )

    assert len(exchange.calls) == 2
    assert exchange.calls[0]["limit"] == 1000
    assert exchange.calls[1]["since"] == page_one[-1][0] + 3_600_000
    assert frame.index.is_monotonic_increasing
    assert frame.index.tz is not None


def test_download_merges_duplicates_and_sorts_cache(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:30:00+00:00")
    cache_path = tmp_path / "btc_usdt_1d.parquet"
    cached_index = pd.DatetimeIndex(
        ["2023-07-24 00:00:00+00:00", "2023-07-23 00:00:00+00:00", "2023-07-24 00:00:00+00:00"],
        tz="UTC",
        name="timestamp",
    )
    cached = pd.DataFrame(
        {
            "open": [10.0, 11.0, 20.0],
            "high": [11.0, 12.0, 21.0],
            "low": [9.0, 10.0, 19.0],
            "close": [10.5, 11.5, 20.5],
            "volume": [100.0, 110.0, 200.0],
        },
        index=cached_index,
    )
    cached.to_parquet(cache_path)

    exchange = FakeExchange(
        [
            [
                [int(pd.Timestamp("2023-07-26 00:00:00+00:00").timestamp() * 1000), 30, 31, 29, 30.5, 300],
            ]
        ]
    )

    frame = download_ohlcv(
        "BTC/USDT",
        "1d",
        data_dir=tmp_path,
        exchange=exchange,
        now=now,
    )

    assert list(frame.index) == list(frame.index.sort_values())
    assert len(frame) == 3
    assert frame.loc[pd.Timestamp("2023-07-24 00:00:00+00:00"), "open"] == 20.0


def test_download_drops_current_incomplete_candle(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:00:00+00:00")
    exchange = FakeExchange(
        [[
            [int(pd.Timestamp("2023-07-24 11:00:00+00:00").timestamp() * 1000), 10, 11, 9, 10.5, 100],
            [int(pd.Timestamp("2026-07-24 11:00:00+00:00").timestamp() * 1000), 15, 16, 14, 15.5, 150],
            [int(pd.Timestamp("2026-07-24 12:00:00+00:00").timestamp() * 1000), 20, 21, 19, 20.5, 200],
        ]]
    )

    frame = download_ohlcv(
        "BTC/USDT",
        "1h",
        data_dir=tmp_path,
        exchange=exchange,
        now=now,
    )

    assert pd.Timestamp("2026-07-24 12:00:00+00:00") not in frame.index
    assert pd.Timestamp("2023-07-24 11:00:00+00:00") in frame.index
    assert pd.Timestamp("2026-07-24 11:00:00+00:00") in frame.index


def test_download_is_idempotent_when_cache_is_current(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:30:00+00:00")
    last_complete = pd.Timestamp("2026-07-24 11:00:00+00:00")
    first = last_complete - pd.DateOffset(years=3)
    index = pd.DatetimeIndex([first, last_complete], tz="UTC", name="timestamp")
    cached = pd.DataFrame(
        {
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [100.0, 200.0],
        },
        index=index,
    )
    cache_path = tmp_path / "btc_usdt_1h.parquet"
    cached.to_parquet(cache_path)
    exchange = FakeExchange([])

    frame = download_ohlcv(
        "BTC/USDT",
        "1h",
        data_dir=tmp_path,
        exchange=exchange,
        now=now,
    )

    assert exchange.calls == []
    reloaded = pd.read_parquet(cache_path)
    pd.testing.assert_frame_equal(frame, reloaded)


def test_download_backfills_missing_left_edge_when_tail_is_current(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:30:00+00:00")
    required_start = pd.Timestamp("2023-07-24 11:00:00+00:00")
    last_complete = pd.Timestamp("2026-07-24 11:00:00+00:00")
    cached_index = pd.DatetimeIndex(
        [required_start + pd.Timedelta(hours=1), last_complete],
        tz="UTC",
        name="timestamp",
    )
    cached = pd.DataFrame(
        {
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [100.0, 200.0],
        },
        index=cached_index,
    )
    cached.to_parquet(tmp_path / "btc_usdt_1h.parquet")
    exchange = FakeExchange(
        [[[int(required_start.timestamp() * 1000), 5, 6, 4, 5.5, 50]]]
    )

    frame = download_ohlcv(
        "BTC/USDT",
        "1h",
        data_dir=tmp_path,
        exchange=exchange,
        now=now,
    )

    assert len(exchange.calls) == 1
    assert exchange.calls[0]["since"] == int(required_start.timestamp() * 1000)
    assert frame.index.min() == required_start
    assert frame.index.max() == last_complete


def test_download_raises_for_insufficient_coverage(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:30:00+00:00")
    exchange = FakeExchange([_rows("2025-01-01 00:00:00+00:00", 5, "1d")])

    with pytest.raises(ValueError, match="insufficient BTC/USDT 1d coverage"):
        download_ohlcv(
            "BTC/USDT",
            "1d",
            data_dir=tmp_path,
            exchange=exchange,
            now=now,
        )


def test_load_cached_data_requires_strict_calendar_coverage(tmp_path: Path) -> None:
    now = pd.Timestamp("2026-07-24 12:30:00+00:00")
    required_start = pd.Timestamp("2023-07-23 00:00:00+00:00")
    last_complete = pd.Timestamp("2026-07-23 00:00:00+00:00")
    index = pd.DatetimeIndex(
        [required_start + pd.Timedelta(days=1), last_complete],
        tz="UTC",
        name="timestamp",
    )
    frame = pd.DataFrame(
        {
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [100.0, 200.0],
        },
        index=index,
    )
    cache_path = tmp_path / "eth_usdt_1d.parquet"
    frame.to_parquet(cache_path)

    with pytest.raises(ValueError, match="insufficient ETH/USDT 1d coverage"):
        load_cached_data("ETH/USDT", "1d", data_dir=tmp_path, now=now)

    exact_frame = frame.copy()
    exact_frame.index = pd.DatetimeIndex(
        [required_start, last_complete], tz="UTC", name="timestamp"
    )
    exact_frame.to_parquet(cache_path)

    loaded = load_cached_data("ETH/USDT", "1d", data_dir=tmp_path, now=now)
    assert loaded.index.min() == required_start


def test_load_cached_data_preserves_utc_index_and_numeric_schema(tmp_path: Path) -> None:
    index = pd.DatetimeIndex(
        ["2023-07-24 00:00:00+00:00", "2023-07-25 00:00:00+00:00"],
        tz="UTC",
        name="timestamp",
    )
    frame = pd.DataFrame(
        {
            "open": ["10", "11"],
            "high": ["12", "13"],
            "low": ["9", "10"],
            "close": ["11", "12"],
            "volume": ["100", "110"],
        },
        index=index,
    )
    frame.to_parquet(tmp_path / "eth_usdt_1d.parquet")

    loaded = load_cached_data(
        "ETH/USDT",
        "1d",
        data_dir=tmp_path,
        now=pd.Timestamp("2026-07-24 12:30:00+00:00"),
        validate_coverage=False,
    )

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz is not None
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert all(str(dtype) == "float64" for dtype in loaded.dtypes)
