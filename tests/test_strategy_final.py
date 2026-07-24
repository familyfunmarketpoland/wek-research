from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import strategy_final
from strategy_final import get_signal, get_signal_for_config, load_config


def _frame(length: int, freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq=freq, tz="UTC")
    open_ = pd.Series(100.0, index=index)
    close = pd.Series(100.0, index=index)
    high = pd.Series(101.0, index=index)
    low = pd.Series(99.0, index=index)
    volume = pd.Series(1000.0, index=index)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _config(**overrides: object) -> dict[str, object]:
    config = {
        "symbol": "BTC_USDT",
        "timeframe": "1h",
        "length": 14,
        "smooth": 3,
        "variant": "mean_reversion",
        "threshold": 50.0,
        "exit_bars": 2,
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "tick_size": 0.01,
        "research_run_id": "final-001",
    }
    config.update(overrides)
    return config


def _stub_wek(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    def fake_wek(frame: pd.DataFrame, length: int, smooth: int, tick_size: float) -> pd.Series:
        return pd.Series(values[: len(frame)], index=frame.index, dtype=float, name="WEK")

    monkeypatch.setattr(strategy_final, "wek", fake_wek)


def test_mean_reversion_returns_next_open_long_then_flat_on_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _frame(40)
    _stub_wek(monkeypatch, [0.0] * 24 + [60.0, 60.0, -1.0, -1.0])
    signal_entry = get_signal_for_config(df.iloc[:26], _config())
    signal_exit = get_signal_for_config(df.iloc[:27], _config())

    assert signal_entry == "LONG"
    assert signal_exit == "FLAT"


def test_prefix_invariance_ignores_future_rows() -> None:
    base = _frame(260)
    base.loc[base.index[:210], "close"] = range(100, 310)
    mutated = base.copy(deep=True)
    mutated.loc[mutated.index[240:], ["open", "high", "low", "close", "volume"]] = [5.0, 6.0, 4.0, 5.0, 9999.0]
    config = _config(variant="trend_filter", exit_bars=5)

    for end in (205, 220, 239):
        assert get_signal_for_config(base.iloc[:end], config) == get_signal_for_config(mutated.iloc[:end], config)


def test_time_exit_replays_state_causally(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _frame(40)
    config = _config(exit_bars=2)
    _stub_wek(monkeypatch, [0.0] * 24 + [60.0, 60.0, 60.0, 60.0, 60.0])

    assert get_signal_for_config(df.iloc[:25], config) == "LONG"
    assert get_signal_for_config(df.iloc[:26], config) == "LONG"
    assert get_signal_for_config(df.iloc[:27], config) == "FLAT"


def test_trend_filter_requires_close_above_causal_ema200(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _frame(260)
    df.loc[:, "open"] = 100.0
    df.loc[:, "close"] = 90.0
    df.loc[:, "high"] = 101.0
    df.loc[:, "low"] = 89.0
    df.loc[df.index[:205], "close"] = 100.0
    df.loc[df.index[205:], "close"] = 80.0
    _stub_wek(monkeypatch, [0.0] * 205 + [60.0, 60.0, 60.0, 60.0])

    config = _config(variant="trend_filter", exit_bars=5)
    blocked = get_signal_for_config(df.iloc[:206], config)

    df.loc[df.index[205:], "close"] = 140.0
    df.loc[df.index[205:], "high"] = 141.0
    df.loc[df.index[205:], "low"] = 99.0
    allowed = get_signal_for_config(df.iloc[:206], config)

    assert blocked == "FLAT"
    assert allowed == "LONG"


def test_long_short_emits_short_and_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _frame(40)
    config = _config(variant="long_short", exit_bars=5)
    _stub_wek(monkeypatch, [0.0] * 21 + [-60.0, -60.0, 1.0, 1.0, 1.0])

    assert get_signal_for_config(df.iloc[:22], config) == "SHORT"
    assert get_signal_for_config(df.iloc[:23], config) == "SHORT"
    assert get_signal_for_config(df.iloc[:24], config) == "FLAT"


def test_breakout_uses_prior_donchian_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _frame(50)
    highs = [100.0 + (i % 3) for i in range(len(df))]
    lows = [90.0 - (i % 2) for i in range(len(df))]
    closes = [95.0] * len(df)
    opens = [95.0] * len(df)
    highs[20] = 110.0
    closes[21] = 109.0
    highs[21] = 109.5
    highs[22] = 111.0
    closes[22] = 111.5
    highs[23] = 112.0
    closes[23] = 88.0
    lows[23] = 87.0
    df["open"] = opens
    df["high"] = highs
    df["low"] = lows
    df["close"] = closes
    _stub_wek(monkeypatch, [60.0] * 50)

    config = _config(variant="breakout", threshold=50.0, exit_bars=10)

    assert get_signal_for_config(df.iloc[:22], config) == "FLAT"
    assert get_signal_for_config(df.iloc[:23], config) == "LONG"
    assert get_signal_for_config(df.iloc[:24], config) == "FLAT"


def test_missing_and_bad_config_raise_clear_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="final config not found"):
        load_config(missing_path)

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"symbol": "BTC_USDT"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(bad_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold", 0.0, "threshold must be positive"),
        ("threshold", float("inf"), "threshold must be finite"),
        ("fee_rate", -0.001, "fee_rate must be nonnegative"),
        ("slippage_rate", float("nan"), "slippage_rate must be finite"),
    ],
)
def test_config_rejects_invalid_strategy_rates(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        get_signal_for_config(_frame(1), _config(**{field: value}))


def test_default_data_path_normalizes_slash_delimited_symbol() -> None:
    config = strategy_final._parse_config(_config(symbol="BTC/USDT"))

    assert strategy_final._default_data_path(config) == (
        strategy_final.PROJECT_DIR / "data" / "btc_usdt_1h.parquet"
    )


def test_get_signal_loads_default_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "final_config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    monkeypatch.setattr(strategy_final, "DEFAULT_CONFIG_PATH", config_path)
    _stub_wek(monkeypatch, [0.0] * 24 + [60.0, 60.0, -1.0, -1.0])

    df = _frame(26)

    assert get_signal(df) == "LONG"
