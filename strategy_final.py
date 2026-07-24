from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from wek import crossover, crossunder, wek


Signal = Literal["LONG", "SHORT", "FLAT"]
Variant = Literal["mean_reversion", "trend_filter", "long_short", "breakout"]

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
REQUIRED_CONFIG_KEYS = (
    "symbol",
    "timeframe",
    "length",
    "smooth",
    "variant",
    "threshold",
    "exit_bars",
    "fee_rate",
    "slippage_rate",
    "tick_size",
)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "results" / "final_config.json"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    timeframe: str
    length: int
    smooth: int
    variant: Variant
    threshold: float
    exit_bars: int
    fee_rate: float
    slippage_rate: float
    tick_size: float
    metadata: dict[str, Any]


def load_config(path: str | Path | None = None) -> StrategyConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"final config not found at {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config at {config_path}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    return _parse_config(raw)


def get_signal(df: pd.DataFrame) -> Signal:
    return get_signal_for_config(df, load_config())


def get_signal_for_config(df: pd.DataFrame, config: StrategyConfig | Mapping[str, Any]) -> Signal:
    parsed_config = config if isinstance(config, StrategyConfig) else _parse_config(config)
    frame = _validate_frame(df)
    if frame.empty:
        return "FLAT"

    wek_values = wek(
        frame,
        length=parsed_config.length,
        smooth=parsed_config.smooth,
        tick_size=parsed_config.tick_size,
    )
    ema200 = frame["close"].ewm(span=200, min_periods=200, adjust=False).mean()
    prior_high = frame["high"].shift(1).rolling(20, min_periods=20).max()
    prior_low = frame["low"].shift(1).rolling(20, min_periods=20).min()
    cross_up = crossover(wek_values, parsed_config.threshold)
    cross_down = crossunder(wek_values, -parsed_config.threshold)

    position: Signal = "FLAT"
    entry_bar: int | None = None

    for i in range(len(frame)):
        held_bars = (i - entry_bar + 1) if entry_bar is not None else 0
        next_position = _next_position(
            variant=parsed_config.variant,
            position=position,
            held_bars=held_bars,
            exit_bars=parsed_config.exit_bars,
            wek_value=wek_values.iloc[i],
            close_value=float(frame["close"].iloc[i]),
            ema200_value=ema200.iloc[i],
            prior_high_value=prior_high.iloc[i],
            prior_low_value=prior_low.iloc[i],
            cross_up=bool(cross_up.iloc[i]),
            cross_down=bool(cross_down.iloc[i]),
            threshold=parsed_config.threshold,
        )

        if i == len(frame) - 1:
            return next_position

        if next_position != position:
            position = next_position
            entry_bar = (i + 1) if position != "FLAT" else None

    return position


def _parse_config(config: Mapping[str, Any]) -> StrategyConfig:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")

    variant = config["variant"]
    if variant not in {"mean_reversion", "trend_filter", "long_short", "breakout"}:
        raise ValueError(f"unsupported variant: {variant}")

    length = _as_int(config["length"], "length", minimum=2)
    smooth = _as_int(config["smooth"], "smooth", minimum=1)
    exit_bars = _as_int(config["exit_bars"], "exit_bars", minimum=1)
    threshold = _as_float(config["threshold"], "threshold", positive=True)
    fee_rate = _as_float(config["fee_rate"], "fee_rate", nonnegative=True)
    slippage_rate = _as_float(config["slippage_rate"], "slippage_rate", nonnegative=True)
    tick_size = _as_float(config["tick_size"], "tick_size", positive=True)

    metadata = {key: value for key, value in dict(config).items() if key not in REQUIRED_CONFIG_KEYS}
    return StrategyConfig(
        symbol=str(config["symbol"]),
        timeframe=str(config["timeframe"]),
        length=length,
        smooth=smooth,
        variant=variant,
        threshold=threshold,
        exit_bars=exit_bars,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        tick_size=tick_size,
        metadata=metadata,
    )


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if df.index.has_duplicates:
        raise ValueError("df index must be unique")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted ascending")

    frame = df.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def _next_position(
    *,
    variant: Variant,
    position: Signal,
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
) -> Signal:
    time_exit = position != "FLAT" and held_bars >= exit_bars

    if variant == "mean_reversion":
        if position == "LONG":
            if pd.notna(wek_value) and wek_value <= 0:
                return "FLAT"
            if time_exit:
                return "FLAT"
        elif cross_up:
            return "LONG"
        return position

    if variant == "trend_filter":
        if position == "LONG":
            if pd.notna(wek_value) and wek_value <= 0:
                return "FLAT"
            if time_exit:
                return "FLAT"
        elif cross_up and pd.notna(ema200_value) and close_value > ema200_value:
            return "LONG"
        return position

    if variant == "long_short":
        if position == "LONG":
            if cross_down:
                return "SHORT"
            if pd.notna(wek_value) and wek_value <= 0:
                return "FLAT"
            if time_exit:
                return "FLAT"
        elif position == "SHORT":
            if cross_up:
                return "LONG"
            if pd.notna(wek_value) and wek_value >= 0:
                return "FLAT"
            if time_exit:
                return "FLAT"
        else:
            if cross_up:
                return "LONG"
            if cross_down:
                return "SHORT"
        return position

    if variant == "breakout":
        if position == "LONG":
            if pd.notna(prior_low_value) and close_value < prior_low_value:
                return "FLAT"
            if time_exit:
                return "FLAT"
        elif (
            pd.notna(prior_high_value)
            and pd.notna(wek_value)
            and close_value > prior_high_value
            and wek_value > threshold
        ):
            return "LONG"
        return position

    raise ValueError(f"unsupported variant: {variant}")


def _as_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{field_name} must be an integer")
    numeric = int(value)
    if numeric < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return numeric


def _as_float(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{field_name} must be positive")
    if nonnegative and numeric < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return numeric


def _default_data_path(config: StrategyConfig) -> Path:
    symbol = config.symbol.lower().replace("/", "_").replace("-", "_")
    timeframe = config.timeframe.lower()
    return PROJECT_DIR / "data" / f"{symbol}_{timeframe}.parquet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Return desired next-open WEK strategy signal.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to final_config.json")
    parser.add_argument("--data", type=Path, help="Optional OHLCV parquet or CSV path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data_path = args.data or _default_data_path(config)
    if not data_path.exists():
        raise FileNotFoundError(f"market data not found at {data_path}")

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        df = pd.read_parquet(data_path)
    signal = get_signal_for_config(df, config)
    print(
        json.dumps(
            {
                "signal": signal,
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "config_path": str(Path(args.config)),
                "data_path": str(data_path),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
