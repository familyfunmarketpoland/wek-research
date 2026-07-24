from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, Mapping

import pandas as pd

from research_lab.config import PRE_REGISTERED_CONFIG, validate_preregistered_config
from research_lab.features import compute_features, validate_ohlcv_frame


Side = Literal["long", "short"]
Mode = Literal["continuation", "reversal"]
SignalKind = Literal["close", "open_target"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    hypothesis_id: str
    symbol: str
    timeframe: str
    side: Side | None = None
    session_utc: str | None = None
    mode: Mode | None = None
    hold_bars: int | None = None
    streak_length: int | None = None

    @property
    def dataset(self) -> str:
        return f"{self.symbol.lower().replace('/', '_')}_{self.timeframe}"

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis_id": self.hypothesis_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "session_utc": self.session_utc,
            "mode": self.mode,
            "hold_bars": self.hold_bars,
            "streak_length": self.streak_length,
        }


@dataclass(frozen=True)
class SignalBundle:
    candidate: Candidate
    kind: SignalKind
    signal: pd.Series
    exit_signal: pd.Series
    allow_reversal: bool = False


def enumerate_candidates(config: Mapping[str, object] = PRE_REGISTERED_CONFIG) -> list[Candidate]:
    """Enumerate the frozen pre-registered candidate universe exactly once."""

    validate_preregistered_config(config)
    symbols = list(config["universe"]["symbols"])  # type: ignore[index]
    candidates: list[Candidate] = []
    for hypothesis in config["hypotheses"]:  # type: ignore[index]
        hypothesis_id = str(hypothesis["id"])
        timeframes = list(hypothesis["applicability"]["timeframes"])
        axes = hypothesis["parameter_grid"]["execution_axes"]
        axis_names = list(axes.keys())
        for symbol, timeframe in product(symbols, timeframes):
            for values in product(*(axes[name] for name in axis_names)):
                params = dict(zip(axis_names, values, strict=True))
                candidate = Candidate(
                    candidate_id=_candidate_id(
                        hypothesis_id=hypothesis_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        params=params,
                    ),
                    hypothesis_id=hypothesis_id,
                    symbol=str(symbol),
                    timeframe=str(timeframe),
                    side=params.get("side"),
                    session_utc=params.get("session_utc"),
                    mode=params.get("mode"),
                    hold_bars=int(params["hold_bars"]) if "hold_bars" in params else (1 if hypothesis_id == "H4" else None),
                    streak_length=int(params["streak_length"]) if "streak_length" in params else None,
                )
                candidates.append(candidate)
    return candidates


def candidate_frame(config: Mapping[str, object] = PRE_REGISTERED_CONFIG) -> pd.DataFrame:
    return pd.DataFrame([candidate.as_dict() for candidate in enumerate_candidates(config)])


def generate_signals(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    features: pd.DataFrame | None = None,
) -> SignalBundle:
    validate_ohlcv_frame(frame)
    if features is None:
        features = compute_features(frame, timeframe=candidate.timeframe)
    features = features.reindex(frame.index)

    if candidate.hypothesis_id == "H1":
        signal = (
            (frame["volume"].astype(float) > 3.0 * features["prior_volume_sma20"])
            & (features["return_z20_prior"] < -2.0)
        )
        return _bundle(candidate, _signed(signal, 1))

    if candidate.hypothesis_id in {"H2", "H6"}:
        if candidate.side not in {"long", "short"}:
            raise ValueError(f"{candidate.hypothesis_id} requires side")
        compressed = features["rv20"] < features["rv20_prior_year_lower_tercile"]
        if candidate.hypothesis_id == "H6":
            compressed = compressed & (
                features["volume_entropy20"] < features["entropy20_prior_year_median"]
            )
        close = frame["close"].astype(float)
        if candidate.side == "long":
            signal = close > features["donchian_high20_prior"]
            exit_signal = close < features["donchian_low20_prior"]
            signed_signal = _signed(signal & compressed, 1)
        else:
            signal = close < features["donchian_low20_prior"]
            exit_signal = close > features["donchian_high20_prior"]
            signed_signal = _signed(signal & compressed, -1)
        return _bundle(candidate, signed_signal, exit_signal=exit_signal.fillna(False))

    if candidate.hypothesis_id == "H3":
        if candidate.side not in {"long", "short"} or candidate.session_utc not in {"Asia", "USA"}:
            raise ValueError("H3 requires side and session_utc")
        hours = frame.index.hour
        if candidate.session_utc == "Asia":
            in_session = (hours >= 0) & (hours <= 7)
            exit_hour = hours == 8
        else:
            in_session = (hours >= 13) & (hours <= 20)
            exit_hour = hours == 21
        side_value = 1 if candidate.side == "long" else -1
        target = pd.Series(0, index=frame.index, dtype=int, name="signal")
        target.loc[in_session] = side_value
        target.loc[exit_hour] = 0
        return SignalBundle(
            candidate=candidate,
            kind="open_target",
            signal=target,
            exit_signal=pd.Series(False, index=frame.index, dtype=bool, name="exit_signal"),
        )

    if candidate.hypothesis_id == "H4":
        if candidate.mode not in {"continuation", "reversal"} or candidate.streak_length is None:
            raise ValueError("H4 requires mode and streak_length")
        n = candidate.streak_length
        up_exact = (features["up_streak"] == n) & (features["up_streak"].shift(1, fill_value=0) == n - 1)
        down_exact = (features["down_streak"] == n) & (features["down_streak"].shift(1, fill_value=0) == n - 1)
        signal = pd.Series(0, index=frame.index, dtype=int, name="signal")
        if candidate.mode == "continuation":
            signal.loc[up_exact] = 1
            signal.loc[down_exact] = -1
        else:
            signal.loc[up_exact] = -1
            signal.loc[down_exact] = 1
        return _bundle(candidate, signal, allow_reversal=True)

    if candidate.hypothesis_id == "H5":
        if candidate.side not in {"long", "short"}:
            raise ValueError("H5 requires side")
        setup = features["nr7"].shift(1, fill_value=False)
        if candidate.side == "long":
            signal = setup & (frame["close"].astype(float) > frame["high"].astype(float).shift(1))
            return _bundle(candidate, _signed(signal, 1))
        signal = setup & (frame["close"].astype(float) < frame["low"].astype(float).shift(1))
        return _bundle(candidate, _signed(signal, -1))

    raise ValueError(f"unknown hypothesis_id: {candidate.hypothesis_id}")


def _bundle(
    candidate: Candidate,
    signal: pd.Series,
    *,
    exit_signal: pd.Series | None = None,
    allow_reversal: bool = False,
) -> SignalBundle:
    if exit_signal is None:
        exit_signal = pd.Series(False, index=signal.index, dtype=bool, name="exit_signal")
    return SignalBundle(
        candidate=candidate,
        kind="close",
        signal=signal.astype(int).rename("signal"),
        exit_signal=exit_signal.astype(bool).rename("exit_signal"),
        allow_reversal=allow_reversal,
    )


def _signed(mask: pd.Series, value: int) -> pd.Series:
    signal = pd.Series(0, index=mask.index, dtype=int, name="signal")
    signal.loc[mask.fillna(False)] = value
    return signal


def _candidate_id(*, hypothesis_id: str, symbol: str, timeframe: str, params: dict[str, object]) -> str:
    parts = [
        hypothesis_id,
        symbol.replace("/", "").lower(),
        timeframe,
    ]
    for key in ("side", "session_utc", "mode", "hold_bars", "streak_length"):
        if key in params:
            parts.append(f"{key}-{params[key]}")
    return "|".join(parts)
