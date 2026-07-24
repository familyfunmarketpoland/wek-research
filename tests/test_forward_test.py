from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import forward_test.runner as forward_runner
from forward_test.core import (
    COST_RATE,
    DEADLINE,
    FIRST_ELIGIBLE,
    TIMEFRAME_DELTA,
    WARMUP_BARS,
    ForwardTestError,
    _drawdown_exceeds_threshold,
    _h2_signals,
    initialize_artifacts,
    run_forward_test as _core_run_forward_test,
    state_projection_hash,
)
from forward_test.data import (
    OhlcvFetchResult,
    _create_spot_binance_exchange,
    default_since_for_warmup,
    fetch_binance_ohlcv,
    fetch_binance_ohlcv_bundle,
)
from forward_test.integrity import GENESIS_HEAD, build_ledger_entry
from research_lab.engine import run_signal_backtest
from research_lab.features import compute_features
from research_lab.hypotheses import Candidate, SignalBundle, generate_signals


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "forward_test"
    root.mkdir(parents=True)
    shutil.copy(Path("forward_test/prereg_forward.json"), root / "prereg_forward.json")
    initialize_artifacts(root)
    return root


def _frame(n: int, *, start: pd.Timestamp = FIRST_ELIGIBLE, open_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [open_price] * n,
            "high": [open_price * 1.01] * n,
            "low": [open_price * 0.99] * n,
            "close": [open_price] * n,
            "volume": [1000.0] * n,
        },
        index=idx,
    )
    return frame


def _with_warmup(frame: pd.DataFrame, *, bars: int = WARMUP_BARS) -> pd.DataFrame:
    if frame.index.min() < FIRST_ELIGIBLE:
        return frame
    warmup_index = pd.date_range(
        end=FIRST_ELIGIBLE - TIMEFRAME_DELTA,
        periods=bars,
        freq=TIMEFRAME_DELTA,
        tz="UTC",
    )
    anchor = float(frame["open"].iloc[0])
    warmup = pd.DataFrame(
        {
            "open": anchor,
            "high": anchor * 1.01,
            "low": anchor * 0.99,
            "close": anchor,
            "volume": 1000.0,
        },
        index=warmup_index,
    )
    return pd.concat([warmup, frame])


def run_forward_test(frame: pd.DataFrame, **kwargs):
    return _core_run_forward_test(_with_warmup(frame), **kwargs)


def _patch_signals(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, signal_at=(), exit_at=()) -> None:
    signal = pd.Series(0, index=frame.index, dtype=int)
    exit_signal = pd.Series(False, index=frame.index, dtype=bool)
    for i in signal_at:
        signal.iloc[i] = -1
    for i in exit_at:
        exit_signal.iloc[i] = True
    monkeypatch.setattr("forward_test.core._h2_signals", lambda f: (signal, exit_signal))


def _ledger_lines(root: Path) -> list[dict]:
    text = (root / "ledger.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _ts_for_test(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def test_initial_artifacts_are_genesis_consistent(tmp_path: Path) -> None:
    root = _root(tmp_path)
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))

    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""
    assert (root / "head.sha256").read_text(encoding="utf-8").strip() == GENESIS_HEAD
    assert state["ledger_head"] == GENESIS_HEAD
    assert state["ledger_entries"] == 0
    assert state["parameter_sha256"] == "a81471a4f44ed58246ef63bb4de420505aac8d725402bdcef9954e04497eac78"
    assert state["prereg_sha256"] == "0e83bddc6fea02ef19fdca69a0face24358da57a8901ca4d8849c0eca9b97c2b"
    assert state["created_at_utc"] == "2026-07-24T12:15:17Z"
    assert state["updated_at_utc"] == "2026-07-24T12:15:17Z"
    assert state["state_checkpoint_sha256"] == state_projection_hash(state)


@pytest.mark.parametrize("artifact", ["state.json", "ledger.jsonl", "head.sha256"])
def test_missing_runtime_artifact_cannot_implicitly_restart(
    tmp_path: Path,
    artifact: str,
) -> None:
    root = _root(tmp_path)
    untouched = {
        name: (root / name).read_bytes()
        for name in ("state.json", "ledger.jsonl", "head.sha256")
        if name != artifact
    }
    (root / artifact).unlink()

    with pytest.raises(ForwardTestError, match="refusing implicit reset"):
        _core_run_forward_test(_frame(1), root=root)

    assert not (root / artifact).exists()
    for name, contents in untouched.items():
        assert (root / name).read_bytes() == contents


def test_prereg_and_state_hash_drift_fail_before_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)
    (root / "prereg_forward.json").write_text((root / "prereg_forward.json").read_text() + "\n", encoding="utf-8")

    with pytest.raises(ForwardTestError, match="preregistration file hash drift"):
        run_forward_test(frame, root=root)

    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""


def test_state_parameter_hash_drift_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["parameter_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)

    with pytest.raises(ForwardTestError, match="state parameter_sha256 drift"):
        run_forward_test(frame, root=root)

    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""


def test_initial_warmup_requires_2210_immediately_preceding_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)
    short_root = _root(tmp_path / "short")

    with pytest.raises(ForwardTestError, match="2210 consecutive 4h warmup"):
        _core_run_forward_test(_with_warmup(frame, bars=WARMUP_BARS - 1), root=short_root)
    assert (short_root / "ledger.jsonl").read_text(encoding="utf-8") == ""

    exact_root = _root(tmp_path / "exact")
    result = _core_run_forward_test(_with_warmup(frame, bars=WARMUP_BARS), root=exact_root)
    assert result.processed == 1


def test_every_active_run_requires_full_feature_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(2)
    _patch_signals(monkeypatch, frame)

    first = _core_run_forward_test(_with_warmup(frame.iloc[:1]), root=root)
    assert first.processed == 1

    with pytest.raises(ForwardTestError, match="2210 consecutive 4h warmup"):
        _core_run_forward_test(frame, root=root)


def test_current_first_requires_warmup_and_can_start_from_open_quote(tmp_path: Path) -> None:
    eligible = _frame(1)
    short = _with_warmup(eligible, bars=WARMUP_BARS - 1).loc[lambda value: value.index < FIRST_ELIGIBLE]
    exact = _with_warmup(eligible, bars=WARMUP_BARS).loc[lambda value: value.index < FIRST_ELIGIBLE]
    quote = {"open_utc": "2026-07-25T00:00:00Z", "open": 100.0}
    now = pd.Timestamp("2026-07-25T00:17:00Z")

    short_root = _root(tmp_path / "short")
    with pytest.raises(ForwardTestError, match="2210 consecutive 4h warmup"):
        _core_run_forward_test(short, root=short_root, current_open=quote, now=now)

    exact_root = _root(tmp_path / "exact")
    result = _core_run_forward_test(exact, root=exact_root, current_open=quote, now=now)
    assert result.state["last_processed_open_utc"] is None
    assert result.state["benchmark"]["entry_price"] == 100.0
    assert result.state["last_open_mark_utc"] == "2026-07-25T00:00:00Z"


def test_prestart_current_open_is_validated_noop_without_writes(tmp_path: Path) -> None:
    root = _root(tmp_path)
    completed = _frame(1, start=FIRST_ELIGIBLE - 2 * TIMEFRAME_DELTA)
    before_state = (root / "state.json").read_bytes()

    result = _core_run_forward_test(
        completed,
        root=root,
        now=pd.Timestamp("2026-07-24T20:17:00Z"),
        current_open={"open_utc": "2026-07-24T20:00:00Z", "open": 100.0},
    )

    assert result.processed == 0
    assert result.appended == 0
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""
    assert (root / "state.json").read_bytes() == before_state


@pytest.mark.parametrize(
    ("constant", "drifted"),
    [
        ("HOLD_BARS", 21),
        ("DONCHIAN_WINDOW_BARS", 21),
        ("REALIZED_VOL_WINDOW_BARS", 21),
        ("COST_RATE", 0.002),
        ("SIDE", "long"),
        ("FIRST_ELIGIBLE", pd.Timestamp("2026-07-25T04:00:00Z")),
        ("DEADLINE", pd.Timestamp("2027-07-25T04:00:00Z")),
    ],
)
def test_runtime_constants_are_asserted_against_prereg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    drifted: object,
) -> None:
    root = _root(tmp_path)
    monkeypatch.setattr(f"forward_test.core.{constant}", drifted)

    with pytest.raises(ForwardTestError, match="runtime contract drift"):
        run_forward_test(_frame(1), root=root)
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""


def test_idempotency_and_observation_mutation_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(3)
    _patch_signals(monkeypatch, frame, signal_at=[0])

    first = run_forward_test(frame, root=root)
    ledger_after_first = (root / "ledger.jsonl").read_text(encoding="utf-8")
    second = run_forward_test(frame, root=root)

    assert first.processed == 3
    assert second.processed == 0
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ledger_after_first

    mutated = frame.copy()
    mutated.iloc[0, mutated.columns.get_loc("close")] = 99.0
    with pytest.raises(ForwardTestError, match="mutated"):
        run_forward_test(mutated, root=root)
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ledger_after_first


@pytest.mark.parametrize("field", ["equity", "observations", "closed_trades", "status", "pending_order"])
def test_state_checkpoint_detects_mutable_state_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame, signal_at=[0])
    run_forward_test(frame, root=root)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if field == "equity":
        state[field] = 2.0
    elif field == "observations":
        state[field]["2026-07-25T00:00:00Z"] = "0" * 64
    elif field == "closed_trades":
        state[field].append({"net_return": 0.1})
    elif field == "status":
        state[field] = "FAIL"
    elif field == "pending_order":
        state[field]["target_position"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ForwardTestError, match="checkpoint"):
        run_forward_test(frame, root=root)


def test_ledger_checkpoint_detects_recomputed_state_checkpoint_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame, signal_at=[0])
    run_forward_test(frame, root=root)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["equity"] = 2.0
    state["state_checkpoint_sha256"] = state_projection_hash(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ForwardTestError, match="latest ledger checkpoint"):
        run_forward_test(frame, root=root)


def test_latest_ledger_event_must_be_state_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)
    result = run_forward_test(frame, root=root)
    extra = build_ledger_entry(
        result.state["ledger_entries"],
        result.state["ledger_head"],
        "audit_note",
        {"note": "validly chained but not a checkpoint"},
    )
    ledger_path = root / "ledger.jsonl"
    ledger_path.write_text(
        ledger_path.read_text(encoding="utf-8") + json.dumps(extra, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "head.sha256").write_text(extra["entry_hash"] + "\n", encoding="utf-8")
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["ledger_head"] = extra["entry_hash"]
    state["hashes"]["ledger_head"] = extra["entry_hash"]
    state["ledger_entries"] += 1
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ForwardTestError, match="latest ledger event must be state_checkpoint"):
        run_forward_test(frame, root=root)


def test_ledger_tamper_reorder_truncate_and_head_mismatch_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(3)
    _patch_signals(monkeypatch, frame)
    run_forward_test(frame, root=root)

    original_ledger = (root / "ledger.jsonl").read_text(encoding="utf-8")
    lines = original_ledger.splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["equity"] = 2.0
    (root / "ledger.jsonl").write_text(json.dumps(tampered) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ForwardTestError, match="hash mismatch"):
        run_forward_test(frame, root=root)

    (root / "ledger.jsonl").write_text(original_ledger, encoding="utf-8")
    (root / "head.sha256").write_text("1" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ForwardTestError, match="head"):
        run_forward_test(frame, root=root)

    (root / "head.sha256").write_text(json.loads(lines[-1])["entry_hash"] + "\n", encoding="utf-8")
    if len(lines) > 1:
        (root / "ledger.jsonl").write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
        with pytest.raises(ForwardTestError, match="previous hash|sequence"):
            run_forward_test(frame, root=root)


def test_next_open_hold20_costs_and_trade_match_research_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(22)
    frame.iloc[1:21, frame.columns.get_loc("close")] = 100.0
    frame.iloc[21, frame.columns.get_loc("open")] = 97.0
    frame.iloc[21, frame.columns.get_loc("high")] = 101.0
    frame.iloc[21, frame.columns.get_loc("low")] = 96.0
    frame.iloc[21, frame.columns.get_loc("close")] = 97.0
    _patch_signals(monkeypatch, frame, signal_at=[0])

    result = run_forward_test(frame, root=root)

    candidate = Candidate(
        candidate_id="H2|solusdt|4h|side-short|hold_bars-20",
        hypothesis_id="H2",
        symbol="SOL/USDT",
        timeframe="4h",
        side="short",
        hold_bars=20,
    )
    signals = SignalBundle(
        candidate=candidate,
        kind="close",
        signal=pd.Series([-1] + [0] * 21, index=frame.index),
        exit_signal=pd.Series(False, index=frame.index),
        allow_reversal=False,
    )
    expected = run_signal_backtest(frame, signals, active_start=FIRST_ELIGIBLE, cost_rate_per_side=COST_RATE)

    assert result.status == "RUNNING"
    assert result.state["closed_trades_count"] == 1
    assert result.state["closed_trades"][0]["bars"] == 20
    assert result.state["closed_trades"][0]["reason"] == "time"
    assert result.state["position"] == 0
    assert result.state["equity"] == pytest.approx(float(expected.equity.iloc[-1]))
    assert result.state["closed_trades"][0]["net_return"] == pytest.approx(float(expected.trades.iloc[0]["net_return"]))
    hold_close = next(
        entry["payload"]
        for entry in _ledger_lines(root)
        if entry["event"] == "closed_candle" and entry["payload"]["open_utc"] == _ts_for_test(frame.index[20])
    )
    assert hold_close["reason"] == "hold_expiry"
    assert hold_close["diagnostics"]["hold_expiry"] is True


def test_held_position_open_gaps_without_pending_match_research_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(22)
    for i in range(2, 21):
        frame.iloc[i, frame.columns.get_loc("open")] = 100.0 + ((-1) ** i) * 2.0
        frame.iloc[i, frame.columns.get_loc("close")] = 100.0 + ((-1) ** i) * 1.0
        frame.iloc[i, frame.columns.get_loc("high")] = max(frame.iloc[i]["open"], frame.iloc[i]["close"]) + 1.0
        frame.iloc[i, frame.columns.get_loc("low")] = min(frame.iloc[i]["open"], frame.iloc[i]["close"]) - 1.0
    _patch_signals(monkeypatch, frame, signal_at=[0])

    result = run_forward_test(frame, root=root)

    candidate = Candidate(
        candidate_id="H2|solusdt|4h|side-short|hold_bars-20",
        hypothesis_id="H2",
        symbol="SOL/USDT",
        timeframe="4h",
        side="short",
        hold_bars=20,
    )
    signals = SignalBundle(
        candidate=candidate,
        kind="close",
        signal=pd.Series([-1] + [0] * 21, index=frame.index),
        exit_signal=pd.Series(False, index=frame.index),
        allow_reversal=False,
    )
    expected = run_signal_backtest(frame, signals, active_start=FIRST_ELIGIBLE, cost_rate_per_side=COST_RATE)
    assert result.state["equity"] == pytest.approx(float(expected.equity.iloc[-1]))


def test_opposite_channel_exit_and_same_close_reentry_after_open_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(4)
    _patch_signals(monkeypatch, frame, signal_at=[0, 2], exit_at=[1])

    result = run_forward_test(frame, root=root)

    assert result.state["closed_trades_count"] == 1
    assert result.state["closed_trades"][0]["reason"] == "opposite_channel"
    assert result.state["pending_order"] is None
    assert result.state["position"] == -1
    assert result.state["open_trade"]["entry_time"] == "2026-07-25T12:00:00Z"
    opposite_close = next(
        entry["payload"]
        for entry in _ledger_lines(root)
        if entry["event"] == "closed_candle" and entry["payload"]["open_utc"] == _ts_for_test(frame.index[1])
    )
    assert opposite_close["reason"] == "opposite_channel"
    assert opposite_close["diagnostics"]["opposite_channel"] is True


def test_drawdown_fail_is_strict_and_keeps_open_position(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(2)
    frame.iloc[1, frame.columns.get_loc("close")] = 130.0
    frame.iloc[1, frame.columns.get_loc("high")] = 131.0
    _patch_signals(monkeypatch, frame, signal_at=[0])

    result = run_forward_test(frame, root=root)

    assert result.status == "FAIL"
    assert result.state["terminal"]["reason"] == "drawdown"
    assert result.state["position"] == -1
    assert result.state["open_trade"] is not None
    assert result.state["closed_trades_count"] == 0
    assert result.state["max_drawdown"] > 0.25


def test_drawdown_threshold_comparison_is_literal_and_strict() -> None:
    assert _drawdown_exceeds_threshold(0.25) is False
    assert _drawdown_exceeds_threshold(np.nextafter(0.25, 1.0)) is True


def test_smallest_representable_drawdown_above_25_percent_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(2)
    drawdown = np.nextafter(0.25, 1.0)
    loss = 1.0 - (1.0 - drawdown) / (1.0 - COST_RATE)
    frame.iloc[1, frame.columns.get_loc("close")] = 100.0 * (1.0 + loss)
    frame.iloc[1, frame.columns.get_loc("high")] = 140.0
    _patch_signals(monkeypatch, frame, signal_at=[0])

    result = run_forward_test(frame, root=root)

    assert result.state["max_drawdown"] > 0.25
    assert result.status == "FAIL"
    assert result.state["terminal"]["reason"] == "drawdown"


def test_30th_trade_pass_and_fail_are_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pass_root = _root(tmp_path / "pass")
    pass_frame = _alternating_trade_frame(exit_open=99.0)
    _patch_signals(monkeypatch, pass_frame, signal_at=range(0, 61, 2), exit_at=range(1, 60, 2))
    passed = run_forward_test(pass_frame, root=pass_root)
    assert passed.status == "PASS"
    assert passed.state["closed_trades_count"] == 30
    assert passed.state["performance"]["net_return"] > 0
    assert passed.state["performance"]["per_trade_sharpe"] > 0
    assert passed.state["pending_order"] is None
    assert pass_frame.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ") not in passed.state["observations"]
    assert passed.state["benchmark"]["equity_curve"][-1]["mark"] == "open"

    fail_root = _root(tmp_path / "fail")
    fail_frame = _alternating_trade_frame(exit_open=100.2)
    _patch_signals(monkeypatch, fail_frame, signal_at=range(0, 60, 2), exit_at=range(1, 60, 2))
    failed = run_forward_test(fail_frame, root=fail_root)
    assert failed.status == "FAIL"
    assert failed.state["closed_trades_count"] == 30
    assert failed.state["terminal"]["reason"] == "closed_trades_target"


def test_deadline_underpowered_after_scheduled_open_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(2191)
    assert frame.index[-1] == pd.Timestamp("2027-07-25T00:00:00Z")
    _patch_signals(monkeypatch, frame, signal_at=[2189])

    result = run_forward_test(frame, root=root)

    assert result.status == "UNDERPOWERED"
    assert result.state["position"] == -1
    assert result.state["terminal"]["open_trade"] is not None
    assert "2027-07-25T00:00:00Z" not in result.state["observations"]
    assert result.state["performance"]["equity_curve"][-1]["open_utc"] == _ts_for_test(DEADLINE)
    assert result.state["performance"]["equity_curve"][-1]["mark"] == "open"
    assert result.state["benchmark"]["equity_curve"][-1]["open_utc"] == _ts_for_test(DEADLINE)
    assert result.state["benchmark"]["equity_curve"][-1]["mark"] == "open"
    deadline_event = next(entry for entry in _ledger_lines(root) if entry["event"] == "deadline_open")
    assert deadline_event["payload"]["open_utc"] == _ts_for_test(DEADLINE)
    assert deadline_event["payload"]["open"] == 100.0


def test_current_open_quote_fills_pending_without_open_candle_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame, signal_at=[0])
    current_open = {"open_utc": "2026-07-25T04:00:00Z", "open": 95.0}

    first = run_forward_test(frame, root=root, current_open=current_open)
    ledger_after_first = (root / "ledger.jsonl").read_text(encoding="utf-8")
    equity_after_first = first.state["equity"]
    second = run_forward_test(frame, root=root, current_open=current_open)

    assert first.state["position"] == -1
    assert first.state["pending_order"] is None
    assert first.state["last_open_mark_utc"] == "2026-07-25T04:00:00Z"
    assert list(first.state["observations"]) == ["2026-07-25T00:00:00Z"]
    assert first.state["open_trade"]["entry_price"] == 95.0
    assert second.appended == 0
    assert second.state["equity"] == equity_after_first
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ledger_after_first

    closed = _frame(2)
    closed.iloc[1, closed.columns.get_loc("open")] = 95.0
    closed.iloc[1, closed.columns.get_loc("close")] = 90.0
    closed.iloc[1, closed.columns.get_loc("low")] = 89.0
    _patch_signals(monkeypatch, closed)
    third = run_forward_test(closed, root=root)
    assert third.state["benchmark"]["equity"] == pytest.approx((1.0 - COST_RATE) * 0.9)

    revised_root = _root(tmp_path / "revised")
    _patch_signals(monkeypatch, frame, signal_at=[0])
    run_forward_test(frame, root=revised_root, current_open=current_open)
    revised = closed.copy()
    revised.iloc[1, revised.columns.get_loc("open")] = 96.0
    _patch_signals(monkeypatch, revised)
    with pytest.raises(ForwardTestError, match="booked candle open was mutated"):
        run_forward_test(revised, root=revised_root)


def test_current_open_marks_held_position_gap_without_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    first = _frame(1)
    _patch_signals(monkeypatch, first, signal_at=[0])
    run_forward_test(first, root=root, current_open={"open_utc": "2026-07-25T04:00:00Z", "open": 100.0})
    closed = _frame(2)
    _patch_signals(monkeypatch, closed)
    run_forward_test(closed, root=root)
    before = json.loads((root / "state.json").read_text(encoding="utf-8"))["equity"]

    marked = run_forward_test(
        closed,
        root=root,
        current_open={"open_utc": "2026-07-25T08:00:00Z", "open": 90.0},
    )

    assert marked.appended > 0
    assert marked.state["pending_order"] is None
    assert marked.state["position"] == -1
    assert marked.state["equity"] == pytest.approx(before * 1.1)


def test_current_open_pending_due_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame, signal_at=[0])

    with pytest.raises(ForwardTestError, match="current open must be exactly the next 4h open"):
        run_forward_test(frame, root=root, current_open={"open_utc": "2026-07-25T08:00:00Z", "open": 100.0})


def test_current_open_cannot_skip_next_open_while_flat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)

    with pytest.raises(ForwardTestError, match="2026-07-25T04:00:00Z"):
        run_forward_test(frame, root=root, current_open={"open_utc": "2026-07-25T08:00:00Z", "open": 100.0})
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("quote", "message"),
    [
        ({"open_utc": "2026-07-25T04:00:00", "open": 100.0}, "must be UTC"),
        ({"open_utc": "2026-07-25T06:00:00+02:00", "open": 100.0}, "must be UTC"),
        ({"open_utc": "2026-07-25T05:00:00Z", "open": 100.0}, "align"),
        ({"open_utc": "2026-07-25T04:00:00Z", "open": 0.0}, "finite and positive"),
        ({"open_utc": "2026-07-25T04:00:00Z", "open": -1.0}, "finite and positive"),
        ({"open_utc": "2026-07-25T04:00:00Z", "open": float("nan")}, "finite and positive"),
        ({"open_utc": "2026-07-25T04:00:00Z", "open": float("inf")}, "finite and positive"),
    ],
)
def test_current_open_requires_utc_alignment_and_positive_finite_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quote: dict[str, object],
    message: str,
) -> None:
    root = _root(tmp_path)
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)

    with pytest.raises(ForwardTestError, match=message):
        run_forward_test(frame, root=root, current_open=quote)
    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""


def test_now_fixes_current_boundary_and_rejects_unclosed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-07-25T06:17:00Z")
    frame = _frame(1)
    _patch_signals(monkeypatch, frame)
    valid_root = _root(tmp_path / "valid")
    valid = run_forward_test(
        frame,
        root=valid_root,
        now=now,
        current_open={"open_utc": "2026-07-25T04:00:00Z", "open": 100.0},
    )
    assert valid.processed == 1

    wrong_current_root = _root(tmp_path / "wrong_current")
    with pytest.raises(ForwardTestError, match=r"floor\(now\)"):
        run_forward_test(
            frame,
            root=wrong_current_root,
            now=now,
            current_open={"open_utc": "2026-07-25T08:00:00Z", "open": 100.0},
        )
    assert (wrong_current_root / "ledger.jsonl").read_text(encoding="utf-8") == ""

    unclosed_root = _root(tmp_path / "unclosed")
    unclosed = _frame(2)
    _patch_signals(monkeypatch, unclosed)
    with pytest.raises(ForwardTestError, match="not closed at now"):
        run_forward_test(
            unclosed,
            root=unclosed_root,
            now=now,
            current_open={"open_utc": "2026-07-25T04:00:00Z", "open": 100.0},
        )
    assert (unclosed_root / "ledger.jsonl").read_text(encoding="utf-8") == ""

    stale_root = _root(tmp_path / "stale")
    stale = _frame(1, start=FIRST_ELIGIBLE - TIMEFRAME_DELTA)
    with pytest.raises(ForwardTestError, match="frame is stale"):
        run_forward_test(
            stale,
            root=stale_root,
            now=now,
            current_open={"open_utc": "2026-07-25T04:00:00Z", "open": 100.0},
        )
    assert (stale_root / "ledger.jsonl").read_text(encoding="utf-8") == ""


def test_buy_hold_background_updates_without_affecting_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(2)
    frame.iloc[0, frame.columns.get_loc("close")] = 110.0
    frame.iloc[0, frame.columns.get_loc("high")] = 111.0
    frame.iloc[1, frame.columns.get_loc("open")] = 110.0
    frame.iloc[1, frame.columns.get_loc("close")] = 121.0
    frame.iloc[1, frame.columns.get_loc("high")] = 122.0
    _patch_signals(monkeypatch, frame)

    result = run_forward_test(frame, root=root)

    assert result.status == "RUNNING"
    assert result.state["equity"] == pytest.approx(1.0)
    assert result.state["benchmark"]["entry_price"] == 100.0
    assert result.state["benchmark"]["equity"] == pytest.approx((1.0 - COST_RATE) * 1.21)
    assert len(result.state["benchmark"]["equity_curve"]) == 2
    first_close = next(entry["payload"] for entry in _ledger_lines(root) if entry["event"] == "closed_candle")
    assert first_close["reason"] == "flat"
    assert first_close["diagnostics"]["flat"] is True


def test_fetch_binance_ohlcv_paginates_warmup_and_excludes_open_candle() -> None:
    since = default_since_for_warmup()
    current_open = since + pd.Timedelta(hours=4 * 2215)
    now = current_open + pd.Timedelta(hours=2)
    index = pd.date_range(since, periods=2225, freq="4h", tz="UTC")
    rows = [
        [int(ts.timestamp() * 1000), 100.0, 101.0, 99.0, 100.0, 1000.0]
        for ts in index
    ]
    exchange = _FakeExchange(rows)

    frame = fetch_binance_ohlcv(since=since, now=now, exchange=exchange, limit=5000)
    bundle = fetch_binance_ohlcv_bundle(since=since, now=now, exchange=_FakeExchange(rows), limit=5000)

    assert len(frame) == 2215
    assert len(bundle.completed) == 2215
    assert bundle.current_open == {"open_utc": current_open.strftime("%Y-%m-%dT%H:%M:%SZ"), "open": 100.0}
    assert len(frame) > 2210
    assert frame.index[0] == since
    assert frame.index[-1] == current_open - pd.Timedelta(hours=4)
    assert current_open not in frame.index
    assert len(exchange.calls) >= 3
    assert all(call["limit"] <= 1000 for call in exchange.calls)
    assert FIRST_ELIGIBLE - default_since_for_warmup() >= pd.Timedelta(hours=4 * 2210)


def test_fetch_binance_default_exchange_is_spot_data_api() -> None:
    exchange = _create_spot_binance_exchange()

    assert exchange.enableRateLimit is True
    assert exchange.options["defaultType"] == "spot"
    assert exchange.options["fetchMarkets"] == {"types": ["spot"], "loadAllOptions": False}
    assert exchange.urls["api"]["public"] == "https://data-api.binance.vision/api/v3"
    assert exchange.sign("exchangeInfo", "public", "GET", {})["url"] == (
        "https://data-api.binance.vision/api/v3/exchangeInfo"
    )
    assert exchange.sign(
        "klines",
        "public",
        "GET",
        {"symbol": "SOLUSDT", "interval": "4h"},
    )["url"] == "https://data-api.binance.vision/api/v3/klines?symbol=SOLUSDT&interval=4h"


def test_runner_uses_one_run_now_for_fetch_and_core(monkeypatch: pytest.MonkeyPatch) -> None:
    run_now = pd.Timestamp("2026-07-25T04:17:00Z")
    calls: dict[str, object] = {"utc_now_count": 0}

    def fake_utc_now() -> pd.Timestamp:
        calls["utc_now_count"] = int(calls["utc_now_count"]) + 1
        return run_now

    def fake_fetch(**kwargs):
        calls["fetch_now"] = kwargs["now"]
        calls["fetch_symbol"] = kwargs["symbol"]
        calls["fetch_timeframe"] = kwargs["timeframe"]
        return OhlcvFetchResult(completed=pd.DataFrame(), current_open=None)

    def fake_run(frame, **kwargs):
        calls["core_now"] = kwargs["now"]
        return SimpleNamespace(status="RUNNING", processed=0, appended=0, ledger_head="0" * 64)

    monkeypatch.setattr(forward_runner, "utc_now", fake_utc_now)
    monkeypatch.setattr(forward_runner, "fetch_binance_ohlcv_bundle", fake_fetch)
    monkeypatch.setattr(forward_runner, "run_forward_test", fake_run)
    monkeypatch.setattr(sys, "argv", ["forward-test"])

    forward_runner.main()

    assert calls["utc_now_count"] == 1
    assert calls["fetch_now"] is run_now
    assert calls["fetch_symbol"] == "SOL/USDT"
    assert calls["fetch_timeframe"] == "4h"
    assert calls["core_now"] is run_now


def test_root_path_with_holdout_component_is_rejected_before_io(tmp_path: Path) -> None:
    forbidden = tmp_path / "safe" / "holdout" / "forward_test"

    with pytest.raises(ForwardTestError, match="holdout component"):
        initialize_artifacts(forbidden)

    assert not forbidden.exists()


def test_local_h2_signals_match_research_lab_on_synthetic_frame() -> None:
    frame = _h2_semantics_frame(final_close=99.949)
    local_signal, local_exit = _h2_signals(frame)
    candidate = Candidate(
        candidate_id="H2|solusdt|4h|side-short|hold_bars-20",
        hypothesis_id="H2",
        symbol="SOL/USDT",
        timeframe="4h",
        side="short",
        hold_bars=20,
    )
    research = generate_signals(frame, candidate, features=compute_features(frame, timeframe="4h"))

    pd.testing.assert_series_equal(local_signal, research.signal)
    pd.testing.assert_series_equal(local_exit, research.exit_signal)


def test_local_h2_strict_donchian_and_prior_baseline_semantics() -> None:
    equal_low = _h2_semantics_frame(final_close=99.95)
    below_low = _h2_semantics_frame(final_close=99.949)
    equal_signal, equal_exit = _h2_signals(equal_low)
    below_signal, below_exit = _h2_signals(below_low)
    assert equal_signal.iloc[-1] == 0
    assert below_signal.iloc[-1] == -1

    equal_high = _h2_semantics_frame(final_close=100.05)
    above_high = _h2_semantics_frame(final_close=100.051)
    _, exit_equal = _h2_signals(equal_high)
    _, exit_above = _h2_signals(above_high)
    assert exit_equal.iloc[-1] == equal_exit.iloc[-1] == below_exit.iloc[-1] == False
    assert exit_above.iloc[-1] == True


def test_closed_candle_ledger_records_entry_filter_diagnostics(tmp_path: Path) -> None:
    passing = _h2_semantics_frame(final_close=99.949)
    passing.index = pd.date_range(end=FIRST_ELIGIBLE, periods=len(passing), freq=TIMEFRAME_DELTA, tz="UTC")
    pass_root = _root(tmp_path / "pass")
    _core_run_forward_test(passing, root=pass_root)
    passed = next(entry["payload"] for entry in _ledger_lines(pass_root) if entry["event"] == "closed_candle")
    assert passed["reason"] == "breakout_short"
    assert passed["diagnostics"] == {
        "breakout_short": True,
        "entry_rule_pass": True,
        "flat": True,
        "held_bars": None,
        "hold_expiry": False,
        "opposite_channel": False,
        "volatility_compression": True,
        "volatility_filter": "pass",
    }

    failing = _h2_semantics_frame(final_close=50.0)
    failing.index = pd.date_range(end=FIRST_ELIGIBLE, periods=len(failing), freq=TIMEFRAME_DELTA, tz="UTC")
    fail_root = _root(tmp_path / "fail")
    _core_run_forward_test(failing, root=fail_root)
    failed = next(entry["payload"] for entry in _ledger_lines(fail_root) if entry["event"] == "closed_candle")
    assert failed["reason"] == "volatility_filter_fail"
    assert failed["diagnostics"]["breakout_short"] is True
    assert failed["diagnostics"]["volatility_compression"] is False
    assert failed["diagnostics"]["volatility_filter"] == "fail"
    assert failed["diagnostics"]["entry_rule_pass"] is False


def test_gaps_duplicates_and_invalid_rows_fail_before_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    frame = _frame(3)
    _patch_signals(monkeypatch, frame)

    gap = frame.drop(frame.index[1])
    with pytest.raises(ForwardTestError, match="gaps|missing"):
        run_forward_test(gap, root=root)

    duplicate = pd.concat([frame, frame.iloc[[1]]]).sort_index()
    with pytest.raises(ForwardTestError, match="duplicates|duplicate|monotonic|not contain duplicates"):
        run_forward_test(duplicate, root=root)

    invalid = frame.copy()
    invalid.iloc[0, invalid.columns.get_loc("open")] = -1.0
    with pytest.raises(ForwardTestError, match="positive"):
        run_forward_test(invalid, root=root)

    inconsistent = frame.copy()
    inconsistent.iloc[0, inconsistent.columns.get_loc("high")] = 99.0
    with pytest.raises(ForwardTestError, match="inconsistent"):
        run_forward_test(inconsistent, root=root)

    assert (root / "ledger.jsonl").read_text(encoding="utf-8") == ""


def _alternating_trade_frame(*, exit_open: float) -> pd.DataFrame:
    frame = _frame(61)
    for i in range(30):
        entry = 1 + 2 * i
        exit_ = 2 + 2 * i
        frame.iloc[entry, frame.columns.get_loc("open")] = 100.0
        frame.iloc[entry, frame.columns.get_loc("close")] = 100.0
        frame.iloc[exit_, frame.columns.get_loc("open")] = exit_open
        frame.iloc[exit_, frame.columns.get_loc("close")] = exit_open
        frame.iloc[exit_, frame.columns.get_loc("high")] = max(101.0, exit_open * 1.01)
        frame.iloc[exit_, frame.columns.get_loc("low")] = min(99.0, exit_open * 0.99)
    assert math.isfinite(float(frame["close"].iloc[-1]))
    return frame


class _FakeExchange:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, int | None]] = []
        self._timestamps = [int(row[0]) for row in rows]

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        timeframe: str,
        since: int | None,
        limit: int,
    ) -> list[list[float]]:
        assert symbol == "SOL/USDT"
        assert timeframe == "4h"
        self.calls.append({"since": since, "limit": limit})
        if since is None:
            start = 0
        else:
            start = next((i for i, ts in enumerate(self._timestamps) if ts >= since), len(self.rows))
            if self.calls and len(self.calls) > 1:
                start = max(start - 1, 0)
        return self.rows[start : start + limit]


def _h2_semantics_frame(*, final_close: float) -> pd.DataFrame:
    n = 2235
    index = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="4h")
    base = 100.0 + 3.0 * np.sin(np.arange(n) / 3.0)
    close = base + np.where(np.arange(n) % 2 == 0, 1.5, -1.5)
    frame = pd.DataFrame(
        {
            "open": base,
            "high": np.maximum(base + 1.0, close),
            "low": np.minimum(base - 1.0, close),
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.iloc[-30:, frame.columns.get_loc("open")] = 100.0
    frame.iloc[-30:, frame.columns.get_loc("high")] = 100.05
    frame.iloc[-30:, frame.columns.get_loc("low")] = 99.95
    frame.iloc[-30:, frame.columns.get_loc("close")] = 100.0
    frame.iloc[-1, frame.columns.get_loc("close")] = final_close
    frame.iloc[-1, frame.columns.get_loc("high")] = max(100.05, final_close)
    frame.iloc[-1, frame.columns.get_loc("low")] = min(99.95, final_close)
    return frame
