from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forward_test.integrity import (
    GENESIS_HEAD,
    IntegrityError,
    append_ledger_entries,
    build_ledger_entry,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    verify_ledger,
    write_json_atomic,
    write_text_atomic,
)


PREREG_COMMIT = "16c82a3"
PREREG_FILE_SHA256 = "0e83bddc6fea02ef19fdca69a0face24358da57a8901ca4d8849c0eca9b97c2b"
PARAMETER_SHA256 = "a81471a4f44ed58246ef63bb4de420505aac8d725402bdcef9954e04497eac78"
STUDY_ID = "h2_solusdt_4h_short_hold20_forward_2026"
CANDIDATE_ID = "H2|solusdt|4h|side-short|hold_bars-20"
HYPOTHESIS_ID = "H2"
SYMBOL = "SOL/USDT"
TIMEFRAME = "4h"
TIMEFRAME_HOURS = 4
FIRST_ELIGIBLE = pd.Timestamp("2026-07-25T00:00:00Z")
DEADLINE = pd.Timestamp("2027-07-25T00:00:00Z")
TIMEFRAME_DELTA = pd.Timedelta(hours=TIMEFRAME_HOURS)
INITIAL_EQUITY = 1.0
COST_RATE = 0.0015
DRAWDOWN_FAIL_THRESHOLD = 0.25
CLOSED_TRADES_TARGET = 30
HOLD_BARS = 20
SIDE = "short"
TARGET_POSITION = -1
ALLOW_REVERSAL = False
DONCHIAN_WINDOW_BARS = 20
REALIZED_VOL_WINDOW_BARS = 20
REALIZED_VOL_STD_DDOF = 0
HISTORY_WINDOW_BARS = 2190
HISTORY_QUANTILE_NUMERATOR = 1
HISTORY_QUANTILE_DENOMINATOR = 3
WARMUP_BARS = 2210
GENESIS_STATE_CHECKPOINT_SHA256 = "7f3b61e0682dc9362d169facbd41c08696d64f70dfbf96ac601ef797fd1a3bcd"


class ForwardTestError(RuntimeError):
    """Raised when the deterministic forward-test runner cannot continue."""


@dataclass(frozen=True)
class ForwardTestResult:
    status: str
    processed: int
    appended: int
    ledger_head: str
    state: dict[str, Any]


def run_forward_test(
    frame: pd.DataFrame,
    *,
    root: Path | str = Path("forward_test"),
    now: pd.Timestamp | None = None,
    current_open: dict[str, Any] | tuple[pd.Timestamp, float] | None = None,
) -> ForwardTestResult:
    """Process newly closed synthetic/public OHLCV candles into forward-test state.

    The caller supplies already fetched OHLCV. The runner performs all schema,
    preregistration, ledger, state, and historical-observation checks before it
    mutates any artifact.
    """

    root = Path(root)
    paths = _paths(root)
    prereg = _load_and_assert_prereg(paths["prereg"])
    _assert_runtime_artifacts_exist(paths)
    frame = _validate_frame(frame)
    run_now = _validate_run_now(now)
    current_quote = _validate_current_open(current_open, run_now=run_now)
    _assert_completed_rows_closed(frame, run_now)
    state = _load_state(paths["state"], prereg)
    _assert_state_matches_prereg(state)
    ledger_head, ledger_count, _ = _verify_artifacts(paths, state)
    _assert_observation_history_immutable(state, frame)
    _assert_initial_warmup(state, frame, current_quote)

    if state["status"] in {"PASS", "FAIL", "UNDERPOWERED"}:
        return ForwardTestResult(state["status"], 0, 0, ledger_head, state)

    signal, exit_signal = _h2_signals(frame)
    diagnostics = _h2_diagnostics(frame)
    next_state = json.loads(json.dumps(state))
    new_entries: list[dict[str, Any]] = []
    head = ledger_head
    sequence = ledger_count
    processed = 0

    for timestamp, row in frame.loc[frame.index >= FIRST_ELIGIBLE].iterrows():
        if timestamp < DEADLINE:
            stamp = _ts(timestamp)
            if stamp in next_state["observations"]:
                continue
            if _has_unprocessed_gap(next_state, timestamp):
                raise ForwardTestError("cannot process a candle while an earlier eligible candle is missing")
            event_payloads = _process_closed_candle(
                next_state,
                timestamp,
                row,
                int(signal.loc[timestamp]),
                bool(exit_signal.loc[timestamp]),
                diagnostics.loc[timestamp].to_dict(),
            )
            for event, payload in event_payloads:
                head, sequence = _add_entry(new_entries, sequence, head, event, payload)
            processed += 1
            if next_state["status"] in {"PASS", "FAIL"}:
                break
            continue

        if timestamp == DEADLINE and next_state["status"] == "RUNNING":
            deadline_payloads = _process_deadline_open(next_state, timestamp, row)
            for event, payload in deadline_payloads:
                head, sequence = _add_entry(new_entries, sequence, head, event, payload)
            break
        break

    if next_state["status"] == "RUNNING" and current_quote is not None:
        if not _is_pristine_prestart_current(next_state, current_quote[0]):
            _assert_current_open_is_next(next_state, current_quote[0])
            current_payloads = _process_current_open(next_state, current_quote)
            for event, payload in current_payloads:
                head, sequence = _add_entry(new_entries, sequence, head, event, payload)

    if not new_entries:
        return ForwardTestResult(next_state["status"], processed, 0, ledger_head, next_state)

    next_state["updated_at_utc"] = _ts(run_now if run_now is not None else pd.Timestamp(datetime.now(timezone.utc)))
    checkpoint_hash = state_projection_hash(next_state)
    head, sequence = _add_entry(
        new_entries,
        sequence,
        head,
        "state_checkpoint",
        {"state_sha256": checkpoint_hash, "open_utc": next_state.get("last_processed_open_utc")},
    )
    next_state["ledger_head"] = head
    next_state["ledger_entries"] = sequence
    next_state["hashes"]["ledger_head"] = head
    next_state["state_checkpoint_sha256"] = checkpoint_hash

    append_ledger_entries(paths["ledger"], new_entries)
    write_text_atomic(paths["head"], head + "\n")
    write_json_atomic(paths["state"], next_state)
    return ForwardTestResult(next_state["status"], processed, len(new_entries), head, next_state)


def initial_state(prereg: dict[str, Any] | None = None) -> dict[str, Any]:
    if prereg is None:
        prereg = _load_and_assert_prereg(Path("forward_test/prereg_forward.json"))
    else:
        _assert_runtime_contract(prereg)
    registered_at = str(prereg["study"]["registered_at_utc"])
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "RUNNING",
        "terminal": None,
        "prereg_commit": PREREG_COMMIT,
        "prereg_sha256": PREREG_FILE_SHA256,
        "parameter_sha256": PARAMETER_SHA256,
        "candidate_id": prereg["frozen_candidate"]["candidate_id"],
        "first_eligible_open_utc": _ts(FIRST_ELIGIBLE),
        "underpowered_deadline_utc": _ts(DEADLINE),
        "last_processed_open_utc": None,
        "last_open_mark_utc": None,
        "open_marks": {},
        "last_close": None,
        "eligible_index": -1,
        "equity": INITIAL_EQUITY,
        "high_watermark": INITIAL_EQUITY,
        "max_drawdown": 0.0,
        "position": 0,
        "pending_order": None,
        "open_trade": None,
        "closed_trades": [],
        "closed_trades_count": 0,
        "performance": {
            "initial_equity": INITIAL_EQUITY,
            "net_return": 0.0,
            "per_trade_sharpe": None,
            "equity_curve": [],
        },
        "benchmark": {
            "name": "SOL/USDT buy-and-hold",
            "position": 1,
            "entry_open_utc": _ts(FIRST_ELIGIBLE),
            "entry_price": None,
            "entry_cost_rate": COST_RATE,
            "equity": None,
            "net_return": None,
            "equity_curve": [],
        },
        "observations": {},
        "hashes": {
            "ledger_head": GENESIS_HEAD,
            "prereg_sha256": PREREG_FILE_SHA256,
            "parameter_sha256": PARAMETER_SHA256,
        },
        "state_checkpoint_sha256": GENESIS_STATE_CHECKPOINT_SHA256,
        "ledger_head": GENESIS_HEAD,
        "ledger_entries": 0,
        "created_at_utc": registered_at,
        "updated_at_utc": registered_at,
    }


def initialize_artifacts(root: Path | str = Path("forward_test")) -> dict[str, Any]:
    root = Path(root)
    paths = _paths(root)
    prereg = _load_and_assert_prereg(paths["prereg"])
    if any(paths[name].exists() for name in ("state", "ledger", "head")):
        raise ForwardTestError("forward-test artifacts already exist")
    state = initial_state(prereg)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(paths["ledger"], "")
    write_text_atomic(paths["head"], GENESIS_HEAD + "\n")
    write_json_atomic(paths["state"], state)
    return state


def _process_closed_candle(
    state: dict[str, Any],
    timestamp: pd.Timestamp,
    row: pd.Series,
    signal: int,
    exit_signal: bool,
    signal_diagnostics: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    stamp = _ts(timestamp)
    obs_hash = _row_hash(timestamp, row)
    state["eligible_index"] += 1
    eligible_index = int(state["eligible_index"])
    events: list[tuple[str, dict[str, Any]]] = []
    open_price = float(row["open"])
    close_price = float(row["close"])

    open_mark = _apply_open_gap(state, timestamp, open_price)
    if open_mark is not None:
        events.append(("open_mark", open_mark))
    fill_payload = _execute_pending_if_due(state, timestamp, open_price, eligible_index=eligible_index)
    if fill_payload is not None:
        events.append(("fill", fill_payload))
        if int(state["closed_trades_count"]) == CLOSED_TRADES_TARGET:
            _mark_buy_hold_open(state, timestamp, open_price)
            _record_candidate_open_mark(state, timestamp)
            verdict = _powered_verdict(state)
            _set_terminal(state, verdict, timestamp, "closed_trades_target")
            events.append(("terminal", _terminal_payload(state, timestamp, "closed_trades_target")))
            return events

    target = int(state["position"])
    equity = float(state["equity"]) * (1.0 + target * (close_price / open_price - 1.0))
    state["equity"] = equity
    _mark_buy_hold(state, timestamp, open_price, close_price)
    _update_equity_metrics(state, timestamp)
    dd = _update_drawdown(state, equity)

    state["observations"][stamp] = obs_hash
    state["last_processed_open_utc"] = stamp
    state["last_close"] = close_price

    decision_diagnostics = _decision_diagnostics(state, signal, exit_signal, signal_diagnostics)
    decision = _schedule_next_order(state, timestamp, signal, exit_signal)
    event_payload = {
        "open_utc": stamp,
        "observation_hash": obs_hash,
        "open": open_price,
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": close_price,
        "volume": float(row["volume"]),
        "signal": int(signal),
        "exit_signal": bool(exit_signal),
        "reason": decision_diagnostics["reason"],
        "diagnostics": decision_diagnostics["diagnostics"],
        "decision": decision,
        "equity": float(state["equity"]),
        "position": int(state["position"]),
        "drawdown": float(dd),
        "closed_trades_count": int(state["closed_trades_count"]),
    }
    events.append(("closed_candle", event_payload))

    if _drawdown_exceeds_threshold(dd):
        _set_terminal(state, "FAIL", timestamp, "drawdown")
        events.append(("terminal", _terminal_payload(state, timestamp, "drawdown")))
    elif int(state["closed_trades_count"]) == CLOSED_TRADES_TARGET:
        verdict = _powered_verdict(state)
        _set_terminal(state, verdict, timestamp, "closed_trades_target")
        events.append(("terminal", _terminal_payload(state, timestamp, "closed_trades_target")))
    return events


def _process_deadline_open(state: dict[str, Any], timestamp: pd.Timestamp, row: pd.Series) -> list[tuple[str, dict[str, Any]]]:
    open_price = float(row["open"])
    events: list[tuple[str, dict[str, Any]]] = []
    open_mark = _apply_open_gap(state, timestamp, open_price)
    if open_mark is not None:
        events.append(("open_mark", open_mark))
    fill_payload = _execute_pending_if_due(
        state,
        timestamp,
        open_price,
        eligible_index=int(state["eligible_index"]) + 1,
    )
    if fill_payload is not None:
        events.append(("fill", fill_payload))
    _mark_buy_hold_open(state, timestamp, open_price)
    _record_candidate_open_mark(state, timestamp)
    events.append(
        (
            "deadline_open",
            {
                "open_utc": _ts(timestamp),
                "open": open_price,
                "candidate_equity": float(state["equity"]),
                "benchmark_equity": float(state["benchmark"]["equity"]),
            },
        )
    )
    if int(state["closed_trades_count"]) == CLOSED_TRADES_TARGET:
        verdict = _powered_verdict(state)
        reason = "closed_trades_target_at_deadline"
    else:
        verdict = "UNDERPOWERED"
        reason = "underpowered_deadline"
    _set_terminal(state, verdict, timestamp, reason)
    events.append(("terminal", _terminal_payload(state, timestamp, reason)))
    return events


def _process_current_open(
    state: dict[str, Any],
    current_open: tuple[pd.Timestamp, float],
) -> list[tuple[str, dict[str, Any]]]:
    timestamp, open_price = current_open
    stamp = _ts(timestamp)
    pending = state.get("pending_order")
    if pending is not None and pending["execute_open_utc"] != stamp:
        raise ForwardTestError("current open does not match pending order execution timestamp")
    should_mark = (
        pending is not None
        or int(state["position"]) != 0
        or (stamp == _ts(FIRST_ELIGIBLE) and state["benchmark"]["entry_price"] is None)
        or stamp == _ts(DEADLINE)
    )
    if not should_mark:
        return []
    events: list[tuple[str, dict[str, Any]]] = []
    already_marked = state.get("last_open_mark_utc") == stamp
    open_mark = _apply_open_gap(state, timestamp, open_price)
    if open_mark is not None:
        events.append(("open_mark", open_mark))
    if already_marked and pending is None and stamp != _ts(DEADLINE):
        return []
    benchmark_marked = False
    if state["benchmark"]["entry_price"] is None or int(state["position"]) != 0 or pending is not None or stamp == _ts(DEADLINE):
        _mark_buy_hold_open(state, timestamp, open_price)
        benchmark_marked = True
    fill_payload = None
    if pending is not None:
        fill_payload = _execute_pending_if_due(
            state,
            timestamp,
            open_price,
            eligible_index=int(state["eligible_index"]) + 1,
        )
        if fill_payload is not None:
            events.append(("fill", fill_payload))
    if stamp == _ts(DEADLINE):
        _record_candidate_open_mark(state, timestamp)
        events.append(
            (
                "deadline_open",
                {
                    "open_utc": stamp,
                    "open": open_price,
                    "candidate_equity": float(state["equity"]),
                    "benchmark_equity": float(state["benchmark"]["equity"]),
                },
            )
        )
    elif open_mark is not None or fill_payload is not None:
        _record_candidate_open_mark(state, timestamp)
    elif benchmark_marked:
        events.append(
            (
                "benchmark_open",
                {
                    "open_utc": stamp,
                    "open": open_price,
                    "benchmark_equity": state["benchmark"]["equity"],
                },
            )
        )
    if int(state["closed_trades_count"]) == CLOSED_TRADES_TARGET:
        verdict = _powered_verdict(state)
        _set_terminal(state, verdict, timestamp, "closed_trades_target")
        events.append(("terminal", _terminal_payload(state, timestamp, "closed_trades_target")))
    elif stamp == _ts(DEADLINE):
        _set_terminal(state, "UNDERPOWERED", timestamp, "underpowered_deadline")
        events.append(("terminal", _terminal_payload(state, timestamp, "underpowered_deadline")))
    return events


def _execute_pending_if_due(
    state: dict[str, Any],
    timestamp: pd.Timestamp,
    open_price: float,
    *,
    eligible_index: int,
) -> dict[str, Any] | None:
    pending = state.get("pending_order")
    if pending is None:
        return None
    stamp = _ts(timestamp)
    if pending["execute_open_utc"] != stamp:
        raise ForwardTestError("pending order execution timestamp does not match next candle")

    previous_position = int(state["position"])
    target = int(pending["target_position"])
    pre_open_equity = float(state["equity"])

    transition_equity = pre_open_equity
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    reason = str(pending["reason"])
    if target != previous_position:
        if previous_position != 0:
            open_trade = state.get("open_trade")
            if not open_trade:
                raise ForwardTestError("cannot exit without an open trade")
            exit_cost = pre_open_equity * COST_RATE
            after_exit = pre_open_equity - exit_cost
            net_return = after_exit / float(open_trade["entry_equity"]) - 1.0
            trade = {
                "entry_time": open_trade["entry_time"],
                "exit_time": stamp,
                "side": SIDE if previous_position == TARGET_POSITION else "long",
                "entry_price": float(open_trade["entry_price"]),
                "exit_price": open_price,
                "bars": int(max(eligible_index - int(open_trade["entry_index"]), 1)),
                "gross_return": float((open_price / float(open_trade["entry_price"]) - 1.0) * previous_position),
                "net_return": float(net_return),
                "reason": reason,
            }
            trades.append(trade)
            state["closed_trades"].append(trade)
            state["closed_trades_count"] = int(state["closed_trades_count"]) + 1
            orders.append(
                {
                    "time": stamp,
                    "from_position": previous_position,
                    "to_position": 0,
                    "price": open_price,
                    "cost": float(exit_cost),
                    "reason": reason,
                }
            )
            transition_equity = after_exit
            state["open_trade"] = None
        if target != 0:
            entry_base_equity = transition_equity
            entry_cost = pre_open_equity * COST_RATE
            transition_equity -= entry_cost
            orders.append(
                {
                    "time": stamp,
                    "from_position": 0,
                    "to_position": target,
                    "price": open_price,
                    "cost": float(entry_cost),
                    "reason": reason,
                }
            )
            state["open_trade"] = {
                "entry_time": stamp,
                "entry_price": open_price,
                "entry_equity": entry_base_equity,
                "entry_index": eligible_index,
                "side": SIDE if target == TARGET_POSITION else "long",
            }
    state["equity"] = transition_equity
    state["position"] = target
    state["pending_order"] = None
    _update_trade_metrics(state)
    return {
        "open_utc": stamp,
        "from_position": previous_position,
        "to_position": target,
        "price": open_price,
        "reason": reason,
        "orders": orders,
        "trades": trades,
        "equity_after_transition": float(state["equity"]),
        "closed_trades_count": int(state["closed_trades_count"]),
    }


def _apply_open_gap(state: dict[str, Any], timestamp: pd.Timestamp, open_price: float) -> dict[str, Any] | None:
    stamp = _ts(timestamp)
    marks = state.setdefault("open_marks", {})
    if stamp in marks:
        recorded_open = float(marks[stamp]["open"])
        if not math.isclose(recorded_open, open_price, rel_tol=0.0, abs_tol=1e-12):
            raise ForwardTestError("previously booked candle open was mutated")
    if state.get("last_open_mark_utc") == stamp:
        return None
    previous_position = int(state["position"])
    before = float(state["equity"])
    after = before
    if previous_position != 0:
        last_close = state.get("last_close")
        if last_close is None:
            raise ForwardTestError("cannot mark open PnL without prior close")
        after = before * (1.0 + previous_position * (open_price / float(last_close) - 1.0))
        state["equity"] = after
        state["performance"]["net_return"] = after / INITIAL_EQUITY - 1.0
    state["last_open_mark_utc"] = stamp
    marks[stamp] = {"open": open_price}
    if previous_position == 0:
        return None
    return {
        "open_utc": stamp,
        "position": previous_position,
        "last_close": float(state["last_close"]),
        "open": open_price,
        "equity_before": before,
        "equity_after": after,
    }


def _schedule_next_order(state: dict[str, Any], timestamp: pd.Timestamp, signal: int, exit_signal: bool) -> dict[str, Any]:
    position = int(state["position"])
    next_target = position
    reason = "signal"
    if position != 0 and exit_signal:
        next_target = 0
        reason = "opposite_channel"
    open_trade = state.get("open_trade")
    if position != 0 and open_trade is not None:
        held = int(state["eligible_index"]) - int(open_trade["entry_index"]) + 1
        if held >= HOLD_BARS:
            next_target = 0
            reason = "time"
    if int(signal) != 0 and position == 0:
        next_target = int(signal)
        reason = "signal"

    if next_target != position:
        execute_at = timestamp + TIMEFRAME_DELTA
        state["pending_order"] = {
            "created_after_close_utc": _ts(timestamp),
            "execute_open_utc": _ts(execute_at),
            "target_position": int(next_target),
            "reason": reason,
        }
        return dict(state["pending_order"])
    state["pending_order"] = None
    return {"target_position": position, "reason": "hold"}


def _decision_diagnostics(
    state: dict[str, Any],
    signal: int,
    exit_signal: bool,
    signal_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    position = int(state["position"])
    flat = position == 0
    open_trade = state.get("open_trade")
    held_bars: int | None = None
    hold_expiry = False
    if position != 0 and open_trade is not None:
        held_bars = int(state["eligible_index"]) - int(open_trade["entry_index"]) + 1
        hold_expiry = held_bars >= HOLD_BARS
    opposite_channel = bool(exit_signal)
    breakout_short = bool(signal_diagnostics.get("breakout_short", False))
    volatility_compression = bool(signal_diagnostics.get("volatility_compression", False))
    entry_rule_pass = int(signal) == TARGET_POSITION

    if hold_expiry:
        reason = "hold_expiry"
    elif position != 0 and opposite_channel:
        reason = "opposite_channel"
    elif flat and entry_rule_pass:
        reason = "breakout_short"
    elif flat and breakout_short and not volatility_compression:
        reason = "volatility_filter_fail"
    elif flat:
        reason = "flat"
    else:
        reason = "hold"

    return {
        "reason": reason,
        "diagnostics": {
            "breakout_short": breakout_short,
            "volatility_compression": volatility_compression,
            "volatility_filter": "pass" if volatility_compression else "fail",
            "entry_rule_pass": entry_rule_pass,
            "hold_expiry": hold_expiry,
            "held_bars": held_bars,
            "opposite_channel": opposite_channel,
            "flat": flat,
        },
    }


def _mark_buy_hold(state: dict[str, Any], timestamp: pd.Timestamp, open_price: float, close_price: float) -> None:
    benchmark = state["benchmark"]
    stamp = _ts(timestamp)
    if benchmark["entry_price"] is None:
        benchmark["entry_price"] = open_price
        benchmark["equity"] = (INITIAL_EQUITY - INITIAL_EQUITY * COST_RATE) * (close_price / open_price)
    else:
        last_point = benchmark["equity_curve"][-1] if benchmark["equity_curve"] else None
        if isinstance(last_point, dict) and last_point.get("open_utc") == stamp and last_point.get("mark") == "open":
            benchmark["equity"] = float(benchmark["equity"]) * (close_price / open_price)
        else:
            last_close = state.get("last_close")
            if last_close is None:
                raise ForwardTestError("cannot update benchmark without prior close")
            benchmark["equity"] = float(benchmark["equity"]) * (close_price / float(last_close))
    benchmark["net_return"] = float(benchmark["equity"]) / INITIAL_EQUITY - 1.0
    benchmark["equity_curve"].append({"open_utc": _ts(timestamp), "equity": float(benchmark["equity"])})


def _mark_buy_hold_open(state: dict[str, Any], timestamp: pd.Timestamp, open_price: float) -> None:
    benchmark = state["benchmark"]
    if benchmark["entry_price"] is None:
        benchmark["entry_price"] = open_price
        benchmark["equity"] = INITIAL_EQUITY - INITIAL_EQUITY * COST_RATE
    else:
        last_close = state.get("last_close")
        if last_close is None:
            raise ForwardTestError("cannot update benchmark open mark without prior close")
        benchmark["equity"] = float(benchmark["equity"]) * (open_price / float(last_close))
    benchmark["net_return"] = float(benchmark["equity"]) / INITIAL_EQUITY - 1.0
    benchmark["equity_curve"].append({"open_utc": _ts(timestamp), "equity": float(benchmark["equity"]), "mark": "open"})


def _record_candidate_open_mark(state: dict[str, Any], timestamp: pd.Timestamp) -> None:
    perf = state["performance"]
    perf["net_return"] = float(state["equity"]) / INITIAL_EQUITY - 1.0
    perf["equity_curve"].append({"open_utc": _ts(timestamp), "equity": float(state["equity"]), "mark": "open"})
    _update_trade_metrics(state)


def _update_equity_metrics(state: dict[str, Any], timestamp: pd.Timestamp) -> None:
    perf = state["performance"]
    perf["net_return"] = float(state["equity"]) / INITIAL_EQUITY - 1.0
    perf["equity_curve"].append({"open_utc": _ts(timestamp), "equity": float(state["equity"])})
    _update_trade_metrics(state)


def _update_trade_metrics(state: dict[str, Any]) -> None:
    perf = state["performance"]
    returns = [float(trade["net_return"]) for trade in state["closed_trades"][:CLOSED_TRADES_TARGET]]
    if len(returns) >= 2:
        std = float(np.std(np.asarray(returns, dtype=float), ddof=1))
        mean = float(np.mean(np.asarray(returns, dtype=float)))
        perf["per_trade_sharpe"] = mean / std if std > 0 and math.isfinite(std) else None
    else:
        perf["per_trade_sharpe"] = None
    perf["net_return"] = float(state["equity"]) / INITIAL_EQUITY - 1.0


def _update_drawdown(state: dict[str, Any], equity: float) -> float:
    hwm = max(float(state["high_watermark"]), equity)
    state["high_watermark"] = hwm
    dd = 1.0 - equity / hwm
    state["max_drawdown"] = max(float(state["max_drawdown"]), dd)
    return dd


def _drawdown_exceeds_threshold(drawdown: float) -> bool:
    return bool(drawdown > DRAWDOWN_FAIL_THRESHOLD)


def _powered_verdict(state: dict[str, Any]) -> str:
    net_return = float(state["performance"]["net_return"])
    sharpe = state["performance"]["per_trade_sharpe"]
    if net_return > 0.0 and sharpe is not None and float(sharpe) > 0.0 and math.isfinite(float(sharpe)):
        return "PASS"
    return "FAIL"


def _set_terminal(state: dict[str, Any], status: str, timestamp: pd.Timestamp, reason: str) -> None:
    state["status"] = status
    state["terminal"] = {
        "status": status,
        "reason": reason,
        "open_utc": _ts(timestamp),
        "equity": float(state["equity"]),
        "position": int(state["position"]),
        "open_trade": state.get("open_trade"),
        "closed_trades_count": int(state["closed_trades_count"]),
        "net_return": float(state["performance"]["net_return"]),
        "per_trade_sharpe": state["performance"]["per_trade_sharpe"],
        "max_drawdown": float(state["max_drawdown"]),
    }


def _terminal_payload(state: dict[str, Any], timestamp: pd.Timestamp, reason: str) -> dict[str, Any]:
    return {
        "open_utc": _ts(timestamp),
        "status": state["status"],
        "reason": reason,
        "terminal": state["terminal"],
    }


def _load_and_assert_prereg(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ForwardTestError(f"missing preregistration file: {path}") from exc
    if sha256_bytes(data) != PREREG_FILE_SHA256:
        raise ForwardTestError("preregistration file hash drift")
    prereg = json.loads(data.decode("utf-8"))
    params = prereg["frozen_candidate"]["strategy_parameters"]
    if prereg["frozen_candidate"]["parameter_sha256"] != PARAMETER_SHA256:
        raise ForwardTestError("embedded parameter hash drift")
    if sha256_bytes(canonical_json_bytes(params)) != PARAMETER_SHA256:
        raise ForwardTestError("strategy parameter hash drift")
    _assert_runtime_contract(prereg)
    return prereg


def _assert_runtime_contract(prereg: dict[str, Any]) -> None:
    study = prereg["study"]
    candidate = prereg["frozen_candidate"]
    params = candidate["strategy_parameters"]
    entry = params["entry"]
    breakout = entry["breakout_short"]
    compression = entry["volatility_compression"]
    exit_params = params["exit"]
    evaluation = prereg["evaluation"]
    benchmark = prereg["background_benchmark"]
    checks = (
        ("study id", STUDY_ID, study["id"]),
        ("candidate id", CANDIDATE_ID, params["candidate_id"]),
        ("hypothesis", HYPOTHESIS_ID, params["hypothesis_id"]),
        ("symbol", SYMBOL, params["symbol"]),
        ("timeframe", TIMEFRAME, params["timeframe"]),
        ("timeframe hours", TIMEFRAME_HOURS, params["timeframe_hours"]),
        ("timeframe delta", TIMEFRAME_DELTA, pd.Timedelta(hours=int(params["timeframe_hours"]))),
        ("side", SIDE, params["side"]),
        ("target position", TARGET_POSITION, params["target_position"]),
        ("allow reversal", ALLOW_REVERSAL, params["allow_reversal"]),
        ("Donchian entry window", DONCHIAN_WINDOW_BARS, breakout["donchian_window_bars"]),
        (
            "Donchian exit window",
            DONCHIAN_WINDOW_BARS,
            exit_params["opposite_channel"]["donchian_window_bars"],
        ),
        ("realized-vol window", REALIZED_VOL_WINDOW_BARS, compression["realized_vol_window_bars"]),
        ("realized-vol ddof", REALIZED_VOL_STD_DDOF, compression["realized_vol_std_ddof"]),
        ("history window", HISTORY_WINDOW_BARS, compression["history_window_bars"]),
        ("history quantile numerator", HISTORY_QUANTILE_NUMERATOR, compression["history_quantile_numerator"]),
        (
            "history quantile denominator",
            HISTORY_QUANTILE_DENOMINATOR,
            compression["history_quantile_denominator"],
        ),
        ("maximum hold", HOLD_BARS, exit_params["maximum_hold_bars"]),
        ("cost rate", COST_RATE, params["costs"]["total_rate_per_side"]),
        ("warmup bars", WARMUP_BARS, params["warmup_bars_before_first_eligible_candle"]),
        ("initial equity", INITIAL_EQUITY, evaluation["initial_equity"]),
        ("closed-trade target", CLOSED_TRADES_TARGET, evaluation["closed_trades_target"]),
        ("drawdown threshold", DRAWDOWN_FAIL_THRESHOLD, evaluation["early_kill_rule"]["threshold"]),
        ("first eligible", _ts(FIRST_ELIGIBLE), study["first_eligible_candle_open_utc"]),
        ("benchmark start", _ts(FIRST_ELIGIBLE), benchmark["entry_timestamp"]),
        ("benchmark side", 1, benchmark["position"]),
        ("benchmark cost", COST_RATE, benchmark["entry_cost_rate"]),
        ("deadline", _ts(DEADLINE), study["underpowered_deadline_utc"]),
        ("evaluation deadline", _ts(DEADLINE), evaluation["underpowered_rule"]["deadline_utc"]),
    )
    for label, actual, frozen in checks:
        if actual != frozen:
            raise ForwardTestError(f"runtime contract drift: {label}")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in ("open", "high", "low", "close", "volume") if column not in frame.columns]
    if missing:
        raise ForwardTestError(f"OHLCV frame is missing required columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ForwardTestError("OHLCV frame must use a UTC DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) != "UTC":
        raise ForwardTestError("OHLCV frame must use a timezone-aware UTC DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ForwardTestError("OHLCV frame index must be monotonic increasing")
    if frame.index.has_duplicates:
        raise ForwardTestError("OHLCV frame index must not contain duplicates")
    if frame.empty:
        raise ForwardTestError("OHLCV frame is empty")
    if not frame.index.is_unique:
        raise ForwardTestError("OHLCV frame contains duplicate timestamps")
    if any(
        timestamp.hour % TIMEFRAME_HOURS != 0
        or timestamp.minute != 0
        or timestamp.second != 0
        or timestamp.microsecond != 0
        or timestamp.nanosecond != 0
        for timestamp in frame.index
    ):
        raise ForwardTestError("OHLCV frame timestamps must align to UTC 4h boundaries")
    deltas = frame.index.to_series().diff().dropna()
    if not deltas.empty and not (deltas == TIMEFRAME_DELTA).all():
        raise ForwardTestError("OHLCV frame contains gaps or non-4h rows")
    for column in ("open", "high", "low", "close", "volume"):
        values = frame[column].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ForwardTestError(f"OHLCV column {column} contains non-finite values")
    if (frame[["open", "high", "low", "close"]].astype(float) <= 0).any().any():
        raise ForwardTestError("OHLCV prices must be positive")
    if (frame["volume"].astype(float) < 0).any():
        raise ForwardTestError("OHLCV volume must be non-negative")
    prices = frame[["open", "high", "low", "close"]].astype(float)
    if (
        (prices["high"] < prices[["open", "close"]].max(axis=1)).any()
        or (prices["low"] > prices[["open", "close"]].min(axis=1)).any()
        or (prices["high"] < prices["low"]).any()
    ):
        raise ForwardTestError("OHLCV high/low values are inconsistent with open/close")
    return frame.astype({column: float for column in ("open", "high", "low", "close", "volume")})


def _validate_run_now(now: pd.Timestamp | None) -> pd.Timestamp | None:
    if now is None:
        return None
    try:
        timestamp = pd.Timestamp(now)
    except (TypeError, ValueError) as exc:
        raise ForwardTestError("now must be a valid timezone-aware timestamp") from exc
    if timestamp.tzinfo is None:
        raise ForwardTestError("now must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _validate_current_open(
    current_open: dict[str, Any] | tuple[pd.Timestamp, float] | None,
    *,
    run_now: pd.Timestamp | None,
) -> tuple[pd.Timestamp, float] | None:
    if current_open is None:
        return None
    try:
        if isinstance(current_open, dict):
            raw_timestamp = current_open["open_utc"]
            raw_open = current_open["open"]
        elif isinstance(current_open, tuple) and len(current_open) == 2:
            raw_timestamp, raw_open = current_open
        else:
            raise TypeError
        timestamp = pd.Timestamp(raw_timestamp)
        open_price = float(raw_open)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ForwardTestError("current open quote is malformed") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise ForwardTestError("current open timestamp must be UTC")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.value % TIMEFRAME_DELTA.value != 0:
        raise ForwardTestError("current open timestamp must align to a UTC 4h boundary")
    if not math.isfinite(open_price) or open_price <= 0.0:
        raise ForwardTestError("current open price must be finite and positive")
    if run_now is not None and timestamp != _floor_to_timeframe(run_now):
        raise ForwardTestError("current open timestamp must equal floor(now) on the 4h grid")
    return timestamp, open_price


def _assert_completed_rows_closed(frame: pd.DataFrame, run_now: pd.Timestamp | None) -> None:
    if run_now is None:
        return
    current_boundary = _floor_to_timeframe(run_now)
    if bool((frame.index >= current_boundary).any()):
        raise ForwardTestError("completed OHLCV frame contains a candle that is not closed at now")
    expected_last = current_boundary - TIMEFRAME_DELTA
    if frame.index[-1] != expected_last:
        raise ForwardTestError(
            f"completed OHLCV frame is stale; expected latest closed candle {_ts(expected_last)}"
        )


def _floor_to_timeframe(timestamp: pd.Timestamp) -> pd.Timestamp:
    epoch_ns = pd.Timestamp("1970-01-01T00:00:00Z").value
    open_ns = epoch_ns + ((timestamp.value - epoch_ns) // TIMEFRAME_DELTA.value) * TIMEFRAME_DELTA.value
    return pd.Timestamp(open_ns, tz="UTC")


def _assert_initial_warmup(
    state: dict[str, Any],
    frame: pd.DataFrame,
    current_open: tuple[pd.Timestamp, float] | None,
) -> None:
    if state.get("status") != "RUNNING":
        return
    starts_from_completed = bool((frame.index >= FIRST_ELIGIBLE).any())
    starts_from_current = current_open is not None and current_open[0] >= FIRST_ELIGIBLE
    if not starts_from_completed and not starts_from_current:
        return
    required = pd.date_range(
        end=FIRST_ELIGIBLE - TIMEFRAME_DELTA,
        periods=WARMUP_BARS,
        freq=TIMEFRAME_DELTA,
        tz="UTC",
    )
    if not required.isin(frame.index).all():
        raise ForwardTestError(
            f"each active forward-test run requires at least {WARMUP_BARS} consecutive 4h warmup bars "
            "immediately before FIRST_ELIGIBLE"
        )


def _assert_current_open_is_next(state: dict[str, Any], timestamp: pd.Timestamp) -> None:
    last_processed = state.get("last_processed_open_utc")
    expected = FIRST_ELIGIBLE if last_processed is None else pd.Timestamp(last_processed) + TIMEFRAME_DELTA
    if timestamp != expected:
        raise ForwardTestError(
            f"current open must be exactly the next 4h open ({_ts(expected)}); got {_ts(timestamp)}"
        )


def _is_pristine_prestart_current(state: dict[str, Any], timestamp: pd.Timestamp) -> bool:
    return (
        timestamp < FIRST_ELIGIBLE
        and state.get("last_processed_open_utc") is None
        and not state.get("observations")
        and int(state.get("position", 0)) == 0
        and state.get("pending_order") is None
        and state.get("open_trade") is None
        and state.get("benchmark", {}).get("entry_price") is None
    )


def _h2_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    diagnostics = _h2_diagnostics(frame)
    signal = pd.Series(0, index=frame.index, dtype=int, name="signal")
    signal.loc[diagnostics["entry_rule_pass"]] = TARGET_POSITION
    exit_signal = diagnostics["opposite_channel"].astype(bool).rename("exit_signal")
    return signal, exit_signal


def _h2_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    log_return = np.log(close / close.shift(1))
    donchian_high20_prior = high.shift(1).rolling(
        DONCHIAN_WINDOW_BARS,
        min_periods=DONCHIAN_WINDOW_BARS,
    ).max()
    donchian_low20_prior = low.shift(1).rolling(
        DONCHIAN_WINDOW_BARS,
        min_periods=DONCHIAN_WINDOW_BARS,
    ).min()
    rv20 = log_return.rolling(
        REALIZED_VOL_WINDOW_BARS,
        min_periods=REALIZED_VOL_WINDOW_BARS,
    ).std(ddof=REALIZED_VOL_STD_DDOF)
    lower_tercile = rv20.shift(1).rolling(
        HISTORY_WINDOW_BARS,
        min_periods=HISTORY_WINDOW_BARS,
    ).quantile(HISTORY_QUANTILE_NUMERATOR / HISTORY_QUANTILE_DENOMINATOR)
    breakout_short = close < donchian_low20_prior
    compressed = rv20 < lower_tercile
    return pd.DataFrame(
        {
            "breakout_short": breakout_short.fillna(False).astype(bool),
            "volatility_compression": compressed.fillna(False).astype(bool),
            "entry_rule_pass": (breakout_short & compressed).fillna(False).astype(bool),
            "opposite_channel": (close > donchian_high20_prior).fillna(False).astype(bool),
        },
        index=frame.index,
    )


def _load_state(path: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return initial_state(prereg)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntegrityError("state JSON is invalid") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise IntegrityError("state schema is invalid")
    return state


def _verify_artifacts(paths: dict[str, Path], state: dict[str, Any]) -> tuple[str, int, list[dict[str, Any]]]:
    try:
        ledger_head, ledger_count, entries = verify_ledger(paths["ledger"], paths["head"])
    except IntegrityError as exc:
        raise ForwardTestError(str(exc)) from exc
    if state.get("ledger_head") != ledger_head or int(state.get("ledger_entries", -1)) != ledger_count:
        raise ForwardTestError("state does not match ledger/head artifacts")
    if not isinstance(state.get("hashes"), dict) or state["hashes"].get("ledger_head") != ledger_head:
        raise ForwardTestError("state hashes ledger head drift")
    projection_hash = state_projection_hash(state)
    if state.get("state_checkpoint_sha256") != projection_hash:
        raise ForwardTestError("state checkpoint hash drift")
    if ledger_count == 0:
        if projection_hash != GENESIS_STATE_CHECKPOINT_SHA256:
            raise ForwardTestError("genesis state checkpoint drift")
    elif entries[-1].get("event") != "state_checkpoint":
        raise ForwardTestError("latest ledger event must be state_checkpoint")
    elif entries[-1].get("payload", {}).get("state_sha256") != projection_hash:
        raise ForwardTestError("state does not match latest ledger checkpoint")
    return ledger_head, ledger_count, entries


def _assert_state_matches_prereg(state: dict[str, Any]) -> None:
    expected = {
        "prereg_commit": PREREG_COMMIT,
        "prereg_sha256": PREREG_FILE_SHA256,
        "parameter_sha256": PARAMETER_SHA256,
        "candidate_id": CANDIDATE_ID,
        "first_eligible_open_utc": _ts(FIRST_ELIGIBLE),
        "underpowered_deadline_utc": _ts(DEADLINE),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ForwardTestError(f"state {key} drift")


def _assert_observation_history_immutable(state: dict[str, Any], frame: pd.DataFrame) -> None:
    observations = state.get("observations", {})
    if not isinstance(observations, dict):
        raise ForwardTestError("state observations schema is invalid")
    if not observations:
        return
    frame_stamps = {_ts(ts): ts for ts in frame.index}
    for stamp, expected_hash in observations.items():
        timestamp = frame_stamps.get(stamp)
        if timestamp is None:
            raise ForwardTestError("previously processed OHLCV row is missing")
        current_hash = _row_hash(timestamp, frame.loc[timestamp])
        if current_hash != expected_hash:
            raise ForwardTestError("previously processed OHLCV row was mutated")


def _has_unprocessed_gap(state: dict[str, Any], timestamp: pd.Timestamp) -> bool:
    last = state.get("last_processed_open_utc")
    if last is None:
        return timestamp != FIRST_ELIGIBLE
    return pd.Timestamp(last) + TIMEFRAME_DELTA != timestamp


def _row_hash(timestamp: pd.Timestamp, row: pd.Series) -> str:
    payload = {
        "open_utc": _ts(timestamp),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
    return sha256_json(payload)


def _add_entry(
    entries: list[dict[str, Any]],
    sequence: int,
    head: str,
    event: str,
    payload: dict[str, Any],
) -> tuple[str, int]:
    entry = build_ledger_entry(sequence, head, event, payload)
    entries.append(entry)
    return str(entry["entry_hash"]), sequence + 1


def state_projection_hash(state: dict[str, Any]) -> str:
    projection = json.loads(json.dumps(state))
    projection.pop("ledger_head", None)
    projection.pop("ledger_entries", None)
    projection.pop("state_checkpoint_sha256", None)
    if isinstance(projection.get("hashes"), dict):
        projection["hashes"].pop("ledger_head", None)
    return sha256_json(projection)


def _paths(root: Path) -> dict[str, Path]:
    _assert_root_path_allowed(root)
    return {
        "root": root,
        "prereg": root / "prereg_forward.json",
        "state": root / "state.json",
        "ledger": root / "ledger.jsonl",
        "head": root / "head.sha256",
    }


def _assert_runtime_artifacts_exist(paths: dict[str, Path]) -> None:
    missing = [name for name in ("state", "ledger", "head") if not paths[name].is_file()]
    if missing:
        raise ForwardTestError(
            "forward-test runtime artifacts are missing; refusing implicit reset: " + ", ".join(missing)
        )


def _assert_root_path_allowed(root: Path) -> None:
    candidates = (root, root.resolve(strict=False))
    if any(part.casefold() == "holdout" for candidate in candidates for part in candidate.parts):
        raise ForwardTestError("forward-test root path must not contain a holdout component")


def _ts(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
