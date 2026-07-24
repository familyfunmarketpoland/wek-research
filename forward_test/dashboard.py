from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PREREG_PATH = ROOT / "prereg_forward.json"
STATE_PATH = ROOT / "state.json"
LEDGER_PATH = ROOT / "ledger.jsonl"
HEAD_PATH = ROOT / "head.sha256"
OUTPUT_PATH = ROOT / "dashboard" / "index.html"

WIDTH = 960
HEIGHT = 280
PADDING_X = 44
PADDING_Y = 24


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def _json_hash(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(serialized)


def _pick(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not mapping:
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct(value: Any) -> str:
    number = _to_float(value, 0.0) * 100.0
    return f"{number:+.2f}%"


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    return f"{_to_float(value):.{digits}f}"


def _fmt_ts(value: Any) -> str:
    return str(value) if value not in (None, "") else "NA"


def _normalize_point(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    timestamp = _pick(raw, "timestamp", "ts", "time", "open_utc", "open_time_utc", "close_time_utc")
    equity = _pick(raw, "equity", "value", "close_marked_equity", "paper_equity", "buy_hold_equity")
    if timestamp in (None, "") or equity in (None, ""):
        return None
    return {"timestamp": str(timestamp), "equity": _to_float(equity, 0.0)}


def _curve_points(state: dict[str, Any] | None, *keys: str) -> list[dict[str, Any]]:
    series = _pick(state, *keys, default=[])
    if not isinstance(series, list):
        return []
    points = []
    for raw in series:
        point = _normalize_point(raw)
        if point is not None:
            points.append(point)
    return points


def _normalize_trade(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "entry_time": _fmt_ts(_pick(raw, "entry_time", "entry_timestamp", "entry_open_utc")),
        "exit_time": _fmt_ts(_pick(raw, "exit_time", "exit_timestamp", "exit_open_utc")),
        "side": str(_pick(raw, "side", default="NA")),
        "bars": str(_pick(raw, "bars", "bars_held", "hold_bars", default="NA")),
        "return": _pct(_pick(raw, "net_return", "trade_return", "return_pct", default=0.0)),
        "reason": str(_pick(raw, "exit_reason", "reason", default="NA")),
    }


def _trade_rows(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    trades = _pick(state, "closed_trades", "trades", default=[])
    if not isinstance(trades, list):
        return []
    rows = []
    for raw in trades:
        trade = _normalize_trade(raw)
        if trade is not None:
            rows.append(trade)
    return rows


def _svg_polyline(
    points: list[dict[str, Any]],
    color: str,
    *,
    min_y: float,
    max_y: float,
    x_positions: dict[str, float],
) -> str:
    if not points:
        return ""
    span_y = max(HEIGHT - 2 * PADDING_Y, 1)
    coords = []
    for point in points:
        x = x_positions[point["timestamp"]]
        y = PADDING_Y + span_y * (1 - ((point["equity"] - min_y) / (max_y - min_y)))
        coords.append(f"{x:.2f},{y:.2f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(coords)}" />'


def _svg_chart(strategy_points: list[dict[str, Any]], buy_hold_points: list[dict[str, Any]]) -> str:
    merged = strategy_points + buy_hold_points
    if not merged:
        return '<div class="empty-chart">Brak punktow equity. Dashboard czeka na pierwszy zapis runnera.</div>'
    values = [point["equity"] for point in merged]
    min_y = min(values)
    max_y = max(values)
    if min_y == max_y:
        min_y -= 0.01
        max_y += 0.01
    timestamps = sorted({point["timestamp"] for point in merged})
    span_x = max(WIDTH - 2 * PADDING_X, 1)
    x_total = max(len(timestamps) - 1, 1)
    x_positions = {
        timestamp: PADDING_X + span_x * (index / x_total)
        for index, timestamp in enumerate(timestamps)
    }
    labels = [
        (PADDING_Y, max_y),
        ((HEIGHT - PADDING_Y) / 2, (min_y + max_y) / 2),
        (HEIGHT - PADDING_Y, min_y),
    ]
    axis = []
    for y, value in labels:
        axis.append(
            f'<text x="8" y="{y:.0f}" class="axis-label">{html.escape(_fmt_num(value, 4))}</text>'
        )
    last_ts = timestamps[-1]
    return f"""
<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Forward-test equity chart">
  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" rx="16" ry="16" class="chart-bg" />
  <line x1="{PADDING_X}" y1="{PADDING_Y}" x2="{PADDING_X}" y2="{HEIGHT - PADDING_Y}" class="axis-line" />
  <line x1="{PADDING_X}" y1="{HEIGHT - PADDING_Y}" x2="{WIDTH - PADDING_X}" y2="{HEIGHT - PADDING_Y}" class="axis-line" />
  {''.join(axis)}
  {_svg_polyline(buy_hold_points, "#6aa7ff", min_y=min_y, max_y=max_y, x_positions=x_positions)}
  {_svg_polyline(strategy_points, "#0f766e", min_y=min_y, max_y=max_y, x_positions=x_positions)}
  <text x="{PADDING_X}" y="{HEIGHT - 6}" class="axis-label">Start</text>
  <text x="{WIDTH - PADDING_X - 56}" y="{HEIGHT - 6}" class="axis-label">{html.escape(last_ts)}</text>
</svg>
""".strip()


def _status_class(status: str) -> str:
    mapping = {
        "PRE_REGISTERED_NOT_STARTED": "status-pending",
        "RUNNING": "status-running",
        "PASS": "status-pass",
        "FAIL": "status-fail",
        "UNDERPOWERED": "status-underpowered",
    }
    return mapping.get(status, "status-pending")


def _audit_rows(prereg: dict[str, Any], state: dict[str, Any] | None) -> list[tuple[str, str]]:
    study = prereg["study"]
    frozen = prereg["frozen_candidate"]
    evaluation = prereg["evaluation"]
    operations = prereg["operations"]
    audit = _pick(state, "audit", default={})
    hashes = _pick(state, "hashes", default={})
    status = str(_pick(state, "status", default=_pick(study, "status", default="NA")))
    rendered_at = (
        str(_pick(state, "updated_at_utc", "last_updated_utc", default="NA"))
        if status in {"PASS", "FAIL", "UNDERPOWERED"}
        else _utc_now()
    )
    return [
        ("Study ID", str(_pick(study, "id", default="NA"))),
        ("Terminal status", str(_pick(state, "status", default=_pick(study, "status", default="NA")))),
        ("Eligible from", str(_pick(state, "first_eligible_open_utc", default=_pick(study, "first_eligible_candle_open_utc", default="NA")))),
        ("Underpowered deadline", str(_pick(state, "underpowered_deadline_utc", default=_pick(study, "underpowered_deadline_utc", default="NA")))),
        ("Target closed trades", str(_pick(evaluation, "closed_trades_target", default=30))),
        ("Candidate ID", str(_pick(frozen, "candidate_id", default="NA"))),
        ("Parameter SHA256", str(_pick(hashes, "parameter_sha256", default=_pick(frozen, "parameter_sha256", default="NA")))),
        ("Config fingerprint", str(_pick(study, "source_config_fingerprint", default="NA"))),
        ("Prereg SHA256", str(_pick(hashes, "prereg_sha256", default=_sha256_file(PREREG_PATH) or "NA"))),
        ("State SHA256", str(_sha256_file(STATE_PATH) or "NA")),
        ("Canonical state SHA256", str(_json_hash(state) or "NA")),
        ("State checkpoint SHA256", str(_pick(state, "state_checkpoint_sha256", default="NA"))),
        ("Ledger SHA256", str(_sha256_file(LEDGER_PATH) or "NA")),
        ("Head SHA256", str(_pick(hashes, "ledger_head", default=_read_text(HEAD_PATH) or "NA"))),
        ("Latest ledger event SHA256", str(_pick(audit, "latest_ledger_event_sha256", "latest_event_sha256", default="NA"))),
        ("Last eligible candle", str(_pick(state, "last_processed_open_utc", default=_pick(audit, "last_processed_candle_open_utc", "last_eligible_candle_open_utc", default="NA")))),
        ("Runner updated at", str(_pick(state, "updated_at_utc", "last_updated_utc", default="NA"))),
        ("Dashboard rendered at", rendered_at),
        ("Workflow schedule", str(_pick(operations, "github_actions_cron", default="NA"))),
    ]


def render_dashboard(prereg: dict[str, Any], state: dict[str, Any] | None = None) -> str:
    study = prereg["study"]
    evaluation = prereg["evaluation"]
    prereg_benchmark = prereg["background_benchmark"]
    performance = _pick(state, "performance", default={})
    benchmark = _pick(state, "benchmark", default={})

    status = str(_pick(state, "status", default=_pick(study, "status", default="PRE_REGISTERED_NOT_STARTED")))
    closed_trades = _to_int(_pick(state, "closed_trade_count", "closed_trades_count", default=None))
    trades = _trade_rows(state)
    if not closed_trades:
        closed_trades = len(trades)
    target_trades = _to_int(_pick(state, "closed_trades_target", default=evaluation["closed_trades_target"]), evaluation["closed_trades_target"])
    strategy_points = _curve_points(performance, "equity_curve")
    if not strategy_points:
        strategy_points = _curve_points(state, "equity_curve", "paper_equity_curve", "close_marked_equity_curve")
    buy_hold_points = _curve_points(benchmark, "equity_curve")
    if not buy_hold_points:
        buy_hold_points = _curve_points(state, "buy_hold_curve", "benchmark_curve", "buy_and_hold_curve")
    current_equity = _to_float(
        _pick(
            performance,
            "equity",
            "current_equity",
            default=_pick(
                state,
                "equity",
                "current_equity",
                "paper_equity",
                default=(strategy_points[-1]["equity"] if strategy_points else 1.0),
            ),
        ),
        1.0,
    )
    current_buy_hold = _to_float(
        _pick(benchmark, "equity", default=_pick(state, "buy_hold_equity", "benchmark_equity", default=(buy_hold_points[-1]["equity"] if buy_hold_points else 1.0))),
        1.0,
    )
    net_return = _to_float(_pick(performance, "net_return", default=_pick(state, "net_return", default=current_equity - 1.0)), current_equity - 1.0)
    buy_hold_return = _to_float(
        _pick(benchmark, "net_return", default=_pick(state, "buy_hold_return", "benchmark_return", default=current_buy_hold - 1.0)),
        current_buy_hold - 1.0,
    )
    drawdown = _to_float(_pick(state, "max_drawdown", "drawdown", "current_drawdown", default=0.0), 0.0)
    sharpe = _pick(performance, "per_trade_sharpe", default=_pick(state, "per_trade_sharpe", "trade_sharpe", default=None))
    open_position = _pick(state, "open_position", "position", default=0)
    latest_signal = _pick(state, "latest_signal", "target_position", default="NA")

    trade_cells = []
    if trades:
        for trade in trades[::-1]:
            trade_cells.append(
                "<tr>"
                f"<td>{html.escape(trade['entry_time'])}</td>"
                f"<td>{html.escape(trade['exit_time'])}</td>"
                f"<td>{html.escape(trade['side'])}</td>"
                f"<td>{html.escape(trade['bars'])}</td>"
                f"<td>{html.escape(trade['return'])}</td>"
                f"<td>{html.escape(trade['reason'])}</td>"
                "</tr>"
            )
    else:
        trade_cells.append('<tr><td colspan="6" class="muted">Brak zamknietych transakcji.</td></tr>')

    audit_rows = []
    for label, value in _audit_rows(prereg, state):
        audit_rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Forward Test Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7f4;
      --panel: #ffffff;
      --ink: #12211d;
      --muted: #54645f;
      --line: #d5e1da;
      --accent: #0f766e;
      --accent-soft: #d9f4f0;
      --warn: #a16207;
      --warn-soft: #fff3c4;
      --fail: #b42318;
      --fail-soft: #fee4e2;
      --pass: #067647;
      --pass-soft: #d1fadf;
      --run: #1d4ed8;
      --run-soft: #dbeafe;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, #e6f4ea 0, transparent 26%),
        linear-gradient(180deg, #f8faf8 0%, var(--bg) 100%);
      color: var(--ink);
      font: 16px/1.45 Menlo, Consolas, Monaco, monospace;
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      display: grid;
      gap: 16px;
      grid-template-columns: 2.1fr 1fr;
      align-items: start;
      margin-bottom: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 14px 30px rgba(18, 33, 29, 0.06);
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 1.9rem; }}
    h2 {{ font-size: 1rem; margin-bottom: 12px; }}
    .muted {{ color: var(--muted); }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.88rem;
      margin-bottom: 14px;
      border: 1px solid transparent;
    }}
    .status-pending {{ background: var(--accent-soft); color: var(--accent); border-color: #9ed8d0; }}
    .status-running {{ background: var(--run-soft); color: var(--run); border-color: #93c5fd; }}
    .status-pass {{ background: var(--pass-soft); color: var(--pass); border-color: #86efac; }}
    .status-fail {{ background: var(--fail-soft); color: var(--fail); border-color: #fda29b; }}
    .status-underpowered {{ background: var(--warn-soft); color: var(--warn); border-color: #facc15; }}
    .summary {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin: 20px 0;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: #fbfdfb;
    }}
    .kpi .label {{ color: var(--muted); font-size: 0.82rem; margin-bottom: 8px; }}
    .kpi .value {{ font-size: 1.35rem; }}
    .chart-wrap {{ margin-top: 8px; }}
    .legend {{ display: flex; gap: 18px; color: var(--muted); font-size: 0.86rem; margin-top: 10px; }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: -1px;
    }}
    .legend .strategy::before {{ background: #0f766e; }}
    .legend .buyhold::before {{ background: #6aa7ff; }}
    .empty-chart {{
      min-height: 280px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 16px;
      color: var(--muted);
      background: #fbfdfb;
    }}
    .chart-bg {{ fill: #fbfdfb; stroke: #d5e1da; }}
    .axis-line {{ stroke: #d5e1da; stroke-width: 1; }}
    .axis-label {{ fill: #6b7c76; font-size: 11px; }}
    .lower {{
      display: grid;
      gap: 20px;
      grid-template-columns: 1.35fr 1fr;
      margin-top: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .audit th {{
      width: 42%;
      color: var(--muted);
      font-weight: 600;
    }}
    .footer-note {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 0.82rem;
    }}
    @media (max-width: 960px) {{
      .hero, .lower, .summary {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <article class="panel">
        <div class="status-pill {_status_class(status)}">{html.escape(status)}</div>
        <h1>{html.escape(study["id"])}</h1>
        <p class="muted">{html.escape(prereg["study"]["purpose"])}</p>
        <div class="summary">
          <div class="kpi">
            <div class="label">Closed trades</div>
            <div class="value">{closed_trades}/{target_trades}</div>
          </div>
          <div class="kpi">
            <div class="label">Strategy equity</div>
            <div class="value">{_fmt_num(current_equity, 4)}</div>
          </div>
          <div class="kpi">
            <div class="label">Net return</div>
            <div class="value">{_pct(net_return)}</div>
          </div>
          <div class="kpi">
            <div class="label">Buy and hold</div>
            <div class="value">{_pct(buy_hold_return)}</div>
          </div>
        </div>
        <div class="chart-wrap">{_svg_chart(strategy_points, buy_hold_points)}</div>
        <div class="legend">
          <span class="strategy">Frozen candidate</span>
          <span class="buyhold">{html.escape(str(_pick(benchmark, "name", default=prereg_benchmark["name"])))}</span>
        </div>
      </article>
      <aside class="panel">
        <h2>Runtime snapshot</h2>
        <table>
          <tr><th>Status</th><td>{html.escape(status)}</td></tr>
          <tr><th>Max drawdown</th><td>{_pct(drawdown)}</td></tr>
          <tr><th>Per-trade Sharpe</th><td>{html.escape(_fmt_num(sharpe, 4) if sharpe is not None else "NA")}</td></tr>
          <tr><th>Open position</th><td>{html.escape(str(open_position))}</td></tr>
          <tr><th>Latest signal</th><td>{html.escape(str(latest_signal))}</td></tr>
          <tr><th>First eligible candle</th><td>{html.escape(str(_pick(state, "first_eligible_open_utc", default=study["first_eligible_candle_open_utc"])))}</td></tr>
          <tr><th>Underpowered deadline</th><td>{html.escape(str(_pick(state, "underpowered_deadline_utc", default=study["underpowered_deadline_utc"])))}</td></tr>
          <tr><th>Last processed open</th><td>{html.escape(str(_pick(state, "last_processed_open_utc", default="NA")))}</td></tr>
          <tr><th>Pass rule</th><td>{html.escape(str(evaluation["pass_rule"]))}</td></tr>
          <tr><th>Early kill</th><td>{html.escape(str(evaluation["early_kill_rule"]["comparison"]))}</td></tr>
        </table>
        <p class="footer-note">No live orders. Public Binance OHLCV only. Files are append-only from the workflow point of view.</p>
      </aside>
    </section>

    <section class="lower">
      <article class="panel">
        <h2>All closed trades</h2>
        <table>
          <thead>
            <tr>
              <th>Entry</th>
              <th>Exit</th>
              <th>Side</th>
              <th>Bars</th>
              <th>Return</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {''.join(trade_cells)}
          </tbody>
        </table>
      </article>
      <article class="panel">
        <h2>Hashes and audit</h2>
        <table class="audit">
          <tbody>
            {''.join(audit_rows)}
          </tbody>
        </table>
      </article>
    </section>
  </main>
</body>
</html>
"""


def write_dashboard(
    *,
    prereg_path: Path = PREREG_PATH,
    state_path: Path = STATE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    for path in (prereg_path, state_path, output_path):
        _assert_path_allowed(path)
    prereg = _read_json(prereg_path)
    state = _read_optional_json(state_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard(prereg, state), encoding="utf-8")
    return output_path


def _assert_path_allowed(path: Path) -> None:
    candidates = (path, path.resolve(strict=False))
    if any(part.casefold() == "holdout" for candidate in candidates for part in candidate.parts):
        raise ValueError("dashboard paths must not contain a holdout component")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the static forward-test dashboard.")
    parser.add_argument("--prereg", type=Path, default=PREREG_PATH)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    write_dashboard(prereg_path=args.prereg, state_path=args.state, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
