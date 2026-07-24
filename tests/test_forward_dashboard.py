from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from forward_test.dashboard import HEIGHT, PADDING_Y, _svg_chart, render_dashboard, write_dashboard


def _line_points(svg: str, color: str) -> list[tuple[float, float]]:
    match = re.search(
        rf'<polyline fill="none" stroke="{re.escape(color)}"[^>]*points="([^"]+)"',
        svg,
    )
    assert match is not None
    return [tuple(map(float, point.split(","))) for point in match.group(1).split()]


def test_equity_curves_share_one_y_scale() -> None:
    strategy = [
        {"timestamp": "t0", "equity": 1.0},
        {"timestamp": "t1", "equity": 2.0},
    ]
    benchmark = [
        {"timestamp": "t0", "equity": 1.0},
        {"timestamp": "t1", "equity": 10.0},
    ]

    svg = _svg_chart(strategy, benchmark)
    strategy_points = _line_points(svg, "#0f766e")
    benchmark_points = _line_points(svg, "#6aa7ff")

    assert strategy_points[0][1] == pytest.approx(benchmark_points[0][1])
    assert benchmark_points[1][1] == pytest.approx(PADDING_Y)
    assert strategy_points[1][1] == pytest.approx(
        PADDING_Y + (HEIGHT - 2 * PADDING_Y) * (1.0 - 1.0 / 9.0),
        abs=0.01,
    )
    assert strategy_points[1][1] > benchmark_points[1][1]


def test_equity_curves_align_shared_timestamps_on_x_axis() -> None:
    strategy = [
        {"timestamp": "t0", "equity": 1.0},
        {"timestamp": "t1", "equity": 1.1},
        {"timestamp": "t2", "equity": 1.2},
    ]
    benchmark = [
        {"timestamp": "t0", "equity": 1.0},
        {"timestamp": "t1", "equity": 1.05},
    ]

    svg = _svg_chart(strategy, benchmark)
    strategy_points = _line_points(svg, "#0f766e")
    benchmark_points = _line_points(svg, "#6aa7ff")

    assert strategy_points[0][0] == pytest.approx(benchmark_points[0][0])
    assert strategy_points[1][0] == pytest.approx(benchmark_points[1][0])


def test_dashboard_lists_all_thirty_closed_trades() -> None:
    prereg = json.loads(Path("forward_test/prereg_forward.json").read_text(encoding="utf-8"))
    trades = [
        {
            "entry_time": f"entry-{index}",
            "exit_time": f"exit-{index}",
            "side": "short",
            "bars": 20,
            "net_return": 0.01,
            "reason": "time",
        }
        for index in range(30)
    ]

    rendered = render_dashboard(prereg, {"status": "RUNNING", "closed_trades": trades})

    assert "entry-0" in rendered
    assert "entry-29" in rendered
    assert rendered.count("<td>short</td>") == 30


def test_dashboard_rejects_holdout_paths_before_io(tmp_path: Path) -> None:
    forbidden = tmp_path / "holdout" / "state.json"

    with pytest.raises(ValueError, match="holdout component"):
        write_dashboard(state_path=forbidden, output_path=tmp_path / "dashboard.html")

    assert not forbidden.parent.exists()
    assert not (tmp_path / "dashboard.html").exists()
