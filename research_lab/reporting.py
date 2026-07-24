from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from research_lab.config import PRE_REGISTERED_CONFIG


def write_report(
    *,
    output_dir: str | Path,
    report_path: str | Path | None = None,
    decision: Mapping[str, Any],
    hypothesis_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    holdout_result: Mapping[str, Any] | None = None,
) -> Path:
    """Write the confirmatory study report in Polish."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = Path(report_path) if report_path is not None else output / "report2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Raport potwierdzajacy H1-H6")
    lines.append("")
    lines.append("## Decyzja")
    if decision.get("decision") == "WINNER_FROZEN":
        winner = decision.get("winner") or {}
        lines.append(
            "Wynik: kandydat spelnil wszystkie zamrozone reguly i zostal zapisany "
            f"jako zwyciezca: `{winner.get('candidate_id')}`."
        )
    else:
        lines.append(
            "Wynik: NO_EDGE. Zaden kandydat nie spelnil kompletu regul prerejestracji. "
            "Holdout nie zostal odczytany i nie wolno go odczytywac bez zamrozonego zwyciezcy."
        )
    lines.append("")
    lines.append("## Zakres prerejestracji")
    lines.append(f"- Commit prerejestracji: `{decision.get('prereg_commit')}`")
    lines.append(f"- Fingerprint konfiguracji: `{decision.get('config_fingerprint')}`")
    lines.append(f"- Fingerprint manifestu danych: `{decision.get('manifest_fingerprint')}`")
    lines.append(f"- Liczba kandydatow: `{decision.get('candidate_count')}`")
    lines.append("- Minimalna liczba transakcji OOS: `30`")
    lines.append("- Rodzina testow wielokrotnych: `186` kandydatow, `500` permutacji")
    lines.append("- WFO: kazdy kandydat jest staly, bez adaptacyjnego wyboru parametrow; scoring dotyczy stitched OOS.")
    lines.append("- Egzekucja: syntetyczna pozycja 1x; 0.15% kosztu na strone; bez funding, borrow i market impact.")
    lines.append("- H1 caveat: sygnal wolumenowy korzysta z proxy wolumenu dostepnego w OHLCV, bez danych order-book/tape.")
    lines.append("")
    lines.append("## Zamrozone uzasadnienia ekonomiczne")
    for hypothesis in PRE_REGISTERED_CONFIG["hypotheses"]:
        lines.append(
            f"- {hypothesis['id']} — {hypothesis['name']}: {hypothesis['economic_rationale']}"
        )
    lines.append("")
    lines.append("## Hipotezy")
    lines.extend(_hypothesis_table(candidates))
    lines.append("")
    if hypothesis_summary.empty:
        lines.append("Brak podsumowania hipotez.")
    else:
        for row in hypothesis_summary.to_dict("records"):
            passing_count = int(row.get("passing_candidates", 0) or 0)
            eligible_count = int(row.get("eligible_candidates", 0) or 0)
            status = "PASS" if passing_count else ("UNDERPOWERED" if eligible_count == 0 else "FAIL")
            nearest = row.get("nearest_candidate_id") or "brak"
            lines.append(
                f"- {row.get('hypothesis_id')}: {status}; kandydaci={int(row.get('candidate_count', 0) or 0)}, "
                f"eligible={eligible_count}, passing={passing_count}, najblizszy=`{nearest}`."
            )
    lines.append("")
    lines.append("## Najblizszy kandydat i powody porazki")
    nearest = decision.get("nearest_candidate") or {}
    if nearest:
        failures = _failure_list(nearest.get("failure_reasons"))
        lines.append(f"Najblizszy kandydat: `{nearest.get('candidate_id')}`.")
        if failures:
            lines.append("Powody niespelnienia regul:")
            for reason in failures:
                lines.append(f"- {reason}")
        else:
            lines.append("Brak powodow porazki dla najblizszego kandydata.")
    else:
        lines.append("Nie wskazano najblizszego kandydata.")
    lines.append("")
    lines.append("## Ostrzezenia mocy")
    warnings = _power_warnings(candidates)
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- Brak automatycznych ostrzezen mocy dla syntetycznego przebiegu.")
    lines.append("")
    lines.append("## Reguly statystyczne")
    lines.append(
        "DSR liczony jest dla calej rodziny 186 kandydatow; niezdefiniowane Sharpe w rodzinie "
        "sa mapowane na 0 tylko na potrzeby korekty rodzinnej. Test permutacyjny raportuje "
        "najlepszy Sharpe rodziny dla kazdej z 500 permutacji."
    )
    lines.append(
        "Regula przejscia: net OOS return > 0, transakcje OOS >= 30, przewaga nad cost-matched buy-and-hold "
        "w zwrocie i Sharpe, DSR > 0.95, observed Sharpe > q95 permutacji oraz familywise empirical p <= 0.05."
    )
    lines.append(
        "Nota zrodla/formuly: Deflated Sharpe Ratio wedlug Bailey i Lopez de Prado, "
        "DOI 10.2139/ssrn.2460551."
    )
    lines.append("")
    lines.append("## Holdout")
    if holdout_result is None:
        lines.append("Holdout: zapieczetowany 6-miesieczny zbior finalny; nieodczytany i nietkniety w tej fazie.")
    else:
        status = holdout_result.get("final_status")
        total_return = holdout_result.get("total_return")
        lines.append(f"Holdout: `{status}`, total_return={_fmt(total_return)}.")
        if holdout_result.get("final_negative"):
            lines.append("Ujemny calkowity zwrot holdout oznacza odrzucenie wyniku finalnego.")
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")
    results_copy = output / "report2.md"
    if results_copy.resolve() != path.resolve():
        results_copy.write_text(content, encoding="utf-8")
    return path


def load_report_inputs(output_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    decision = json.loads((output / "study_decision.json").read_text(encoding="utf-8"))
    hypotheses = pd.read_csv(output / "hypothesis_summary.csv")
    candidates = pd.read_csv(output / "candidates.csv")
    return decision, hypotheses, candidates


def _power_warnings(candidates: pd.DataFrame) -> list[str]:
    if candidates.empty:
        return ["Brak kandydatow w artefaktach wynikowych."]
    warnings: list[str] = []
    if "trades" in candidates:
        eligible = int((pd.to_numeric(candidates["trades"], errors="coerce") >= 30).sum())
        if eligible == 0:
            warnings.append("Zaden kandydat nie osiagnal progu 30 transakcji OOS; moc testu jest ograniczona.")
        elif eligible < max(5, int(0.05 * len(candidates))):
            warnings.append("Bardzo malo kandydatow osiagnelo prog 30 transakcji OOS; interpretacja mocy wymaga ostroznosci.")
    if "fold_count" in candidates:
        folds = pd.to_numeric(candidates["fold_count"], errors="coerce")
        if folds.notna().any() and int((folds <= 1).sum()) > 0:
            warnings.append("Czesc kandydatow ma jeden lub zero kompletnych foldow OOS.")
    return warnings


def _hypothesis_table(candidates: pd.DataFrame) -> list[str]:
    header = (
        "| Hipoteza | Verdict | candidate_id | symbol | timeframe | params | OOS total return | "
        "annualized Sharpe | DSR | familywise permutation p | trades | benchmark return | annualized benchmark Sharpe |"
    )
    separator = "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, separator]
    for hypothesis_id in ("H1", "H2", "H3", "H4", "H5", "H6"):
        group = candidates.loc[candidates["hypothesis_id"] == hypothesis_id] if "hypothesis_id" in candidates else pd.DataFrame()
        if group.empty:
            rows.append(f"| {hypothesis_id} | UNDERPOWERED | brak |  |  |  | NA | NA | NA | NA | 0 | NA | NA |")
            continue
        passing = group.loc[group.get("passes_all", False) == True] if "passes_all" in group else pd.DataFrame()  # noqa: E712
        eligible = int((pd.to_numeric(group.get("trades", pd.Series(dtype=float)), errors="coerce") >= 30).sum())
        verdict = "PASS" if not passing.empty else ("UNDERPOWERED" if eligible == 0 else "FAIL")
        ranked = _rank_candidates(passing if not passing.empty else group)
        row = ranked.iloc[0]
        rows.append(
            "| "
            + " | ".join(
                [
                    hypothesis_id,
                    verdict,
                    _cell(row.get("candidate_id")),
                    _cell(row.get("symbol")),
                    _cell(row.get("timeframe")),
                    _cell(_params(row)),
                    _fmt(row.get("total_return")),
                    _fmt(row.get("Sharpe")),
                    _fmt(row.get("dsr")),
                    _fmt(row.get("permutation_empirical_p")),
                    _fmt(row.get("trades")),
                    _fmt(row.get("benchmark_total_return")),
                    _fmt(row.get("benchmark_Sharpe")),
                ]
            )
            + " |"
        )
    return rows


def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["_rank_dsr"] = pd.to_numeric(ranked.get("dsr", pd.Series(index=ranked.index)), errors="coerce").fillna(-math.inf)
    ranked["_rank_p"] = pd.to_numeric(
        ranked.get("permutation_empirical_p", pd.Series(index=ranked.index)),
        errors="coerce",
    ).fillna(math.inf)
    tie_breakers = [
        "hypothesis_id",
        "symbol",
        "timeframe",
        "side",
        "session_utc",
        "mode",
        "hold_bars",
        "streak_length",
        "candidate_index",
    ]
    for key in tie_breakers:
        if key not in ranked:
            ranked[key] = ""
    return ranked.sort_values(
        by=["_rank_dsr", "_rank_p", *tie_breakers],
        ascending=[False, True, *([True] * len(tie_breakers))],
        kind="mergesort",
    ).drop(columns=["_rank_dsr", "_rank_p"])


def _params(row: Mapping[str, Any]) -> str:
    parts = []
    for key in ("side", "session_utc", "mode", "hold_bars", "streak_length"):
        value = row.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value)
        if text and text != "nan":
            parts.append(f"{key}={text}")
    return ", ".join(parts) if parts else "-"


def _failure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split(";") if part]
    if isinstance(value, Sequence):
        return [str(part) for part in value if str(part)]
    return [str(value)]


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.6g}"
