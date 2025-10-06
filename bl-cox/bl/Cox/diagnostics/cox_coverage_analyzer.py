#!/usr/bin/env python3
"""Cox Coverage Analyzer
=========================

Diagnose-Skript zur Überprüfung der Churn-Coverage zwischen Survival-Panel
und finalem Trainingsdatensatz (Merge mit `customer_churn_details`).

Funktionen:
- Survival-Panel aus Stage0-Daten erzeugen
- Churn-Events und Zensierungen je Jahr zählen
- Coverage nach Merge mit Feature-Tabelle berechnen
- Historische Verfügbarkeit (History-Länge) je Jahr analysieren
- Plots und Markdown-Report generieren

Das Skript kann als eigenständiges CLI-Tool ausgeführt werden.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
MODULE_ROOT = CURRENT_FILE.parents[3]  # .../bl-cox
REPO_ROOT = CURRENT_FILE.parents[4]    # Projektwurzel

for candidate in (MODULE_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from config.paths_config import ProjectPaths
from bl.Cox.cox_data_loader import CoxDataLoader


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class CoverageResult:
    year: int
    events_panel: int
    censored_panel: int
    events_train: int
    coverage_rate: float


@dataclass
class HistoryInsight:
    year: int
    median_months: float
    percentile_25: float
    percentile_75: float
    sample_size: int


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _logger() -> logging.Logger:
    logger = logging.getLogger("cox_coverage_analyzer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _months_add(yyyymm: int, months: int = 1) -> int:
    """Addiert Monate auf YYYYMM-Format."""
    year = yyyymm // 100
    month = yyyymm % 100
    total = year * 12 + (month - 1) + months
    new_year = total // 12
    new_month = total % 12 + 1
    return new_year * 100 + new_month


def _ensure_directories(report_dir: Path) -> Tuple[Path, Path]:
    ProjectPaths.ensure_directory_exists(report_dir)
    plots_dir = report_dir / "plots"
    ProjectPaths.ensure_directory_exists(plots_dir)
    return report_dir, plots_dir


def _load_json_db_table(table_name: str) -> pd.DataFrame:
    """Lädt eine Tabelle aus der JSON-Datenbank in einen DataFrame."""
    db_path = ProjectPaths.dynamic_system_outputs_directory() / "churn_database.json"
    if not db_path.exists():
        raise FileNotFoundError(f"JSON-Datenbank nicht gefunden: {db_path}")

    with db_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    table = data.get("tables", {}).get(table_name, {})
    records = table.get("records", [])
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "Kunde" in df.columns:
        df["Kunde"] = pd.to_numeric(df["Kunde"], errors="coerce").astype("Int64")
    return df


def _determine_cutoff(stage0: pd.DataFrame) -> int:
    max_tb = int(pd.to_numeric(stage0["I_TIMEBASE"], errors="coerce").dropna().max())
    return _months_add(max_tb, months=1)


def _prepare_survival_panel(loader: CoxDataLoader, stage0: pd.DataFrame) -> pd.DataFrame:
    loader.cutoff_exclusive = _determine_cutoff(stage0)
    panel = loader.create_survival_panel(stage0)
    panel["event_year"] = (pd.to_numeric(panel["last_observed"], errors="coerce") // 100).astype("Int64")
    panel["censored_year"] = (
        pd.to_numeric(panel["last_observed"], errors="coerce") // 100
    ).astype("Int64")
    return panel


def _select_latest_features(features: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    if features.empty:
        return features

    prepared = features.copy()
    if "Letzte_Timebase" in prepared.columns:
        prepared["Letzte_Timebase"] = pd.to_numeric(prepared["Letzte_Timebase"], errors="coerce")
        prepared = prepared[prepared["Letzte_Timebase"].notnull()].copy()
        prepared["Letzte_Timebase"] = prepared["Letzte_Timebase"].astype("Int64")
        prepared = prepared[prepared["Letzte_Timebase"] <= cutoff]
        prepared.sort_values(["Kunde", "Letzte_Timebase"], inplace=True)
    else:
        prepared.sort_values(["Kunde"], inplace=True)

    prepared = prepared.drop_duplicates(subset=["Kunde"], keep="last")
    return prepared


def _merge_training_data(panel: pd.DataFrame, features: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=list(panel.columns))

    filtered_features = _select_latest_features(features, cutoff)
    feature_cols = [col for col in filtered_features.columns if col != "Kunde"]
    merged = panel.merge(filtered_features[["Kunde"] + feature_cols], on="Kunde", how="inner")
    return merged


def _compute_yearly_coverage(panel: pd.DataFrame, merged: pd.DataFrame) -> List[CoverageResult]:
    panel_events = panel[panel["event"] == 1]
    panel_censored = panel[panel["event"] == 0]
    train_events = merged[merged["event"] == 1] if not merged.empty else merged

    years = sorted(set(panel_events["event_year"].dropna().astype(int).tolist()))
    results: List[CoverageResult] = []

    for year in years:
        events_panel = int((panel_events["event_year"] == year).sum())
        censored_panel = int((panel_censored["censored_year"] == year).sum())
        events_train = int((train_events["event_year"] == year).sum()) if not train_events.empty else 0
        coverage = float(events_train / events_panel) if events_panel else 0.0
        results.append(
            CoverageResult(
                year=year,
                events_panel=events_panel,
                censored_panel=censored_panel,
                events_train=events_train,
                coverage_rate=coverage,
            )
        )

    return results


def _compute_history_insights(
    stage0: pd.DataFrame,
    panel: pd.DataFrame,
    loader: CoxDataLoader,
) -> pd.DataFrame:
    stage0 = stage0.copy()
    stage0["I_TIMEBASE"] = pd.to_numeric(stage0["I_TIMEBASE"], errors="coerce").astype("Int64")
    stage0.sort_values(["Kunde", "I_TIMEBASE"], inplace=True)

    history_records: List[Dict[str, float]] = []

    for _, row in panel.iterrows():
        if row["event"] != 1:
            continue

        kunde = row["Kunde"]
        event_timebase = int(row.get("last_observed", 0))
        if pd.isna(event_timebase):
            continue

        kunde_history = stage0[(stage0["Kunde"] == kunde) & (stage0["I_TIMEBASE"] < event_timebase)]
        if kunde_history.empty:
            history_months = 0
            observations = 0
        else:
            first_tb = int(kunde_history["I_TIMEBASE"].min())
            last_tb = int(kunde_history["I_TIMEBASE"].max())
            history_months = loader.months_diff(first_tb, last_tb) + 1
            observations = int(kunde_history["I_TIMEBASE"].nunique())

        history_records.append(
            {
                "Kunde": kunde,
                "event_year": int(row["event_year"]),
                "history_months": history_months,
                "history_observations": observations,
            }
        )

    return pd.DataFrame(history_records)


def _summarize_history(history_df: pd.DataFrame) -> List[HistoryInsight]:
    if history_df.empty:
        return []

    insights: List[HistoryInsight] = []
    grouped = history_df.groupby("event_year")
    for year, group in grouped:
        months_values = group["history_months"].astype(float)
        insights.append(
            HistoryInsight(
                year=int(year),
                median_months=float(np.nanmedian(months_values)),
                percentile_25=float(np.nanpercentile(months_values, 25)),
                percentile_75=float(np.nanpercentile(months_values, 75)),
                sample_size=int(len(group)),
            )
        )
    return insights


def _plot_events_vs_training(results: List[CoverageResult], plots_dir: Path) -> Path:
    plt.switch_backend("Agg")
    years = [r.year for r in results]
    panel_counts = [r.events_panel for r in results]
    train_counts = [r.events_train for r in results]

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, panel_counts, width, label="Survival Panel")
    ax.bar(x + width / 2, train_counts, width, label="Training nach Merge")
    ax.set_title("Churn-Events je Jahr")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl Events")
    ax.set_xticks(x, years)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plot_path = plots_dir / "cox_events_panel_vs_train.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _plot_coverage(results: List[CoverageResult], plots_dir: Path) -> Path:
    plt.switch_backend("Agg")
    years = [r.year for r in results]
    coverage_rates = [r.coverage_rate * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, coverage_rates, marker="o", color="#1f77b4")
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50%-Schwelle")
    ax.set_title("Coverage (%) je Jahr")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Coverage [%]")
    ax.set_ylim(0, max(100, max(coverage_rates) * 1.1 if coverage_rates else 100))
    ax.grid(alpha=0.3)
    ax.legend()

    plot_path = plots_dir / "cox_coverage_by_year.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _build_report_table(results: List[CoverageResult]) -> pd.DataFrame:
    data = {
        "year": [r.year for r in results],
        "events_panel": [r.events_panel for r in results],
        "censored_panel": [r.censored_panel for r in results],
        "events_train": [r.events_train for r in results],
        "coverage_pct": [round(r.coverage_rate * 100, 2) for r in results],
    }
    return pd.DataFrame(data)


def _format_history_section(insights: List[HistoryInsight]) -> str:
    if not insights:
        return "Keine Events im Survival-Panel – keine History-Auswertung möglich."

    lines = ["Jahr | Median Monate | P25 | P75 | Events", "--- | --- | --- | --- | ---"]
    for item in sorted(insights, key=lambda i: i.year):
        lines.append(
            f"{item.year} | {item.median_months:.1f} | {item.percentile_25:.1f} | "
            f"{item.percentile_75:.1f} | {item.sample_size}"
        )
    return "\n".join(lines)


def _craft_recommendations(
    coverage_results: List[CoverageResult],
    history_insights: List[HistoryInsight],
) -> List[str]:
    recommendations: List[str] = []

    if history_insights:
        medians = [ins.median_months for ins in history_insights]
        global_median = float(np.nanmedian(medians))
        recommendations.append(
            f"Mindestens {int(round(global_median))} Monate Historie bereitstellen, "
            "damit >50% der Churn-Fälle abgedeckt werden."
        )

        low_years = [ins for ins in history_insights if ins.median_months < global_median * 0.75]
        for ins in low_years:
            recommendations.append(
                f"Jahr {ins.year}: nur Median {ins.median_months:.1f} Monate verfügbar – "
                "expanding Fenster (mit shift(1)) oder kürzere Rolling-Fenster verwenden."
            )

    weak_years = [r for r in coverage_results if r.coverage_rate < 0.5]
    for res in weak_years:
        recommendations.append(
            f"Coverage {res.year}: {res.coverage_rate*100:.1f}% – Feature-Merge prüfen "
            "(fehlende Kunden in `customer_churn_details`) und History rückwärts erweitern."
        )

    if not recommendations:
        recommendations.append("Coverage stabil >50% – aktuelle Pipeline beibehalten.")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for rec in recommendations:
        if rec not in seen:
            deduped.append(rec)
            seen.add(rec)
    return deduped


def _write_markdown_report(
    table_df: pd.DataFrame,
    events_plot: Path,
    coverage_plot: Path,
    history_section: str,
    recommendations: List[str],
    coverage_results: List[CoverageResult],
    report_path: Path,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    coverage_below_50 = [r for r in coverage_results if r.coverage_rate < 0.5]

    lines: List[str] = [
        "# Cox Coverage & Rolling Report",
        f"*Generated: {now}*",
        "",
        "## Coverage-Übersicht",
        table_df.to_markdown(index=False),
        "",
        "![Events je Jahr](plots/cox_events_panel_vs_train.png)",
        "",
        "![Coverage je Jahr](plots/cox_coverage_by_year.png)",
        "",
        "## Jahre mit Coverage < 50%",
    ]

    if coverage_below_50:
        for item in coverage_below_50:
            lines.append(
                f"- **{item.year}**: {item.coverage_rate*100:.1f}% "
                f"({item.events_train}/{item.events_panel} Events im Training)"
            )
    else:
        lines.append("- Keine Jahre unter 50%.")

    lines.extend(
        [
            "",
            "## History-Verfügbarkeit vor dem Event",
            history_section,
            "",
            "## Rolling-Empfehlung (leckagefrei)",
            "```python",
            "customer_data = customer_data.sort_values(['Kunde', 'I_TIMEBASE'])",
            "for feature in ROLLING_COLS:",
            "    customer_data[f'{feature}_roll3'] = (",
            "        customer_data.groupby('Kunde')[feature]",
            "        .shift(1)",
            "        .rolling(window=3, min_periods=1)",
            "        .mean()",
            "    )",
            "```",
            "",
            "**Prüfhinweis:** Keine forward-looking Berechnungen (shift(1) vor rolling) – bestätigt.",
            "",
            "## Handlungsempfehlungen",
        ]
    )

    for rec in recommendations:
        lines.append(f"- {rec}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _console_summary(results: List[CoverageResult]) -> None:
    logger = _logger()
    sorted_results = sorted(results, key=lambda r: r.coverage_rate)
    top3 = sorted_results[:3]
    summary = ", ".join(
        f"{item.year}: {item.coverage_rate*100:.1f}% ({item.events_train}/{item.events_panel})"
        for item in top3
    )
    logger.info("Schwächste Coverage (Top 3): %s", summary if summary else "Keine Events")


def _print_table_to_console(table_df: pd.DataFrame) -> None:
    logger = _logger()
    logger.info("Jahresübersicht:\n%s", table_df.to_markdown(index=False))


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def run() -> Dict[str, pd.DataFrame]:
    logger = _logger()
    logger.info("Starte Cox Coverage Analyse")

    loader = CoxDataLoader()
    stage0 = loader.load_stage0_data()
    logger.info(
        "Stage0: %s Records, %s Kunden, Zeitraum %s-%s",
        len(stage0),
        stage0["Kunde"].nunique(),
        int(stage0["I_TIMEBASE"].min()),
        int(stage0["I_TIMEBASE"].max()),
    )

    panel = _prepare_survival_panel(loader, stage0)
    logger.info(
        "Survival-Panel: %s Kunden, Events=%s, Censored=%s",
        len(panel),
        int(panel["event"].sum()),
        int((panel["event"] == 0).sum()),
    )

    features = _load_json_db_table("customer_churn_details")
    if features.empty:
        logger.warning("customer_churn_details leer – Coverage kann nicht berechnet werden")

    merged = _merge_training_data(panel, features, cutoff=loader.cutoff_exclusive - 1)
    logger.info("Training nach Merge: %s Kunden", len(merged))

    coverage_results = _compute_yearly_coverage(panel, merged)
    table_df = _build_report_table(coverage_results)

    history_df = _compute_history_insights(stage0, panel, loader)
    history_insights = _summarize_history(history_df)
    history_section = _format_history_section(history_insights)
    recommendations = _craft_recommendations(coverage_results, history_insights)

    report_dir = ProjectPaths.project_root() / "reports"
    _, plots_dir = _ensure_directories(report_dir)
    events_plot = _plot_events_vs_training(coverage_results, plots_dir)
    coverage_plot = _plot_coverage(coverage_results, plots_dir)

    report_path = report_dir / "cox_coverage_and_rolling.md"
    _write_markdown_report(
        table_df=table_df,
        events_plot=events_plot,
        coverage_plot=coverage_plot,
        history_section=history_section,
        recommendations=recommendations,
        coverage_results=coverage_results,
        report_path=report_path,
    )

    _print_table_to_console(table_df)
    _console_summary(coverage_results)

    logger.info("Report gespeichert unter %s", report_path)

    return {
        "survival_panel": panel,
        "training_data": merged,
        "coverage_table": table_df,
        "history_details": history_df,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()

