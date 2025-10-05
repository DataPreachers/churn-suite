#!/usr/bin/env python3
"""
Churn Persistence Service
-------------------------

Zentrale Persistenz-Schicht für Churn-Ergebnisse in die JSON-Datenbank.
Einheitliche Speicherung für verschiedene Pipeline-Varianten (enhanced/standard).

Regeln:
- Pfade ausschließlich über paths_config (indirekt via ChurnJSONDatabase)
- Keine Saves hier – der aufrufende Orchestrator speichert einmal pro Experiment
- Nur echte, vorhandene Felder persistieren (keine Fallback-Werte)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from bl.json_database.churn_json_database import ChurnJSONDatabase


class ChurnPersistenceService:
    """
    Vereinheitlicht die Persistenz von Churn-Ergebnissen in die JSON-DB.

    Nutzung:
        svc = ChurnPersistenceService()
        svc.persist_to_db(db, experiment, results)
    """

    def persist_to_db(self,
                      db: ChurnJSONDatabase,
                      experiment: Dict[str, Any],
                      results: Dict[str, Any]) -> None:
        """
        Persistiert Modell- und Business-Metriken in die JSON-DB.

        Args:
            db: Offene JSON-DB Instanz (wird NICHT gespeichert)
            experiment: Experiment-Datensatz (inkl. experiment_id, Zeitangaben)
            results: Ergebnis-Dictionary der Pipeline (enhanced/standard)
        """
        if not isinstance(results, dict):
            return

        # Kontext-Zeitangaben aus Experiment ableiten
        exp_id = experiment.get("experiment_id")
        training_from = experiment.get("training_from")
        training_to = experiment.get("training_to")
        prediction_timebase = experiment.get("prediction_timebase")

        training_timebase = None
        if training_from and training_to:
            training_timebase = f"{training_from}-{training_to}"

        # Modell-Metriken – Backtest bevorzugen
        backtest_metrics = results.get("backtest_results") or {}
        if isinstance(backtest_metrics, dict) and backtest_metrics:
            self._persist_model_metrics(db,
                                        int(exp_id) if exp_id is not None else exp_id,
                                        backtest_metrics,
                                        data_split="backtest",
                                        training_timebase=training_timebase,
                                        prediction_timebase=prediction_timebase,
                                        model_version=results.get("model_version"))

        training_metrics = results.get("training_metrics") or {}
        if isinstance(training_metrics, dict) and training_metrics:
            self._persist_model_metrics(db,
                                        int(exp_id) if exp_id is not None else exp_id,
                                        training_metrics,
                                        data_split="training",
                                        training_timebase=training_timebase,
                                        prediction_timebase=prediction_timebase,
                                        model_version=results.get("model_version"))

        # Business-Metriken
        business_metrics = results.get("business_metrics") or {}
        if isinstance(business_metrics, dict) and business_metrics:
            self._persist_business_metrics(db,
                                           int(exp_id) if exp_id is not None else exp_id,
                                           business_metrics,
                                           prediction_timebase=prediction_timebase)

    # -----------------------------
    # Internals
    # -----------------------------

    def _persist_model_metrics(self,
                               db: ChurnJSONDatabase,
                               experiment_id: int,
                               metrics: Dict[str, Any],
                               data_split: str,
                               training_timebase: Optional[str],
                               prediction_timebase: Optional[str],
                               model_version: Optional[str]) -> None:
        # Nur existierende Felder übernehmen
        payload: Dict[str, Any] = {}
        for src_key, dst_key in (
            ("auc", "auc"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("f1_score", "f1"),
            ("threshold_used", "threshold_used"),
        ):
            if metrics.get(src_key) is not None and dst_key not in payload:
                payload[dst_key] = metrics.get(src_key)

        # Zeit-/Version-Kontext nur setzen, wenn vorhanden
        if training_timebase:
            payload["training_timebase"] = training_timebase
        if prediction_timebase:
            payload["prediction_timebase"] = prediction_timebase
        if model_version:
            payload["model_version"] = model_version

        if payload:
            try:
                db.add_churn_model_metrics(
                    experiment_id=experiment_id,
                    metrics=payload,
                    data_split=data_split,
                )
            except Exception:
                # Fehlertolerant – keine Saves hier
                pass

    def _persist_business_metrics(self,
                                  db: ChurnJSONDatabase,
                                  experiment_id: int,
                                  business_metrics: Dict[str, Any],
                                  prediction_timebase: Optional[str]) -> None:
        # Struktur: {'risk_segmentation': {...}, 'financial_impact': {...}, ...}
        out: Dict[str, Any] = {}

        rs = business_metrics.get("risk_segmentation") or {}
        fi = business_metrics.get("financial_impact") or {}

        # Risiko-Segmente
        for k_src, k_dst in (
            ("customers_at_risk", "customers_at_risk"),
            ("customers_high_risk", "customers_high_risk"),
            ("customers_medium_risk", "customers_medium_risk"),
            ("customers_low_risk", "customers_low_risk"),
            ("total_customers", "total_customers"),
        ):
            v = rs.get(k_src)
            if v is not None:
                out[k_dst] = v

        # Finanzieller Impact
        for k_src, k_dst in (
            ("potential_revenue_loss", "potential_revenue_loss"),
            ("prevention_cost_estimate", "prevention_cost_estimate"),
            ("roi_estimate", "roi_estimate"),
            ("avg_customer_value", "avg_customer_value"),
            ("max_customer_value", "max_customer_value"),
        ):
            v = fi.get(k_src)
            if v is not None:
                out[k_dst] = v

        if prediction_timebase:
            out["prediction_timebase"] = prediction_timebase

        if out:
            try:
                db.add_churn_business_metrics(
                    experiment_id=experiment_id,
                    business=out,
                )
            except Exception:
                # Fehlertolerant – keine Saves hier
                pass



