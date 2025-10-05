#!/usr/bin/env python3
"""
Churn Automatic Experiment Processor
====================================

Automatische Verarbeitung aller unverarbeiteten Churn-Experimente aus der experiments Tabelle.
Analog zur erfolgreichen cox_auto_processor.py Implementation.

Features:
- Automatische Erkennung unverarbeiteter Churn-Experimente
- Batch-Verarbeitung mit churn_working_main.py Pipeline
- Status-Management: 'created' → 'processed'
- Robuste Fehlerbehandlung und Logging
- Performance Monitoring über alle Experimente

Autor: AI Assistant
Datum: 2025-01-27
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Füge Projekt-Root zum Python-Path hinzu
project_root = Path(__file__).parent.parent.parent.parent  # Gehe eine Ebene höher zur echten Root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "json-database" / "bl"))

from json_database.churn_json_database import ChurnJSONDatabase
# from bl.Churn.churn_working_main import ChurnWorkingPipeline  # AUSKOMMENTIERT  
from .enhanced_early_warning import EnhancedEarlyWarningSystem
from .churn_persistence_service import ChurnPersistenceService
from config.paths_config import ProjectPaths  # Zentrale Root-ProjectPaths
import json
import shutil
from .churn_constants import *

class ChurnAutoProcessor:
    """
    Automatischer Churn-Experiment-Prozessor
    Verarbeitet alle unverarbeiteten Experimente aus der experiments Tabelle
    """
    
    def __init__(self, training_from: str = "202001", training_to: str = "202312", 
                 prediction_timebase: str = "202401"):
        """
        Initialisiert Auto Processor
        
        Args:
            training_from: Default Training Start (YYYYMM)
            training_to: Default Training Ende (YYYYMM)
            prediction_timebase: Default Prediction Zeitpunkt (YYYYMM)
        """
        self.training_from = training_from
        self.training_to = training_to
        self.prediction_timebase = prediction_timebase
        self.logger = self._setup_logging()
        self.db = ChurnJSONDatabase()
        
        # Processing Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup für strukturiertes Logging"""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=LOG_FORMAT
        )
        return logging.getLogger('churn_auto_processor')

    def _progress(self, exp_id: int, phase: str, step: int, total: int, detail: str = "") -> None:
        try:
            msg = f"PROGRESS|exp_id={exp_id}|phase={phase}|step={step}|total={total}|detail={detail}"
            self.logger.info(msg)
        except Exception:
            pass
    
    def get_unprocessed_experiments(self) -> List[Dict[str, Any]]:
        """
        Ermittelt alle Churn-Experimente, die noch nicht verarbeitet wurden
        
        Returns:
            Liste unverarbeiteter Churn-Experimente
        """
        self.logger.info("🔍 Suche nach unverarbeiteten Churn-Experimenten")
        
        # Alle Churn-Experimente mit Status 'created'
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        churn_experiments = [
            exp for exp in experiments 
            if self._is_churn_experiment(exp.get("model_type", ""))
            and exp.get("status", "") == EXPERIMENT_STATUS['CREATED']
        ]
        
        self.logger.info(f"   📊 Gefundene Churn-Experimente: {len(churn_experiments)}")
        
        # Prüfe, welche bereits Churn-Daten haben (check alle 5 Churn-Tabellen)
        processed_experiment_ids = self._get_processed_experiment_ids()
        
        # Filtere unverarbeitete Experimente
        unprocessed = []
        for exp in churn_experiments:
            exp_id = exp.get("experiment_id")
            has_churn_data = exp_id in processed_experiment_ids
            
            if not has_churn_data:
                unprocessed.append(exp)
                self.logger.info(f"   📋 Unverarbeitet: ID {exp_id} - {exp.get('experiment_name', 'N/A')}")
            else:
                self.logger.info(f"   ✅ Bereits verarbeitet: ID {exp_id}")
        
        self.logger.info(f"🎯 Unverarbeitete Churn-Experimente: {len(unprocessed)}")
        return unprocessed
    
    def _is_churn_experiment(self, model_type: str) -> bool:
        """Prüft ob Experiment ein Churn-Experiment ist"""
        if not model_type:
            return False
        
        churn_keywords = ['churn', 'random_forest', 'rf_churn', 'classification']
        return any(keyword in model_type.lower() for keyword in churn_keywords)
    
    def _get_processed_experiment_ids(self) -> set:
        """Ermittelt alle Experiment-IDs die bereits Churn-Daten haben"""
        processed_ids = set()
        
        # Prüfe alle 5 Churn-Tabellen
        churn_table_names = [
            'churn_training_data', 'churn_predictions', 'churn_feature_importance',
            'churn_business_metrics'
        ]
        
        for table_name in churn_table_names:
            table_data = self.db.data.get("tables", {}).get(table_name, {}).get("records", [])
            for record in table_data:
                exp_id = record.get("id_experiments") or record.get("experiment_id")
                if exp_id:
                    processed_ids.add(exp_id)
        
        if processed_ids:
            self.logger.info(f"   📊 Experimente mit Churn-Daten: {sorted(processed_ids)}")
        
        return processed_ids
    
    def process_experiment(self, experiment: Dict[str, Any], 
                          custom_periods: Optional[Dict[str, str]] = None,
                          test_reduction: float = 0.0) -> bool:
        """
        Verarbeitet ein einzelnes Experiment mit der Churn-Pipeline
        
        Args:
            experiment: Experiment-Dictionary
            custom_periods: Optional custom training periods
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        exp_id = experiment.get("experiment_id")
        exp_name = experiment.get("experiment_name", "Unknown")
        
        self.logger.info(f"🚀 Starte Churn-Verarbeitung: Experiment {exp_id}")
        self.logger.info(f"   📋 Name: {exp_name}")
        self.logger.info(f"   🎯 Model Type: {experiment.get('model_type', 'N/A')}")
        try:
            self._progress(exp_id, 'start', 0, 1, 'Init')
        except Exception:
            pass
        
        try:
            # Zeitperioden bestimmen (aus Experiment oder Default)
            if custom_periods:
                training_from = custom_periods.get('training_from', self.training_from)
                training_to = custom_periods.get('training_to', self.training_to)
                prediction_timebase = custom_periods.get('prediction_timebase', self.prediction_timebase)
            else:
                # Versuche aus Experiment-Daten zu extrahieren
                training_from = experiment.get('training_from', self.training_from)
                training_to = experiment.get('training_to', self.training_to)
                prediction_timebase = experiment.get('prediction_timebase', self.prediction_timebase)
            
            self.logger.info(f"   📅 Training: {training_from} - {training_to}")
            
            # Status auf 'processing' setzen
            self._update_experiment_status(exp_id, EXPERIMENT_STATUS['PROCESSING'])
            self._progress(exp_id, 'status', 1, 3, 'processing')
            
            # Aktiviere Verarbeitung über EnhancedEarlyWarningSystem (bewährte Pipeline)
            # Merke aktuelle Experiment-ID für nachgelagerte Persistierung (Threshold-Metriken)
            try:
                self.current_experiment_id = int(exp_id)
            except Exception:
                self.current_experiment_id = exp_id
            backtest_from = experiment.get('backtest_from')
            backtest_to = experiment.get('backtest_to')
            self.logger.info(f"   📅 Backtest: {backtest_from} - {backtest_to}")

            # Validierung der Zeiträume (keine Fallbacks)
            if not training_from or not training_to or not backtest_from or not backtest_to:
                raise ValueError("Fehlende Zeitangaben im Experiment (training_from/to, backtest_from/to)")

            # Sicherstellen: rawdata ist stets geladen (ohne Deduplizierung)
            try:
                added = self.db.rebuild_rawdata_from_all_stage0_files_no_dedupe()
                self.logger.info(f"   📥 rawdata geladen (no-dedupe): {added} Records")
            except Exception as e:
                self.logger.warning(f"   ⚠️ rawdata Laden fehlgeschlagen: {e}")

            ees = EnhancedEarlyWarningSystem(experiment_id=exp_id if isinstance(exp_id, int) else None, test_reduction=test_reduction)
            # Experiment-spezifische Algorithmus-Konfiguration (falls vorhanden) an EES übergeben
            try:
                algo_cfg = (experiment.get('hyperparameters') or {}).get('algorithm_config')
                if isinstance(algo_cfg, dict) and algo_cfg:
                    ees._optimized_config_override = algo_cfg
                    self.logger.info("   🧩 Verwende experimentsspezifische Algorithmus-Konfiguration (override)")
            except Exception:
                pass
            start_time = datetime.now()
            self._progress(exp_id, 'analysis', 0, 3, 'run_complete_analysis')
            results = ees.run_complete_analysis(
                force_retrain=True,
                training_from=training_from,
                training_to=training_to,
                test_from=backtest_from,
                test_to=backtest_to
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Erfolg werten, wenn kein Fehler gemeldet wurde
            is_success = results is not None and ('error' not in results)
            if is_success:
                self.logger.info(f"✅ Experiment {exp_id} erfolgreich verarbeitet")
                self._progress(exp_id, 'analysis', 1, 3, 'completed')
                # AUC aus Backtest-Resultaten bevorzugen; fallback auf Training-Metriken
                backtest_auc = 0.0
                try:
                    backtest_auc = (results.get('backtest_results') or {}).get('auc', 0.0)
                    if not backtest_auc:
                        backtest_auc = (results.get('training_metrics') or {}).get('auc', 0.0)
                except Exception:
                    backtest_auc = 0.0
                self.logger.info(f"   🎯 AUC: {backtest_auc:.4f}")
                # Feature-Anzahl optional; wenn vorhanden in optimized_config/Feature-Set nutzbar
                self.logger.info(f"   📊 Features: {results.get('feature_count', 0)}")
                self.logger.info(f"   ⏱️ Laufzeit: {processing_time:.1f}s")

                # Einheitliche Persistenz: Modell-/Business-Metriken in JSON-DB
                try:
                    ChurnPersistenceService().persist_to_db(
                        self.db,
                        experiment,
                        results
                    )
                    self.logger.info("   💾 Persistenz-Service: model/business metrics vorbereitet")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Persistenz-Service Fehler: {e}")

                # Schreibe KPIs in experiment_kpis (minimal-invasiv)
                try:
                    self.db.add_experiment_kpi(exp_id, "auc", float(backtest_auc), "backtest")
                    br = results.get('backtest_results') or {}
                    if isinstance(br, dict):
                        if br.get('precision') is not None:
                            self.db.add_experiment_kpi(exp_id, "precision", float(br.get('precision', 0.0)), "backtest")
                        if br.get('recall') is not None:
                            self.db.add_experiment_kpi(exp_id, "recall", float(br.get('recall', 0.0)), "backtest")
                        f1v = br.get('f1_score', br.get('f1'))
                        if f1v is not None:
                            self.db.add_experiment_kpi(exp_id, "f1", float(f1v), "backtest")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ KPIs konnten nicht gespeichert werden: {e}")
                
                # NEU: Customer-Details aus Backtest in JSON-DB schreiben
                try:
                    models_dir = ProjectPaths.models_directory()
                    backtests = sorted(models_dir.glob("Enhanced_EarlyWarning_Backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if backtests:
                        latest_backtest = backtests[0]
                        self.logger.info(f"   📝 Customer-Details zur JSON-DB aus: {latest_backtest.name}")
                        ok = self.db.add_customer_churn_details_from_backtest(str(latest_backtest), experiment_id=exp_id)
                        if ok:
                            self.logger.info("   ✅ Customer-Details in Memory aktualisiert (persistiere am Ende des Experiments)")
                            self._progress(exp_id, 'persist', 1, 3, 'customer_details')
                        else:
                            self.logger.warning("   ⚠️ Customer-Details konnten nicht aktualisiert werden")

                        # NEU: backtest_results direkt importieren (für Threshold-Berechnung und SQL-Analysen)
                        try:
                            if self.db.add_backtest_results(str(latest_backtest), experiment_id=exp_id):
                                self.logger.info("   ✅ backtest_results aktualisiert")
                                self._progress(exp_id, 'persist', 2, 3, 'backtest_results')
                            else:
                                self.logger.warning("   ⚠️ backtest_results konnten nicht aktualisiert werden")
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Fehler beim Update von backtest_results: {e}")
                    else:
                        self.logger.warning("   ⚠️ Kein Backtest-JSON gefunden – Customer-Details übersprungen")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Fehler beim JSON-DB Update der Customer-Details: {e}")
                
                # Performance Details
                if results.get('target_achieved', False):
                    self.logger.info(f"   🎯 Target AUC erreicht: ✅")
                else:
                    self.logger.warning(f"   ⚠️ Target AUC verfehlt")
                
                # Detaillierte Schwellenwert-Metriken aus Backtest-JSON (klar gelabelt)
                try:
                    self._log_threshold_metrics_from_backtest()
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Konnte Threshold-Metriken nicht aus Backtest lesen: {e}")

                # Outbox-Export (Sink): Backtest + KPIs bereitstellen
                try:
                    if 'latest_backtest' in locals() and latest_backtest:
                        self._export_outbox_churn(exp_id, latest_backtest, results)
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Outbox-Export fehlgeschlagen: {e}")

                # Business Impact
                business_metrics = results.get('business_metrics', {})
                if business_metrics:
                    financial = business_metrics.get('financial_impact', {})
                    revenue_at_risk = financial.get('potential_revenue_loss', 0.0)
                    self.logger.info(f"   💰 Revenue at Risk: €{revenue_at_risk:,.0f}")
                
                # Status auf 'processed' setzen (persistiere am Ende)
                self._update_experiment_status(exp_id, EXPERIMENT_STATUS['PROCESSED'])
                self._progress(exp_id, 'status', 3, 3, 'processed')
                
                # Statistiken aktualisieren
                self.processed_count += 1
                self.total_processing_time += processing_time
                
                # Einmaliges Speichern aller Änderungen für dieses Experiment
                try:
                    self.db.save()
                    self.logger.info("   💾 JSON-DB gespeichert (einmal pro Experiment)")
                    self._progress(exp_id, 'done', 1, 1, 'saved')
                    # Vollständigen Outbox-Export für dieses Experiment ausführen
                    try:
                        self.db.export_churn_to_outbox(int(exp_id))
                        self.logger.info("   📦 Outbox-Export (Churn) abgeschlossen")
                    except Exception as _e:
                        self.logger.warning(f"   ⚠️ Outbox-Export (Churn) fehlgeschlagen: {_e}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Speichern der JSON-DB fehlgeschlagen: {e}")
                
                return True
            else:
                error_msg = results.get("error", "Unbekannter Fehler")
                self.logger.error(f"❌ Experiment {exp_id} fehlgeschlagen: {error_msg}")
                
                # Status auf 'failed' setzen
                self._update_experiment_status(exp_id, EXPERIMENT_STATUS['FAILED'])
                
                # Persistiere den Fehlerstatus einmalig
                try:
                    self.db.save()
                except Exception:
                    pass
                
                self.failed_count += 1
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Schwerwiegender Fehler bei Experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Status auf 'failed' setzen
            self._update_experiment_status(exp_id, EXPERIMENT_STATUS['FAILED'])
            
            # Persistiere den Fehlerstatus einmalig
            try:
                self.db.save()
            except Exception:
                pass
            
            self.failed_count += 1
            return False
    
    def process_all_unprocessed(self, custom_periods: Optional[Dict[str, str]] = None,
                               max_experiments: Optional[int] = None) -> Dict[str, Any]:
        """
        Verarbeitet alle unverarbeiteten Churn-Experimente automatisch
        
        Args:
            custom_periods: Optional custom training periods für alle Experimente
            max_experiments: Optional Limit für Anzahl zu verarbeitender Experimente
            
        Returns:
            Dictionary mit Verarbeitungsstatistiken
        """
        self.logger.info("🚀 STARTE AUTOMATISCHE CHURN-EXPERIMENT-VERARBEITUNG")
        self.logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Unverarbeitete Experimente ermitteln
        unprocessed_experiments = self.get_unprocessed_experiments()
        
        if not unprocessed_experiments:
            self.logger.info("✅ Alle Churn-Experimente bereits verarbeitet")
            return {
                'total_experiments': 0,
                'processed_successfully': 0,
                'failed_experiments': 0,
                'duration_seconds': 0.0,
                'all_processed': True
            }
        
        # Limitiere Anzahl falls max_experiments gesetzt
        if max_experiments and len(unprocessed_experiments) > max_experiments:
            self.logger.info(f"🔢 Limitiere auf {max_experiments} Experimente")
            unprocessed_experiments = unprocessed_experiments[:max_experiments]
        
        total_experiments = len(unprocessed_experiments)
        self.logger.info(f"📊 Zu verarbeitende Experimente: {total_experiments}")
        self.logger.info("=" * 70)
        
        # Verarbeite alle Experimente
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0
        
        for i, experiment in enumerate(unprocessed_experiments, 1):
            exp_id = experiment.get("experiment_id")
            
            self.logger.info(f"\n📈 FORTSCHRITT: {i}/{total_experiments} - Experiment {exp_id}")
            self.logger.info("-" * 50)
            
            # Verarbeite Experiment
            success = self.process_experiment(experiment, custom_periods)
            
            if success:
                self.logger.info(f"✅ Experiment {exp_id}: ERFOLGREICH")
            else:
                self.logger.error(f"❌ Experiment {exp_id}: FEHLGESCHLAGEN")
            
            # Zwischenstatus
            success_rate = self.processed_count / i if i > 0 else 0.0
            avg_time = self.total_processing_time / max(self.processed_count, 1)
            remaining = total_experiments - i
            estimated_remaining_time = remaining * avg_time
            
            self.logger.info(f"📊 Status: {self.processed_count}/{i} erfolgreich ({success_rate:.1%})")
            self.logger.info(f"⏱️ Ø Zeit: {avg_time:.1f}s, Geschätzt verbleibend: {estimated_remaining_time:.1f}s")
        
        # Final Statistics
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🎯 AUTOMATISCHE VERARBEITUNG ABGESCHLOSSEN")
        self.logger.info("=" * 70)
        self.logger.info(f"📊 Gesamte Experimente: {total_experiments}")
        self.logger.info(f"✅ Erfolgreich verarbeitet: {self.processed_count}")
        self.logger.info(f"❌ Fehlgeschlagen: {self.failed_count}")
        self.logger.info(f"📈 Erfolgsrate: {self.processed_count/total_experiments:.1%}")
        self.logger.info(f"⏱️ Gesamtlaufzeit: {total_duration:.1f}s")
        self.logger.info(f"⏱️ Ø Zeit pro Experiment: {self.total_processing_time/max(self.processed_count, 1):.1f}s")
        
        return {
            'total_experiments': total_experiments,
            'processed_successfully': self.processed_count,
            'failed_experiments': self.failed_count,
            'success_rate': self.processed_count / total_experiments if total_experiments > 0 else 0.0,
            'duration_seconds': total_duration,
            'avg_processing_time': self.total_processing_time / max(self.processed_count, 1),
            'all_processed': self.failed_count == 0,
            'processing_start': start_time.isoformat(),
            'processing_end': end_time.isoformat()
        }

    def _log_threshold_metrics_from_backtest(self) -> None:
        """Liest die letzte Backtest-JSON und loggt klar gelabelte Metriken je Threshold."""
        models_dir = ProjectPaths.models_directory()
        pattern = "Enhanced_EarlyWarning_Backtest_*.json"
        files = list(models_dir.glob(pattern))
        if not files:
            self.logger.info("   ℹ️ Keine Backtest-JSON gefunden – überspringe Threshold-Metriken")
            return
        latest = max(files, key=lambda p: p.stat().st_mtime)
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # EXKLUSIV: Thresholds aus backtest_results der aktuellen experiment_id
        try:
            exp_id = int(self.current_experiment_id) if hasattr(self, 'current_experiment_id') else None
        except Exception:
            exp_id = None
        bt_rows = []
        if exp_id is not None:
            bt_rows = [
                r for r in self.db.data.get('tables', {}).get('backtest_results', {}).get('records', [])
                if int(r.get('id_experiments') or r.get('experiment_id') or 0) == exp_id
            ]
        if not bt_rows:
            self.logger.info("   ℹ️ Keine backtest_results gefunden – überspringe Threshold-Metriken")
            return
        records = [
            {
                'ACTUAL_CHURN': r.get('actual_churn'),
                'CHURN_PROBABILITY': r.get('churn_probability')
            } for r in bt_rows
        ]
        self.logger.info(f"   🔁 Thresholds aus backtest_results berechnet ({len(records)} Zeilen)")

        # JSON-abhängige Werte nicht verwenden – wir arbeiten ausschließlich mit backtest_results
        opt_threshold = None
        method = 'precision_optimal'

        # Extrahiere Labels und Scores
        y_true = []
        y_prob = []
        for r in records:
            y_true.append(r.get('actual_churn') or r.get('ACTUAL_CHURN') or 0)
            y_prob.append(r.get('churn_probability') or r.get('CHURN_PROBABILITY') or 0.0)

        def eval_at(th: float):
            tp = fp = tn = fn = 0
            for t, p in zip(y_true, y_prob):
                pred = 1 if p >= th else 0
                if pred == 1 and t == 1:
                    tp += 1
                elif pred == 1 and t == 0:
                    fp += 1
                elif pred == 0 and t == 0:
                    tn += 1
                else:
                    fn += 1
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            total = tp + fp + tn + fn
            positives = sum(1 for t in y_true if t == 1)
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'samples': total, 'positives': positives
            }

        # Logge Metriken für optimalen Threshold (falls vorhanden) und persistiere
        # WICHTIG: dieselbe DB-Instanz verwenden (kein Zwischen-Save hier!)
        db = self.db
        dirty = False

        if opt_threshold is not None:
            m = eval_at(float(opt_threshold))
            self.logger.info(
                f"   🔎 Metrics at threshold {float(opt_threshold):.3f} ({method}): "
                f"Precision {m['precision']:.3f}, Recall {m['recall']:.3f}, F1 {m['f1']:.3f}"
            )
            self.logger.info(
                f"   🔢 Confusion Matrix: TP {m['tp']}, FP {m['fp']}, TN {m['tn']}, FN {m['fn']} "
                f"(Samples {m['samples']}, Positives {m['positives']})"
            )
            try:
                exp_id = int(self.current_experiment_id) if hasattr(self, 'current_experiment_id') else None
            except Exception:
                exp_id = None
            if exp_id:
                if db.add_threshold_metrics(exp_id, method or 'precision_optimal', float(opt_threshold), m['precision'], m['recall'], m['f1'], 'backtest', is_selected=1):
                    dirty = True

        # Vergleich bei 0.5
        m05 = eval_at(0.5)
        self.logger.info(
            f"   🧪 Metrics at 0.500: Precision {m05['precision']:.3f}, Recall {m05['recall']:.3f}, F1 {m05['f1']:.3f}"
        )
        self.logger.info(
            f"   🔢 Confusion Matrix @0.500: TP {m05['tp']}, FP {m05['fp']}, TN {m05['tn']}, FN {m05['fn']} "
            f"(Samples {m05['samples']}, Positives {m05['positives']})"
        )
        if 'exp_id' in locals() and exp_id:
            if db.add_threshold_metrics(exp_id, 'standard_0_5', 0.5, m05['precision'], m05['recall'], m05['f1'], 'backtest', is_selected=0):
                dirty = True

        # Zusätzliche Methoden: elbow und f1_optimal
        try:
            import numpy as _np
            from sklearn.metrics import roc_curve, f1_score
            y_true = [r.get('actual_churn') or r.get('ACTUAL_CHURN') or 0 for r in records]
            y_prob = [r.get('churn_probability') or r.get('CHURN_PROBABILITY') or 0.0 for r in records]
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            distances = _np.sqrt((1 - tpr) ** 2 + fpr ** 2)
            elbow_thr = float(thr[_np.argmin(distances)])
            melb = eval_at(elbow_thr)
            self.logger.info(
                f"   📐 Metrics at elbow {elbow_thr:.3f}: Precision {melb['precision']:.3f}, Recall {melb['recall']:.3f}, F1 {melb['f1']:.3f}"
            )
            if 'exp_id' in locals() and exp_id:
                if db.add_threshold_metrics(exp_id, 'elbow', elbow_thr, melb['precision'], melb['recall'], melb['f1'], 'backtest', is_selected=0):
                    dirty = True

            thr_range = _np.arange(0.1, 0.9, 0.01)
            best_f1 = -1.0
            best_thr = 0.5
            for th in thr_range:
                preds = (_np.array(y_prob) >= th).astype(int)
                f1v = f1_score(y_true, preds, zero_division=0)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_thr = float(th)
            mf1 = eval_at(best_thr)
            self.logger.info(
                f"   🧮 Metrics at f1_optimal {best_thr:.3f}: Precision {mf1['precision']:.3f}, Recall {mf1['recall']:.3f}, F1 {mf1['f1']:.3f}"
            )
            if 'exp_id' in locals() and exp_id:
                if db.add_threshold_metrics(exp_id, 'f1_optimal', best_thr, mf1['precision'], mf1['recall'], mf1['f1'], 'backtest', is_selected=0):
                    dirty = True
        except Exception as e:
            self.logger.warning(f"   ⚠️ Zusätzliche Threshold-Metriken (elbow/f1_optimal) nicht berechnet: {e}")
        
        # Persistiere Änderungen einmalig, falls etwas hinzugefügt wurde
        # Speichern erfolgt atomar einmalig im Erfolgszweig von process_experiment()
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Gibt eine Übersicht über den Verarbeitungsstatus aller Churn-Experimente
        
        Returns:
            Dictionary mit Status-Übersicht
        """
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        churn_experiments = [exp for exp in experiments if self._is_churn_experiment(exp.get("model_type", ""))]
        
        status_counts = {}
        for status in EXPERIMENT_STATUS.values():
            status_counts[status] = 0
        
        experiment_details = []
        for exp in churn_experiments:
            exp_id = exp.get("experiment_id")
            status = exp.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Prüfe ob Churn-Daten vorhanden
            has_churn_data = exp_id in self._get_processed_experiment_ids()
            
            experiment_details.append({
                'experiment_id': exp_id,
                'experiment_name': exp.get('experiment_name', 'N/A'),
                'model_type': exp.get('model_type', 'N/A'),
                'status': status,
                'has_churn_data': has_churn_data,
                'training_from': exp.get('training_from', 'N/A'),
                'training_to': exp.get('training_to', 'N/A'),
                'prediction_timebase': exp.get('prediction_timebase', 'N/A')
            })
        
        return {
            'total_churn_experiments': len(churn_experiments),
            'status_counts': status_counts,
            'unprocessed_count': status_counts.get(EXPERIMENT_STATUS['CREATED'], 0),
            'processed_count': status_counts.get(EXPERIMENT_STATUS['PROCESSED'], 0),
            'failed_count': status_counts.get(EXPERIMENT_STATUS['FAILED'], 0),
            'processing_count': status_counts.get(EXPERIMENT_STATUS['PROCESSING'], 0),
            'experiment_details': experiment_details,
            'last_updated': datetime.now().isoformat()
        }
    
    def _update_experiment_status(self, experiment_id: int, status: str) -> None:
        """Aktualisiert den Status eines Experiments"""
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        
        for exp in experiments:
            if exp.get("experiment_id") == experiment_id:
                exp["status"] = status
                exp["last_updated"] = datetime.now().isoformat()
                if status == EXPERIMENT_STATUS['PROCESSED']:
                    exp["processed_at"] = datetime.now().isoformat()
                break
        
        # Persistierung erfolgt gesammelt an anderer Stelle, um Saves zu reduzieren
        self.logger.debug(f"Status für Experiment {experiment_id} auf '{status}' gesetzt (deferred save)")
    
    def _export_outbox_churn(self, experiment_id: int, backtest_path: Path, results: Dict[str, Any]) -> None:
        """Exportiert minimale Artefakte in die Outbox-Struktur für das Experiment."""
        try:
            out_dir = ProjectPaths.outbox_churn_experiment_directory(int(experiment_id))
            ProjectPaths.ensure_directory_exists(out_dir)
        except Exception:
            out_dir = None
        if not out_dir:
            return

        # 1) Backtest JSON in Outbox kopieren (Quelle: models/Enhanced_EarlyWarning_Backtest_*.json)
        try:
            dest_bt = out_dir / Path(backtest_path).name
            shutil.copyfile(str(backtest_path), str(dest_bt))
        except Exception as e:
            self.logger.warning(f"   ⚠️ Konnte Backtest-JSON nicht in Outbox kopieren: {e}")

        # 2) KPIs als schlankes JSON ablegen (aus results/backtest_results)
        try:
            br = (results or {}).get('backtest_results') or {}
            kpis = {
                'experiment_id': int(experiment_id),
                'source': 'churn_auto_processor',
                'metrics': {
                    'auc': br.get('auc'),
                    'precision': br.get('precision'),
                    'recall': br.get('recall'),
                    'f1': br.get('f1_score', br.get('f1'))
                },
                'thresholds': {
                    'optimal': br.get('optimal_threshold'),
                    'elbow': br.get('elbow_threshold'),
                    'f1_optimal': br.get('f1_optimal_threshold'),
                    'standard_0_5': 0.5
                }
            }
            with open(out_dir / 'kpis.json', 'w', encoding='utf-8') as f:
                json.dump(kpis, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"   ⚠️ Konnte KPIs nicht in Outbox schreiben: {e}")

    def reset_failed_experiments(self) -> int:
        """
        Setzt den Status aller fehlgeschlagenen Experimente auf 'created' zurück
        
        Returns:
            Anzahl zurückgesetzter Experimente
        """
        self.logger.info("🔄 Setze fehlgeschlagene Experimente zurück...")
        
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        reset_count = 0
        
        for exp in experiments:
            if (self._is_churn_experiment(exp.get("model_type", "")) and 
                exp.get("status") == EXPERIMENT_STATUS['FAILED']):
                
                exp["status"] = EXPERIMENT_STATUS['CREATED']
                exp["last_updated"] = datetime.now().isoformat()
                reset_count += 1
                
                self.logger.info(f"   🔄 Reset: Experiment {exp.get('experiment_id')}")
        
        if reset_count > 0:
            try:
                self.db.save()
                self.logger.info(f"✅ {reset_count} Experimente zurückgesetzt")
            except Exception as e:
                self.logger.error(f"❌ Fehler beim Speichern: {e}")
                return 0
        else:
            self.logger.info("ℹ️ Keine fehlgeschlagenen Experimente gefunden")
        
        return reset_count


def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(description='Churn Automatic Experiment Processor')
    parser.add_argument('--training-from', type=str, default='202001', help='Default Training Start (YYYYMM)')
    parser.add_argument('--training-to', type=str, default='202312', help='Default Training Ende (YYYYMM)')
    parser.add_argument('--prediction-timebase', type=str, default='202401', help='Default Prediction Zeitpunkt (YYYYMM)')
    parser.add_argument('--status', action='store_true', help='Zeige nur Status-Übersicht an')
    parser.add_argument('--max-experiments', type=int, default=None, help='Maximale Anzahl zu verarbeitender Experimente')
    parser.add_argument('--reset-failed', action='store_true', help='Setze fehlgeschlagene Experimente zurück')
    parser.add_argument('--experiment-id', type=int, default=None, help='Verarbeite nur spezifisches Experiment')
    
    args = parser.parse_args()
    
    # Initialisiere Auto Processor
    processor = ChurnAutoProcessor(
        training_from=args.training_from,
        training_to=args.training_to,
        prediction_timebase=args.prediction_timebase
    )
    
    print("🚀 Churn Automatic Experiment Processor")
    print("=" * 60)
    
    # Status-Übersicht
    if args.status:
        status = processor.get_processing_status()
        
        print(f"📊 CHURN-EXPERIMENTE STATUS:")
        print(f"   🔢 Total: {status['total_churn_experiments']}")
        print(f"   ⏳ Unverarbeitet: {status['unprocessed_count']}")
        print(f"   ✅ Verarbeitet: {status['processed_count']}")
        print(f"   ❌ Fehlgeschlagen: {status['failed_count']}")
        print(f"   🔄 In Bearbeitung: {status['processing_count']}")
        
        if status['experiment_details']:
            print(f"\n📋 EXPERIMENT-DETAILS:")
            for exp in status['experiment_details']:
                status_icon = {
                    'created': '⏳',
                    'processing': '🔄', 
                    'processed': '✅',
                    'failed': '❌'
                }.get(exp['status'], '❓')
                
                print(f"   {status_icon} ID {exp['experiment_id']}: {exp['experiment_name']}")
                print(f"     📊 Type: {exp['model_type']}")
                print(f"     📅 Training: {exp['training_from']}-{exp['training_to']} → {exp['prediction_timebase']}")
                print(f"     💾 Churn Data: {'✅' if exp['has_churn_data'] else '❌'}")
        
        exit(0)
    
    # Reset fehlgeschlagener Experimente
    if args.reset_failed:
        reset_count = processor.reset_failed_experiments()
        print(f"🔄 {reset_count} fehlgeschlagene Experimente zurückgesetzt")
        exit(0)
    
    # Automatische Verarbeitung
    print(f"📅 Training Period: {args.training_from} - {args.training_to}")
    print(f"📅 Prediction Timebase: {args.prediction_timebase}")
    if args.max_experiments:
        print(f"🔢 Max Experiments: {args.max_experiments}")
    print("=" * 60)
    
    custom_periods = {
        'training_from': args.training_from,
        'training_to': args.training_to,
        'prediction_timebase': args.prediction_timebase
    }
    
    results = processor.process_all_unprocessed(
        custom_periods=custom_periods,
        max_experiments=args.max_experiments
    )
    
    print(f"\n🎯 ERGEBNIS:")
    print(f"   📊 Experimente: {results['total_experiments']}")
    print(f"   ✅ Erfolgreich: {results['processed_successfully']}")
    print(f"   ❌ Fehlgeschlagen: {results['failed_experiments']}")
    print(f"   📈 Erfolgsrate: {results['success_rate']:.1%}")
    print(f"   ⏱️ Gesamtlaufzeit: {results['duration_seconds']:.1f}s")
    
    if results['processed_successfully'] > 0:
        print(f"   ⏱️ Ø Zeit pro Experiment: {results['avg_processing_time']:.1f}s")
    
    if results['all_processed']:
        print(f"   🎉 Alle Experimente erfolgreich verarbeitet!")
        exit(0)
    else:
        print(f"   ⚠️ {results['failed_experiments']} Experimente fehlgeschlagen")
        exit(1)


if __name__ == "__main__":
    main()
