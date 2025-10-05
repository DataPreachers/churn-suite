"""
CHURN JSON DATABASE
==================

Vereinheitlichte JSON-Datenbank für alle Customer-Daten im Churn Prediction System.

Features:
- Vereinheitlichte Customer-IDs (Kunde)
- Automatische Konvertierung von customer_id -> Kunde
- JQL-ähnliche Queries
- Strukturierte Datenorganisation
- Einfache Migration von bestehenden JSON-Dateien
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Project imports
from config.paths_config import ProjectPaths
import json as _json


class ChurnJSONDatabase:
    """
    Vereinheitlichte JSON-Datenbank für Churn Prediction System (SINGLETON)
    
    Singleton-Pattern: Es gibt nur eine Instanz pro Prozess für konsistente Daten.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            cls._instance = super(ChurnJSONDatabase, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = None):
        """
        Initialisiert die JSON-Datenbank (nur beim ersten Aufruf)
        
        Args:
            db_path: Pfad zur JSON-Datenbank-Datei
        """
        # Verhindere mehrfache Initialisierung
        if ChurnJSONDatabase._initialized:
            return
            
        if db_path is None:
            db_path = ProjectPaths.dynamic_system_outputs_directory() / "churn_database.json"
        
        self.db_path = Path(db_path)
        # Merke zuletzt geladenes mtime (für Auto-Reload über Prozesslebensdauer)
        try:
            self._last_loaded_mtime: float = self.db_path.stat().st_mtime
        except Exception:
            self._last_loaded_mtime = 0.0
        ChurnJSONDatabase._initialized = True
        self.customer_id_field = "Kunde"  # Standardisiertes Feld
        self.data = self._load_or_create()
        # Erzwinge sofort: Experiments auf 'file_id' normalisieren und 'id_files' entfernen
        try:
            self.migrate_files_to_fk(remove_id_files=True)
        except Exception:
            pass
        # Schema-Bereinigung: 'id_files' aus Experiments-Schema entfernen
        try:
            tables = self.data.setdefault("tables", {})
            exp_tbl = tables.setdefault("experiments", {"schema": {}, "records": []})
            schema = exp_tbl.setdefault("schema", {})
            if isinstance(schema, dict) and "id_files" in schema:
                try:
                    del schema["id_files"]
                except Exception:
                    pass
            schema.setdefault("file_id", {"display_type": "integer", "description": "FK → files.id"})
        except Exception:
            pass
        # Seed: Views aus json-database/Views in die JSON-DB laden (persistente Verwaltung)
        try:
            self._seed_views_from_files()
        except Exception:
            pass

    def maybe_reload(self) -> bool:
        """Lädt die JSON-DB neu, wenn sich die Datei auf Disk geändert hat.

        Returns:
            True, wenn ein Reload stattfand; sonst False.
        """
        try:
            current_mtime = self.db_path.stat().st_mtime
        except Exception:
            current_mtime = 0.0
        if current_mtime and current_mtime > getattr(self, "_last_loaded_mtime", 0.0):
            try:
                import json as _json  # lokale, robuste Re-Importe vermeiden Namenskonflikte
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.data = _json.load(f)
                self._last_loaded_mtime = current_mtime
                print(f"♻️ JSON-DB neu geladen: {self.db_path}")
                return True
            except Exception as _e:
                # Bei Fehlern keinen Reload erzwingen – weiter mit altem Zustand
                return False
        return False
    
    def _load_or_create(self) -> Dict[str, Any]:
        """Lädt bestehende Datenbank oder erstellt neue"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Bestehende Datenbank geladen: {self.db_path}")
                return data
            except Exception as e:
                print(f"⚠️ Fehler beim Laden der Datenbank: {e}")
        
        # Erstelle neue Datenbank
        data = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_customers": 0,
                "data_sources": [],
                "customer_id_mapping": {
                    "standardized_field": "Kunde",
                    "source_mappings": {
                        "stage0_cache": "Kunde",
                        "stage1_outputs": "Kunde", 
                        "backtest_results": "Kunde",
                        "cox_panel": "Kunde",
                        "prioritization": "Kunde"
                    }
                }
            },
            "tables": {
                "rawdata": {
                    "description": "Basis-Customer-Daten aus Stage0 (umbenannt von customers)",
                    "source": "stage0_cache/*.json",
                    "records": []
                },
                "files": {
                    "description": "Datei-Tracking für alle importierten Datenquellen",
                    "source": "various",
                    "records": [
                        {
                            "id": 1,
                            "file_name": "churn_Data_cleaned.csv",
                            "dt_inserted": datetime.now().isoformat(),
                            "source_type": "input_data",
                            "description": "Ursprungs-Input-Datei für alle Experimente"
                        }
                    ]
                },
                "backtest_results": {
                    "description": "Enhanced Early Warning Backtest-Ergebnisse",
                    "source": "models/Enhanced_EarlyWarning_Backtest_*.json",
                    "metadata": {},
                    "schema": {
                        "Kunde": {"display_type": "integer", "description": "Kunden-ID"},
                        "churn_probability": {"display_type": "decimal", "description": "Churn-Wahrscheinlichkeit"},
                        "risk_level": {"display_type": "text", "description": "Risiko-Level"},
                        "actual_churn": {"display_type": "integer", "description": "Tatsächlicher Churn"},
                        "id_experiments": {"display_type": "integer", "description": "Experiment-ID"}
                    },
                    "records": []
                },
                "cox_survival": {
                    "description": "Cox Survival Panel Daten",
                    "source": "cox_survival_data/cox_survival_panel_v4_*.json",
                    "metadata": {},
                    "records": []
                },
                "prioritization": {
                    "description": "Cox-Priorisierung Ergebnisse",
                    "source": "prioritization/prioritization_*_full.json",
                    "metadata": {},
                    "records": []
                },
                "experiments": {
                    "description": "Experiment-Tracking für verschiedene Modell-Tests",
                    "source": "generated",
                    "metadata": {},
                    "schema": {
                        "experiment_id": {"display_type": "integer", "description": "Experiment-ID"},
                        "experiment_name": {"display_type": "text", "description": "Experiment-Name"},
                        "training_period": {"display_type": "json", "description": "Training-Zeitraum"},
                        "backtest_period": {"display_type": "json", "description": "Backtest-Zeitraum"},
                        "model_type": {"display_type": "text", "description": "Modell-Typ"},
                        "file_id": {"display_type": "integer", "description": "FK → files.id"}
                    },
                    "records": []
                },
                "views": {
                    "description": "Logische Views (Name + SELECT) für Management Studio",
                    "source": "managementstudio",
                    "metadata": {},
                    "schema": {
                        "name": {"display_type": "text", "description": "View-Name"},
                        "query": {"display_type": "text", "description": "SELECT-Statement"},
                        "description": {"display_type": "text", "description": "Beschreibung"},
                        "created_at": {"display_type": "datetime", "description": "Erstellt am"},
                        "updated_at": {"display_type": "datetime", "description": "Aktualisiert am"}
                    },
                    "records": []
                },
                "cli": {
                    "description": "Gespeicherte Prozeduren und CLI-Runs (Referenzen auf erzeugte Tabellen)",
                    "source": "sql_query_interface",
                    "metadata": {},
                    "schema": {
                        "run_id": {"display_type": "integer", "description": "Eindeutige Run-ID"},
                        "procedure": {"display_type": "text", "description": "Prozedur-Name (z. B. pivot_case)"},
                        "params": {"display_type": "json", "description": "Parameter als JSON"},
                        "table_name": {"display_type": "text", "description": "Name der erzeugten Tabelle"},
                        "description": {"display_type": "text", "description": "Beschreibung"},
                        "created_at": {"display_type": "datetime", "description": "Erstellt am"}
                    },
                    "records": []
                },
                "experiment_kpis": {
                    "description": "KPI-Metriken für Experimente",
                    "source": "generated",
                    "metadata": {},
                    "schema": {
                        "kpi_id": {"display_type": "integer", "description": "KPI-ID"},
                        "experiment_id": {"display_type": "integer", "description": "Experiment-ID"},
                        "metric_name": {"display_type": "text", "description": "Metrik-Name"},
                        "metric_value": {"display_type": "decimal", "description": "Metrik-Wert"},
                        "metric_type": {"display_type": "text", "description": "Metrik-Typ"},
                        "calculated_at": {"display_type": "datetime", "description": "Berechnungszeitpunkt"}
                    },
                    "records": []
                },
                "customer_churn_details": {
                    "description": "Detaillierte Customer-Churn-Daten mit verschiedenen Schwellwerten und Predictions",
                    "source": "generated",
                    "metadata": {},
                    "schema": {
                        "Kunde": {"display_type": "integer", "description": "Kunden-ID"},
                        "Letzte_Timebase": {"display_type": "integer", "description": "Letzter aktiver Monat"},
                        "I_ALIVE": {"display_type": "integer", "description": "Aktiver Status (1=aktiv, 0=churned)"},
                        "Churn_Wahrscheinlichkeit": {"display_type": "decimal", "description": "Churn-Wahrscheinlichkeit"},
                        "Threshold_Standard_0.5": {"display_type": "decimal", "description": "Standard-Schwellwert 0.5"},
                        "Predicted_Standard_0.5": {"display_type": "integer", "description": "Prediction für Standard 0.5 (1=aktiv, 0=churned)"},
                        "Threshold_Optimal": {"display_type": "decimal", "description": "Optimaler Schwellwert"},
                        "Predicted_Optimal": {"display_type": "integer", "description": "Prediction für Optimal (1=aktiv, 0=churned)"},
                        "Threshold_Elbow": {"display_type": "decimal", "description": "Elbow-Schwellwert"},
                        "Predicted_Elbow": {"display_type": "integer", "description": "Prediction für Elbow (1=aktiv, 0=churned)"},
                        "Threshold_F1_Optimal": {"display_type": "decimal", "description": "F1-optimaler Schwellwert"},
                        "Predicted_F1_Optimal": {"display_type": "integer", "description": "Prediction für F1-Optimal (1=aktiv, 0=churned)"},
                        "Threshold_Precision_First": {"display_type": "decimal", "description": "Precision-First Schwellwert"},
                        "Predicted_Precision_First": {"display_type": "integer", "description": "Prediction für Precision-First (1=aktiv, 0=churned)"},
                        "Threshold_Recall_First": {"display_type": "decimal", "description": "Recall-First Schwellwert"},
                        "Predicted_Recall_First": {"display_type": "integer", "description": "Prediction für Recall-First (1=aktiv, 0=churned)"},
                        "experiment_id": {"display_type": "integer", "description": "Verknüpfung zu Experiment"},
                        "Error": {"display_type": "text", "description": "Fehler-Information (falls vorhanden)"}
                    },
                    "records": []
                },
                "cox_prioritization_results": {
                    "description": "Cox-Priorisierung Ergebnisse (aus CSV konvertiert)",
                    "source": "cox_priorization.py",
                    "metadata": {},
                    "schema": {
                        "Kunde": {"display_type": "integer", "description": "Kunden-ID"},
                        "P_Event_6m": {"display_type": "decimal", "description": "6-Monats-Churn-Wahrscheinlichkeit"},
                        "P_Event_12m": {"display_type": "decimal", "description": "12-Monats-Churn-Wahrscheinlichkeit"},
                        "RMST_12m": {"display_type": "decimal", "description": "Restricted Mean Survival Time 12 Monate"},
                        "RMST_24m": {"display_type": "decimal", "description": "Restricted Mean Survival Time 24 Monate"},
                        "MonthsToLive_Conditional": {"display_type": "decimal", "description": "Konditionelle verbleibende Lebensdauer"},
                        "MonthsToLive_Unconditional": {"display_type": "decimal", "description": "Unkonditionelle verbleibende Lebensdauer"},
                        "PriorityScore": {"display_type": "decimal", "description": "Prioritäts-Score (0-100, höher = risikanter)"},
                        "StartTimebase": {"display_type": "integer", "description": "Start-Zeitpunkt"},
                        "LastAliveTimebase": {"display_type": "integer", "description": "Letzter aktiver Zeitpunkt"},
                        "CutoffExclusive": {"display_type": "integer", "description": "Cutoff-Zeitpunkt (exklusiv)"},
                        "ChurnTimebase": {"display_type": "integer", "description": "Churn-Zeitpunkt (falls bekannt)"},
                        "LeadMonthsToChurn": {"display_type": "decimal", "description": "Monate bis zum Churn"},
                        "Actual_Event_6m": {"display_type": "integer", "description": "Tatsächliches 6-Monats-Event"},
                        "Actual_Event_12m": {"display_type": "integer", "description": "Tatsächliches 12-Monats-Event"},
                        "id_experiments": {"display_type": "integer", "description": "Verknüpfung zu Experiment"}
                    },
                    "records": []
                },
                "cox_analysis_metrics": {
                    "description": "Cox-Analyse Metriken und Performance-Kennzahlen",
                    "source": "cox_priorization.py",
                    "metadata": {},
                    "schema": {
                        "metric_id": {"display_type": "integer", "description": "Metrik-ID"},
                        "experiment_id": {"display_type": "integer", "description": "Experiment-ID"},
                        "metric_name": {"display_type": "text", "description": "Metrik-Name"},
                        "metric_value": {"display_type": "decimal", "description": "Metrik-Wert"},
                        "metric_type": {"display_type": "text", "description": "Metrik-Typ"},
                        "cutoff_exclusive": {"display_type": "integer", "description": "Cutoff-Zeitpunkt"},
                        "feature_count": {"display_type": "integer", "description": "Anzahl verwendeter Features"},
                        "c_index": {"display_type": "decimal", "description": "Concordance Index"},
                        "horizon_max": {"display_type": "decimal", "description": "Maximaler Horizont"},
                        "num_samples": {"display_type": "integer", "description": "Anzahl Samples"},
                        "num_active": {"display_type": "integer", "description": "Anzahl aktiver Kunden"},
                        "mean_p12": {"display_type": "decimal", "description": "Durchschnittliche 12-Monats-Wahrscheinlichkeit"},
                        "runtime_s": {"display_type": "decimal", "description": "Laufzeit in Sekunden"},
                        "calculated_at": {"display_type": "datetime", "description": "Berechnungszeitpunkt"}
                    },
                    "records": []
                },
                "threshold_methods": {
                    "description": "Lookup-Tabelle für Threshold-Methoden",
                    "source": "generated",
                    "metadata": {},
                    "schema": {
                        "method_id": {"display_type": "integer", "description": "Eindeutige Methoden-ID"},
                        "method_name": {"display_type": "text", "description": "Methodenname"},
                        "description": {"display_type": "text", "description": "Beschreibung"}
                    },
                    "records": []
                },
                "churn_threshold_metrics": {
                    "description": "Churn Threshold-Metriken je Methode",
                    "source": "generated",
                    "metadata": {},
                    "schema": {
                        "experiment_id": {"display_type": "integer", "description": "Experiment-ID (FK)"},
                        "method_id": {"display_type": "integer", "description": "Threshold-Methoden-ID (FK)"},
                        "id_files": {"display_type": "list", "description": "Datei-IDs (vom Experiment)"},
                        "threshold_value": {"display_type": "decimal", "description": "Threshold"},
                        "precision": {"display_type": "decimal", "description": "Precision"},
                        "recall": {"display_type": "decimal", "description": "Recall"},
                        "f1": {"display_type": "decimal", "description": "F1-Score"},
                        "data_split": {"display_type": "text", "description": "Split (training/validation/backtest)"},
                        "calculated_at": {"display_type": "datetime", "description": "Zeitstempel"}
                    },
                    "records": []
                }
            }
        }
        
        return data
    
    def _standardize_customer_id(self, record: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        Konvertiert customer_id zu Kunde falls nötig (nicht mehr benötigt)
        
        Args:
            record: Datensatz
            source_type: Typ der Datenquelle
            
        Returns:
            Standardisierter Datensatz
        """
        # Alle Module verwenden jetzt durchgängig 'Kunde'
        return record
    
    def _update_metadata(self, source_type: str, record_count: int):
        """Aktualisiert Metadaten"""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        if source_type not in self.data["metadata"]["data_sources"]:
            self.data["metadata"]["data_sources"].append(source_type)
        
        # Aktualisiere Gesamtanzahl Kunden
        total_customers = len(set(
            record.get("Kunde") for table in self.data["tables"].values() 
            for record in table.get("records", [])
            if record.get("Kunde") is not None
        ))
        self.data["metadata"]["total_customers"] = total_customers
    
    def add_customers_from_stage0(self, stage0_json_path: str) -> bool:
        """
        Fügt Customer-Daten aus Stage0 JSON-Datei hinzu
        
        Args:
            stage0_json_path: Pfad zur Stage0 JSON-Datei
            
        Returns:
            True wenn erfolgreich
        """
        try:
            with open(stage0_json_path, 'r', encoding='utf-8') as f:
                stage0_data = json.load(f)

            # Immer neuen File-Record anlegen (keine Deduplizierung auf files-Ebene)
            stage0_name = Path(stage0_json_path).name
            file_id = self.create_file_record(
                file_name=stage0_name,
                source_type="stage0_cache"
            )

            # Versuche, Records aus Stage0 zu extrahieren
            records_candidate = None
            if isinstance(stage0_data, dict):
                records_candidate = stage0_data.get('records') or stage0_data.get('complete_data')
            elif isinstance(stage0_data, list):
                records_candidate = stage0_data

            # Wenn kein verwertbares Records-Format vorliegt: nur files aktualisieren
            if not isinstance(records_candidate, list):
                self.data["tables"]["rawdata"]["source"] = stage0_json_path
                self._update_metadata("stage0_cache", 0)
                print("⚠️ Stage0-Format unbekannt – files aktualisiert, rawdata unverändert")
                return True

            # Optional: Konvertierung in rawdata-Struktur nur durchführen, wenn Felder passen
            customers = []
            record_id = 1  # Start-ID für rawdata
            convertible = all(isinstance(r, dict) for r in records_candidate)
            if convertible and any('Kunde' in r for r in records_candidate):
                for customer_data in records_candidate:
                    # Erweiterte rawdata-Struktur mit allen Original-Spalten als Top-Level Felder
                    rawdata_record = {
                        "id": record_id,
                        "Kunde": customer_data.get("Kunde", customer_data.get("customer_id", record_id)),  # Korrigiert: "name" → "Kunde"
                        "dt_inserted": datetime.now().isoformat(),
                        "id_files": [file_id],  # Verweis auf diese Stage0-Datei
                        "file_id": int(file_id),
                    }
                    
                    # Alle Original-Spalten als Top-Level Felder hinzufügen (Problem 1 Lösung)
                    for column_name, value in customer_data.items():
                        if column_name not in ["Kunde"]:  # Kunde schon gesetzt
                            rawdata_record[column_name] = value
                    
                    customers.append(rawdata_record)
                    record_id += 1

                # Ersetze bestehende rawdata-Tabelle
                self.data["tables"]["rawdata"]["records"] = customers
                self.data["tables"]["rawdata"]["source"] = stage0_json_path

                self._update_metadata("stage0_cache", len(customers))
                print(f"✅ {len(customers)} Customer-Records zu rawdata hinzugefügt (verweisen auf File-ID {file_id})")
                return True
            else:
                # Kein geeignetes Mapping: Nur files aktualisieren
                self.data["tables"]["rawdata"]["source"] = stage0_json_path
                self._update_metadata("stage0_cache", 0)
                print("ℹ️ Kein direktes Customer-Mapping – files aktualisiert, rawdata belassen")
                return True

        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Stage0-Daten: {e}")
            return False

    def merge_add_customers_from_stage0(self, stage0_json_path: str) -> bool:
        """
        Fügt Customer-Daten aus einer Stage0-JSON kumulativ zu 'rawdata' hinzu
        (ohne bestehende Records zu überschreiben). Dubletten werden anhand
        von 'name' (Kunde) vermieden. Die referenzierte Datei wird in 'files'
        registriert und per 'id_files' an neuen Records verknüpft.
        """
        try:
            with open(stage0_json_path, 'r', encoding='utf-8') as f:
                stage0_data = json.load(f)

            # Immer neuen File-Record anlegen (keine Deduplizierung auf files-Ebene)
            stage0_name = Path(stage0_json_path).name
            file_id = self.create_file_record(
                file_name=stage0_name,
                source_type="stage0_cache"
            )

            # Records extrahieren
            records_candidate = None
            if isinstance(stage0_data, dict):
                records_candidate = stage0_data.get('records') or stage0_data.get('complete_data')
            elif isinstance(stage0_data, list):
                records_candidate = stage0_data

            if not isinstance(records_candidate, list):
                # Nur Files-Tracking aktualisieren
                self.data["tables"]["rawdata"]["source"] = stage0_json_path
                self._update_metadata("stage0_cache", 0)
                print("ℹ️ Stage0-Format unbekannt – nur files registriert (merge)")
                return True

            convertible = all(isinstance(r, dict) for r in records_candidate)
            if not (convertible and any('Kunde' in r for r in records_candidate)):
                # Kein geeignetes Mapping
                self.data["tables"]["rawdata"]["source"] = stage0_json_path
                self._update_metadata("stage0_cache", 0)
                print("ℹ️ Kein direktes Customer-Mapping (merge) – rawdata unverändert")
                return True

            # Bestehende rawdata prüfen
            raw_tbl = self.data["tables"].setdefault("rawdata", {"records": []})
            existing_records = raw_tbl.setdefault("records", [])
            # Deduplizierung auf Ebene (name, timebase) – nicht nur name!
            def _tb_str(v):
                return str(v) if v is not None else ""
            existing_keys = {(str(r.get("name")), _tb_str(r.get("timebase"))) for r in existing_records}
            key_to_record = {(str(r.get("name")), _tb_str(r.get("timebase"))): r for r in existing_records if r.get("name") is not None}
            next_id = max([int(r.get("id", 0)) for r in existing_records] or [0]) + 1

            added = 0
            linked = 0
            for customer_data in records_candidate:
                name_val = customer_data.get("Kunde", customer_data.get("customer_id"))
                if name_val is None:
                    continue
                name_str = str(name_val)
                timebase_val = customer_data.get("I_TIMEBASE") or customer_data.get("timebase")
                timebase_str = _tb_str(timebase_val)
                key = (name_str, timebase_str)
                if key in existing_keys:
                    # Update Lineage: fehlende File-ID ergänzen
                    rec = key_to_record.get(key)
                    if isinstance(rec, dict):
                        ids = rec.setdefault("id_files", []) or []
                        if file_id not in ids:
                            ids.append(file_id)
                            rec["id_files"] = ids
                            linked += 1
                        if rec.get("file_id") is None:
                            rec["file_id"] = int(file_id)
                    continue
                rawdata_record = {
                    "id": next_id,
                    "Kunde": name_str,  # Korrigiert: "name" → "Kunde"
                    "dt_inserted": datetime.now().isoformat(),
                    "id_files": [file_id],
                    "file_id": int(file_id),
                }
                
                # Alle Original-Spalten als Top-Level Felder hinzufügen (Problem 1 Lösung)
                for column_name, value in customer_data.items():
                    if column_name not in ["Kunde"]:  # Kunde schon gesetzt
                        rawdata_record[column_name] = value
                
                existing_records.append(rawdata_record)
                existing_keys.add(key)
                key_to_record[key] = rawdata_record
                next_id += 1
                added += 1

            self.data["tables"]["rawdata"]["records"] = existing_records
            self.data["tables"]["rawdata"]["source"] = stage0_json_path
            self._update_metadata("stage0_cache", added)
            print(f"✅ {added} Customer-Records zu rawdata gemerged, {linked} Lineage-Updates (File-ID {file_id})")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Merge der Stage0-Daten: {e}")
            return False

    def backfill_rawdata_from_experiment_files(self, experiment_id: Optional[int] = None) -> int:
        """
        Lädt alle in experiments.id_files referenzierten Stage0-Dateien kumulativ
        in 'rawdata' nach, ohne bestehende Einträge zu überschreiben.

        Args:
            experiment_id: Optional – nur für ein bestimmtes Experiment, sonst alle.

        Returns:
            Anzahl neu hinzugefügter Customer-Records
        """
        try:
            # Sammle alle File-Records (Stage0) aus referenzierten Experimenten
            exps = self.data.get("tables", {}).get("experiments", {}).get("records", []) or []
            file_ids: set[int] = set()
            for exp in exps:
                if experiment_id is not None and exp.get("experiment_id") != experiment_id:
                    continue
                try:
                    fid = exp.get("file_id")
                    if fid is not None:
                        file_ids.add(int(fid))
                except Exception:
                    continue

            if not file_ids:
                print("ℹ️ Keine referenzierten Stage0-Dateien gefunden (file_id leer)")
                return 0

            files_tbl = self.data.get("tables", {}).get("files", {}).get("records", []) or []
            added_total = 0
            base_dir = ProjectPaths.dynamic_system_outputs_directory() / "stage0_cache"
            for fr in files_tbl:
                try:
                    if int(fr.get("id")) not in file_ids:
                        continue
                except Exception:
                    continue
                if (fr.get("source_type") or "").lower() != "stage0_cache":
                    continue
                fname = fr.get("file_name")
                if not fname:
                    continue
                path = base_dir / str(fname)
                if not path.exists():
                    print(f"⚠️ Datei fehlt: {path}")
                    continue
                before = len(self.data.get("tables", {}).get("rawdata", {}).get("records", []) or [])
                self.merge_add_customers_from_stage0(str(path))
                after = len(self.data.get("tables", {}).get("rawdata", {}).get("records", []) or [])
                added_total += max(after - before, 0)

            print(f"✅ Backfill abgeschlossen – hinzugefügt: {added_total} Records")
            return added_total
        except Exception as e:
            print(f"❌ Fehler beim Backfill aus Experiment-Files: {e}")
            return 0

    # ==============================
    # Rawdata: Materialisierung der Originalspalten
    # ==============================
    def replace_rawdata_with_flattened_features(self) -> int:
        """
        Ersetzt die Tabelle 'rawdata' durch eine flache Darstellung der Originalspalten
        aus dem Feld 'features' jedes Datensatzes. Zusätzliche technische Felder wie
        'id', 'name', 'dt_inserted', 'status', 'timebase' werden entfernt. Falls vorhanden,
        wird 'id_files' zur Lineage erhalten.

        Returns:
            Anzahl der neu geschriebenen Datensätze in 'rawdata'.
        """
        try:
            tables = self.data.setdefault("tables", {})
            raw_tbl = tables.setdefault("rawdata", {"records": []})
            records = raw_tbl.get("records", []) or []

            flattened: list[dict] = []
            for rec in records:
                try:
                    feats = rec.get("features")
                    if not isinstance(feats, dict):
                        # Wenn keine Features vorliegen, überspringen
                        continue
                    new_row = dict(feats)
                    # Ergänze fehlende Kernfelder aus Altstruktur
                    if "Kunde" not in new_row and rec.get("name") is not None:
                        new_row["Kunde"] = rec.get("name")
                    if "I_TIMEBASE" not in new_row and rec.get("timebase") is not None:
                        new_row["I_TIMEBASE"] = rec.get("timebase")
                    if "I_Alive" not in new_row and isinstance(rec.get("status"), dict):
                        if rec["status"].get("I_Alive") is not None:
                            new_row["I_Alive"] = rec["status"].get("I_Alive")
                    # Lineage beibehalten (falls vorhanden)
                    if "id_files" in rec and "id_files" not in new_row:
                        new_row["id_files"] = rec.get("id_files")
                    flattened.append(new_row)
                except Exception:
                    continue

            # Ersetzen der Tabelle 'rawdata'
            raw_tbl["records"] = flattened
            # Schema zurücksetzen – dynamisch aus Records ableitbar
            if isinstance(raw_tbl.get("schema"), dict):
                raw_tbl["schema"] = {}

            # Metadaten aktualisieren
            self._update_metadata("stage0_cache", len(flattened))

            print(f"✅ rawdata durch flache Originalspalten ersetzt – Records: {len(flattened)}")
            return len(flattened)
        except Exception as e:
            print(f"❌ Fehler bei replace_rawdata_with_flattened_features: {e}")
            return 0

    # ==============================
    # Aktivierung: Rawdata exakt aus Experiment-Dateien (ohne Merge)
    # ==============================
    def _extract_stage0_records(self, stage0_json_path: str) -> Optional[List[Dict[str, Any]]]:
        try:
            with open(stage0_json_path, 'r', encoding='utf-8') as f:
                stage0_data = json.load(f)
            if isinstance(stage0_data, dict):
                recs = stage0_data.get('records') or stage0_data.get('complete_data')
            elif isinstance(stage0_data, list):
                recs = stage0_data
            else:
                recs = None
            return recs if isinstance(recs, list) else None
        except Exception:
            return None

    def _transform_stage0_to_raw(self, customer_data: Dict[str, Any], file_id: int, record_id: int) -> Dict[str, Any]:
        rawdata_record = {
            "id": record_id,
            "Kunde": customer_data.get("Kunde", customer_data.get("customer_id", record_id)),  # Korrigiert: "name" → "Kunde"
            "dt_inserted": datetime.now().isoformat(),
            "id_files": [file_id],
        }
        
        # Alle Original-Spalten als Top-Level Felder hinzufügen (Problem 1 Lösung)
        for column_name, value in customer_data.items():
            if column_name not in ["Kunde"]:  # Kunde schon gesetzt
                rawdata_record[column_name] = value
        
        
        return rawdata_record

    def replace_rawdata_for_stage0_files(self, file_ids: List[int]) -> int:
        """
        Erzeugt 'rawdata' ausschließlich aus den angegebenen Stage0-Dateien (ohne Merge/Dedupe).
        Vorhandene rawdata.records werden ersetzt.
        Returns: Anzahl geladener Records
        """
        files_tbl = self.data.get("tables", {}).get("files", {}).get("records", []) or []
        base_dir = ProjectPaths.dynamic_system_outputs_directory() / "stage0_cache"
        all_records: List[Dict[str, Any]] = []
        rid = 1
        for fr in files_tbl:
            try:
                fid = int(fr.get("id"))
            except Exception:
                continue
            if fid not in file_ids:
                continue
            if (fr.get("source_type") or "").lower() != "stage0_cache":
                continue
            fname = fr.get("file_name")
            if not fname:
                continue
            path = base_dir / str(fname)
            recs = self._extract_stage0_records(str(path))
            if not recs:
                continue
            for r in recs:
                if not isinstance(r, dict):
                    continue
                raw = self._transform_stage0_to_raw(r, fid, rid)
                all_records.append(raw)
                rid += 1

        self.data["tables"].setdefault("rawdata", {"records": []})
        self.data["tables"]["rawdata"]["records"] = all_records
        self.data["tables"]["rawdata"]["source"] = ",".join(map(str, file_ids))
        self._update_metadata("stage0_cache", len(all_records))
        print(f"✅ rawdata ersetzt – Records: {len(all_records)} aus Files {file_ids}")
        return len(all_records)

    def activate_rawdata_for_experiment(self, experiment_id: int) -> int:
        """
        Setzt 'rawdata' auf genau die Stage0-Dateien, die im Experiment referenziert sind
        (ohne Merge-Logik). Returns: Anzahl geladener Records.
        """
        exp = self.get_experiment_by_id(experiment_id)
        if not exp:
            print(f"❌ Experiment {experiment_id} nicht gefunden")
            return 0
        fid = exp.get("file_id")
        if fid is None:
            print(f"ℹ️ Experiment {experiment_id} hat kein file_id")
            return 0
        return self.replace_rawdata_for_stage0_files([int(fid)])

    def append_rawdata_from_experiment_files_no_dedupe(self, experiment_id: Optional[int] = None) -> int:
        """
        Hängt ALLE Datensätze aus den in experiments.id_files referenzierten Stage0-Dateien
        an 'rawdata' an – ohne jegliche Deduplizierung. Bestehende Records bleiben erhalten.

        Args:
            experiment_id: Optional – nur für ein bestimmtes Experiment, sonst alle.

        Returns:
            Anzahl neu hinzugefügter Customer-Records
        """
        try:
            exps = self.data.get("tables", {}).get("experiments", {}).get("records", []) or []
            file_ids: set[int] = set()
            for exp in exps:
                if experiment_id is not None and exp.get("experiment_id") != experiment_id:
                    continue
                try:
                    fid = exp.get("file_id")
                    if fid is not None:
                        file_ids.add(int(fid))
                except Exception:
                    continue

            if not file_ids:
                print("ℹ️ Keine referenzierten Stage0-Dateien gefunden (file_id leer)")
                return 0

            files_tbl = self.data.get("tables", {}).get("files", {}).get("records", []) or []
            base_dir = ProjectPaths.dynamic_system_outputs_directory() / "stage0_cache"

            raw_tbl = self.data["tables"].setdefault("rawdata", {"records": []})
            existing_records = raw_tbl.setdefault("records", [])
            next_id = max([int(r.get("id", 0)) for r in existing_records] or [0]) + 1

            added_total = 0

            for fr in files_tbl:
                try:
                    fid = int(fr.get("id"))
                except Exception:
                    continue
                if fid not in file_ids:
                    continue
                if (fr.get("source_type") or "").lower() != "stage0_cache":
                    continue
                fname = fr.get("file_name")
                if not fname:
                    continue
                path = base_dir / str(fname)
                recs = self._extract_stage0_records(str(path))
                if not recs:
                    continue
                for r in recs:
                    if not isinstance(r, dict):
                        continue
                    new_rec = self._transform_stage0_to_raw(r, fid, next_id)
                    existing_records.append(new_rec)
                    next_id += 1
                    added_total += 1

            self.data["tables"]["rawdata"]["records"] = existing_records
            # source informativ: Liste der File-IDs, die zuletzt appended wurden
            self.data["tables"]["rawdata"]["source"] = "+append:" + ",".join(map(str, sorted(file_ids)))
            self._update_metadata("stage0_cache", added_total)
            print(f"✅ rawdata append (no dedupe) – hinzugefügt: {added_total} Records aus Files {sorted(file_ids)}")
            return added_total
        except Exception as e:
            print(f"❌ Fehler beim Append (no dedupe): {e}")
            return 0

    def register_all_stage0_files(self) -> List[int]:
        """
        Legt für alle Dateien in stage0_cache jeweils einen neuen files-Record an
        (ohne Deduplizierung). Gibt die erzeugten File-IDs zurück.
        """
        base_dir = ProjectPaths.dynamic_system_outputs_directory() / "stage0_cache"
        returned: List[int] = []
        created_count: int = 0
        try:
            # Vorhandene files-Records (source_type=stage0_cache) nach file_name indizieren
            files_tbl = self.data.get("tables", {}).get("files", {}).get("records", []) or []
            existing_by_name: Dict[str, int] = {}
            for rec in files_tbl:
                try:
                    if (rec.get("source_type") or "").lower() != "stage0_cache":
                        continue
                    name = rec.get("file_name")
                    if not name:
                        continue
                    existing_by_name[name] = int(rec.get("id"))
                except Exception:
                    continue

            # Für jede Stage0-Datei: vorhandene ID wiederverwenden oder neu anlegen
            for p in sorted(base_dir.glob("*.json")):
                name = p.name
                if name in existing_by_name:
                    returned.append(existing_by_name[name])
                else:
                    fid = self.create_file_record(file_name=name, source_type="stage0_cache")
                    returned.append(fid)
                    created_count += 1

            print(f"✅ {len(returned)} Stage0 files registriert (neu: {created_count}, vorhanden: {len(returned) - created_count})")
        except Exception as e:
            print(f"❌ Fehler bei register_all_stage0_files: {e}")
        return returned

    def rebuild_rawdata_from_all_stage0_files_no_dedupe(self) -> int:
        """
        Baut rawdata aus ALLEN Stage0-Dateien (stage0_cache/*.json) neu auf –
        ohne jegliche Deduplizierung. Vorherige rawdata.records werden ersetzt.
        """
        try:
            file_ids = self.register_all_stage0_files()
            if not file_ids:
                print("ℹ️ Keine Stage0-Dateien gefunden")
                return 0
            return self.replace_rawdata_for_stage0_files(file_ids)
        except Exception as e:
            print(f"❌ Fehler beim vollständigen Neuaufbau von rawdata: {e}")
            return 0
    
    def _extract_experiment_id_from_metadata(self, data: Dict[str, Any]) -> Optional[int]:
        """
        Extrahiert experiment_id aus JSON-Datei Meta-Info
        
        Args:
            data: JSON-Daten
            
        Returns:
            experiment_id oder None
        """
        # Verschiedene Meta-Info-Strukturen prüfen
        if 'experiment_id' in data:
            return data['experiment_id']
        elif 'metadata' in data and 'experiment_id' in data['metadata']:
            return data['metadata']['experiment_id']
        elif 'stage1_metadata' in data and 'experiment_id' in data['stage1_metadata']:
            return data['stage1_metadata']['experiment_id']
        elif 'cox_survival_data' in data and 'metadata' in data['cox_survival_data']:
            return data['cox_survival_data']['metadata'].get('experiment_id')
        else:
            return None

    def add_backtest_results(self, backtest_json_path: str, experiment_id: int = None) -> bool:
        """
        Fügt Backtest-Ergebnisse hinzu
        
        Args:
            backtest_json_path: Pfad zur Backtest JSON-Datei
            experiment_id: ID des zugehörigen Experiments (optional, wird aus Meta-Info extrahiert)
            
        Returns:
            True wenn erfolgreich
        """
        print(f"🔍 DEBUG: add_backtest_results aufgerufen mit experiment_id={experiment_id}, file={backtest_json_path}")
        try:
            with open(backtest_json_path, 'r', encoding='utf-8') as f:
                backtest_data = json.load(f)
            
            # Extrahiere experiment_id aus Meta-Info falls nicht angegeben
            if experiment_id is None:
                experiment_id = self._extract_experiment_id_from_metadata(backtest_data)
                if experiment_id is None:
                    # Fallback: Verwende neuestes Experiment aus JSON-DB statt harter 1
                    try:
                        experiments = self.data.get("tables", {}).get("experiments", {}).get("records", [])
                        if experiments:
                            experiment_id = max(int(e.get("experiment_id")) for e in experiments if e.get("experiment_id") is not None)
                            print(f"⚠️ Keine experiment_id in Meta-Info gefunden, verwende neuestes Experiment: {experiment_id}")
                        else:
                            raise ValueError("Keine Experiments in JSON-DB vorhanden")
                    except Exception:
                        # Explizit fehlschlagen, damit Pipeline korrekt parametriert wird
                        raise RuntimeError("experiment_id fehlt sowohl in Backtest-JSON als auch in JSON-DB – bitte experiment_id explizit übergeben")
                else:
                    print(f"🔍 Experiment-ID aus Meta-Info extrahiert: {experiment_id}")
            
            # Backtest-Daten haben verschiedene Strukturen
            if "results" in backtest_data:
                # Standard-Struktur mit results Liste
                results = []
                for result in backtest_data.get("results", []):
                    result["id_experiments"] = experiment_id
                    results.append(result)
            elif "backtest_results" in backtest_data:
                # Struktur mit backtest_results Metriken
                # Extrahiere echte validation_customers falls vorhanden
                results = []
                
                if "validation_customers" in backtest_data:
                    # Echte Kunden-Daten extrahieren
                    for customer in backtest_data.get("validation_customers", []):
                        result = {
                            "Kunde": customer.get("Kunde"),
                            "churn_probability": customer.get("churn_probability"),
                            "risk_level": customer.get("risk_level"),
                            "actual_churn": customer.get("actual_churn"),
                            "id_experiments": experiment_id
                        }
                        results.append(result)
                else:
                    # Kein validation_customers-Block vorhanden → keine Mock-Daten erzeugen
                    results = []
            else:
                results = []
            
            # Stelle sicher, dass die Tabelle existiert
            if "backtest_results" not in self.data.get("tables", {}):
                self.data.setdefault("tables", {})["backtest_results"] = {
                    "description": "Zeilenweise Backtest-Ergebnisse je Kunde (pro Experiment)",
                    "source": "",
                    "schema": {},
                    "records": []
                }
            
            # Dedupliziere Records basierend auf Kunde + experiment_id
            existing_records = self.data["tables"].get("backtest_results", {}).get("records", [])
            
            # Entferne alle bestehenden Records für diese experiment_id
            filtered_records = [r for r in existing_records if r.get("id_experiments") != experiment_id]
            print(f"🔍 DEBUG: existing={len(existing_records)}, filtered={len(filtered_records)}, neue={len(results)}")
            
            # Füge neue Records hinzu (ersetzt alte Records für diese experiment_id)
            self.data["tables"]["backtest_results"]["records"] = filtered_records + results
            self.data["tables"]["backtest_results"]["source"] = backtest_json_path
            
            self._update_metadata("backtest_results", len(results))
            print(f"✅ {len(results)} Backtest-Results hinzugefügt (Experiment-ID: {experiment_id})")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Backtest-Results: {e}")
            return False
    
    def add_cox_survival(self, cox_json_path: str, experiment_id: int = None) -> bool:
        """
        Fügt Cox Survival Daten hinzu
        
        Args:
            cox_json_path: Pfad zur Cox JSON-Datei
            experiment_id: ID des zugehörigen Experiments (optional, wird aus Meta-Info extrahiert)
            
        Returns:
            True wenn erfolgreich
        """
        try:
            with open(cox_json_path, 'r', encoding='utf-8') as f:
                cox_data = json.load(f)
            
            # Extrahiere experiment_id aus Meta-Info falls nicht angegeben
            if experiment_id is None:
                experiment_id = self._extract_experiment_id_from_metadata(cox_data)
                if experiment_id is None:
                    experiment_id = 2  # Fallback für Cox
                    print(f"⚠️ Keine experiment_id in Meta-Info gefunden, verwende Fallback: {experiment_id}")
                else:
                    print(f"🔍 Experiment-ID aus Meta-Info extrahiert: {experiment_id}")
            
            survival_records = cox_data.get("cox_survival_data", {}).get("survival_records", [])
            for record in survival_records:
                # Erweitere um ID_EXPERIMENTS
                record["id_experiments"] = experiment_id
            
            # Akkumuliere Records statt überschreiben
            existing_records = self.data["tables"]["cox_survival"].get("data", [])
            self.data["tables"]["cox_survival"]["records"] = existing_records + survival_records
            self.data["tables"]["cox_survival"]["source"] = cox_json_path
            
            self._update_metadata("cox_survival", len(survival_records))
            print(f"✅ {len(survival_records)} Cox Survival Records hinzugefügt (Experiment-ID: {experiment_id})")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Cox Survival Daten: {e}")
            return False
    
    def add_prioritization(self, prioritization_json_path: str, experiment_id: int = None) -> bool:
        """
        Fügt Prioritization-Daten hinzu
        
        Args:
            prioritization_json_path: Pfad zur Prioritization JSON-Datei
            experiment_id: ID des zugehörigen Experiments (optional, wird aus Meta-Info extrahiert)
            
        Returns:
            True wenn erfolgreich
        """
        try:
            with open(prioritization_json_path, 'r', encoding='utf-8') as f:
                prioritization_data = json.load(f)
            
            # Extrahiere experiment_id aus Meta-Info falls nicht angegeben
            if experiment_id is None:
                experiment_id = self._extract_experiment_id_from_metadata(prioritization_data)
                if experiment_id is None:
                    experiment_id = 2  # Fallback für Prioritization
                    print(f"⚠️ Keine experiment_id in Meta-Info gefunden, verwende Fallback: {experiment_id}")
                else:
                    print(f"🔍 Experiment-ID aus Meta-Info extrahiert: {experiment_id}")
            
            # Prioritization-Daten haben verschiedene Strukturen
            if "prioritization_records" in prioritization_data:
                prioritization_records = prioritization_data.get("prioritization_records", [])
            elif "data" in prioritization_data:
                prioritization_records = prioritization_data.get("data", [])
            elif "records" in prioritization_data:
                prioritization_records = prioritization_data.get("records", [])
            else:
                # Fallback: versuche direkt Records zu finden
                prioritization_records = prioritization_data.get("records", [])
            
            for record in prioritization_records:
                # Erweitere um ID_EXPERIMENTS
                record["id_experiments"] = experiment_id
            
            # Akkumuliere Records statt überschreiben
            existing_records = self.data["tables"]["prioritization"]["records"]
            self.data["tables"]["prioritization"]["records"] = existing_records + prioritization_records
            self.data["tables"]["prioritization"]["source"] = prioritization_json_path
            
            self._update_metadata("prioritization", len(prioritization_records))
            print(f"✅ {len(prioritization_records)} Prioritization Records hinzugefügt (Experiment-ID: {experiment_id})")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Prioritization-Daten: {e}")
            return False
    
    def import_from_outbox_churn(self, experiment_id: int) -> bool:
        """Importiert Outbox-Artefakte für ein Churn-Experiment (Backtest + KPIs)."""
        try:
            from config.paths_config import ProjectPaths
            out_dir = ProjectPaths.outbox_churn_experiment_directory(int(experiment_id))
            if not out_dir.exists():
                print(f"ℹ️ Outbox-Verzeichnis nicht gefunden: {out_dir}")
                return False

            # 1) Backtest JSON suchen und einlesen
            backtests = sorted(out_dir.glob("Enhanced_EarlyWarning_Backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if backtests:
                self.add_backtest_results(str(backtests[0]), experiment_id=experiment_id)
            else:
                print("ℹ️ Kein Backtest-JSON in Outbox gefunden")

            # 2) KPIs einlesen und in Tabellen mappen
            kpi_file = out_dir / 'kpis.json'
            if kpi_file.exists():
                with open(kpi_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metrics = (data or {}).get('metrics') or {}
                thresholds = (data or {}).get('thresholds') or {}

                # Modell-Metriken persistieren
                self.add_churn_model_metrics(experiment_id, {
                    'auc': metrics.get('auc'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1': metrics.get('f1')
                }, data_split="backtest")

                # Schwellenwerte persistieren (optional vorhanden)
                if thresholds.get('optimal') is not None:
                    self.add_threshold_metrics(experiment_id, 'precision_optimal', float(thresholds['optimal']), 0, 0, 0, 'backtest', is_selected=1)
                if thresholds.get('elbow') is not None:
                    self.add_threshold_metrics(experiment_id, 'elbow', float(thresholds['elbow']), 0, 0, 0, 'backtest', is_selected=0)
                if thresholds.get('f1_optimal') is not None:
                    self.add_threshold_metrics(experiment_id, 'f1_optimal', float(thresholds['f1_optimal']), 0, 0, 0, 'backtest', is_selected=0)
                # Standard 0.5 immer als Referenz speichern
                self.add_threshold_metrics(experiment_id, 'standard_0_5', 0.5, 0, 0, 0, 'backtest', is_selected=0)

            # Speichern übernimmt der aufrufende Flow (atomar)
            return True
        except Exception as e:
            print(f"❌ Fehler beim Outbox-Import (Churn): {e}")
            return False
    
    def import_from_outbox_cox(self, experiment_id: int) -> bool:
        """Importiert Outbox-Artefakte für ein Cox-Experiment (survival/prioritization/metrics/kpis)."""
        try:
            from config.paths_config import ProjectPaths
            out_dir = ProjectPaths.outbox_cox_experiment_directory(int(experiment_id))
            if not out_dir.exists():
                print(f"ℹ️ Cox-Outbox-Verzeichnis nicht gefunden: {out_dir}")
                return False

            # Survival
            surv_file = out_dir / 'cox_survival.json'
            if surv_file.exists():
                surv = json.load(open(surv_file, 'r', encoding='utf-8'))
                # Replace records for this experiment
                tbl = self.data["tables"].setdefault("cox_survival", {"records": []})
                existing = tbl.get("records", [])
                remaining = [r for r in existing if int(r.get('id_experiments', -1)) != int(experiment_id)]
                tbl["records"] = remaining + surv
                self._update_metadata("cox_survival", len(surv))

            # Prioritization
            prio_file = out_dir / 'cox_prioritization.json'
            if prio_file.exists():
                prio = json.load(open(prio_file, 'r', encoding='utf-8'))
                tbl = self.data["tables"].setdefault("cox_prioritization_results", {"records": []})
                existing = tbl.get("records", [])
                remaining = [r for r in existing if int(r.get('id_experiments', -1)) != int(experiment_id)]
                tbl["records"] = remaining + prio
                self._update_metadata("cox_prioritization_results", len(prio))

            # Metrics
            metrics_file = out_dir / 'metrics.json'
            if metrics_file.exists():
                metrics = json.load(open(metrics_file, 'r', encoding='utf-8'))
                tbl = self.data["tables"].setdefault("cox_analysis_metrics", {"records": []})
                existing = tbl.get("records", [])
                remaining = [r for r in existing if int(r.get('experiment_id', -1)) != int(experiment_id)]
                tbl["records"] = remaining + metrics
                self._update_metadata("cox_analysis_metrics", len(metrics))

            # KPIs (optional)
            kpis_file = out_dir / 'kpis.json'
            if kpis_file.exists():
                kpis = json.load(open(kpis_file, 'r', encoding='utf-8'))
                for rec in kpis:
                    if int(rec.get('experiment_id', -1)) == int(experiment_id):
                        self.add_experiment_kpi(rec.get('experiment_id'), rec.get('metric_name'), rec.get('metric_value'), rec.get('metric_type'))

            return True
        except Exception as e:
            print(f"❌ Fehler beim Cox-Outbox-Import: {e}")
            return False
    
    def import_from_outbox_stage0_union(self, replace: bool = True) -> int:
        """
        Liest alle Stage0-JSONs aus der Outbox (outbox/stage0_cache/*.json) und
        materialisiert sie als Union in die Tabelle 'rawdata'.

        - Alle Originalspalten werden als Top-Level-Felder übernommen (Union über Dateien)
        - Jede Zeile enthält eine Lineage-Referenz 'id_files' auf den zugehörigen File-Record
        - Optional: ersetzt bestehende 'rawdata' (replace=True), sonst appendet an bestehende

        Returns:
            Anzahl der neu geschriebenen Records (nur die dieser Operation hinzugefügten)
        """
        try:
            base_dir = ProjectPaths.outbox_directory() / "stage0_cache"
            if not base_dir.exists():
                print(f"ℹ️ Outbox Stage0-Verzeichnis nicht gefunden: {base_dir}")
                return 0

            # files-Index: Eingangsdateien (input_data) nach file_name auflösen
            files_tbl = self.data.get("tables", {}).get("files", {}).get("records", []) or []
            input_by_name: Dict[str, int] = {}
            for rec in files_tbl:
                try:
                    if (rec.get("source_type") or "").lower() != "input_data":
                        continue
                    name = rec.get("file_name")
                    if not name:
                        continue
                    input_by_name[name] = int(rec.get("id"))
                except Exception:
                    continue

            # Ziel-Tabelle vorbereiten
            raw_tbl = self.data["tables"].setdefault("rawdata", {"records": []})
            if replace:
                target_records: List[Dict[str, Any]] = []
                next_id: int = 1
            else:
                target_records = raw_tbl.get("records", []) or []
                next_id = max([int(r.get("id", 0)) for r in target_records] or [0]) + 1

            added_total = 0
            file_ids_used: List[int] = []

            # Alle Outbox-Stage0-Dateien verarbeiten
            for p in sorted(base_dir.glob("*.json")):
                # Ursprungs-CSV-Dateinamen aus Stage0-Datei ermitteln
                try:
                    import json as _json
                    import os as _os
                    with open(p, 'r', encoding='utf-8') as _f:
                        _stage0_head = _json.load(_f)
                    _orig_path = _stage0_head.get("file_path") or ""
                    _orig_name = _os.path.basename(_orig_path) if _orig_path else ""
                except Exception:
                    _orig_name = ""

                # Input-File-ID wiederverwenden oder neu anlegen (nur input_data registrieren)
                if _orig_name and _orig_name in input_by_name:
                    fid = input_by_name[_orig_name]
                else:
                    # Fallback: Wenn keine Zuordnung möglich, registriere einen input_data-Eintrag mit dem ermittelten Namen
                    # oder – falls unbekannt – mit einem generischen Namen aus der Stage0-Datei
                    register_name = _orig_name if _orig_name else p.stem
                    fid = self.create_file_record(file_name=register_name, source_type="input_data")
                    input_by_name[register_name] = fid
                try:
                    recs = self._extract_stage0_records(str(p))
                except Exception:
                    recs = None
                if not recs:
                    continue
                for r in recs:
                    if not isinstance(r, dict):
                        continue
                    new_rec = self._transform_stage0_to_raw(r, fid, next_id)
                    target_records.append(new_rec)
                    next_id += 1
                    added_total += 1
                file_ids_used.append(int(fid))

            # Persistieren
            self.data["tables"].setdefault("rawdata", {"records": []})
            self.data["tables"]["rawdata"]["records"] = target_records
            self.data["tables"]["rawdata"]["source"] = f"outbox_stage0_input:{','.join(map(str, sorted(set(file_ids_used))))}"
            # Metadaten als input_data-Quelle aktualisieren (Stage0 nicht als Quelle zählen)
            self._update_metadata("input_data", added_total)
            print(f"✅ rawdata aus Outbox-Stage0 {'ersetzt' if replace else 'erweitert'} – hinzugefügt: {added_total} Records aus {len(set(file_ids_used))} Files")
            return added_total
        except Exception as e:
            print(f"❌ Fehler beim Outbox-Stage0-Union-Import: {e}")
            return 0

    def _calculate_risk_level(self, churn_probability: float) -> str:
        """Berechnet Risk-Level basierend auf Churn-Wahrscheinlichkeit"""
        if churn_probability >= 0.6:
            return "high"
        elif churn_probability >= 0.3:
            return "medium"
        else:
            return "low"
    
    def query(self, jql_query: str) -> List[Dict[str, Any]]:
        """
        Führt JQL-ähnliche Query aus
        
        Args:
            jql_query: JQL-Query-String
            
        Returns:
            Liste der Ergebnisse
        """
        # Einfache JQL-Implementierung mit WHERE-Unterstützung
        # TODO: Vollständige JQL-Parser implementieren
        
        if "SELECT" in jql_query.upper() and "FROM" in jql_query.upper():
            # Query in Teile aufteilen
            query_upper = jql_query.upper()
            original_query = jql_query
            
            # JOIN-Teil extrahieren (falls vorhanden)
            join_info = self._parse_join_info(original_query, query_upper)
            
            # FROM-Teil extrahieren
            from_index = query_upper.find("FROM")
            from_part = original_query[from_index + 4:].strip()
            
            # SELECT-Teil extrahieren
            select_part = original_query[:from_index].replace("SELECT", "").strip()
            
            # Tabellenname extrahieren (aus FROM-Teil)
            table_name = from_part.split()[0].strip().lower()
            
            # Falls JOIN vorhanden, nur die erste Tabelle verwenden
            if join_info:
                # FROM-Teil bis zum JOIN parsen
                from_parts = from_part.split()
                table_name = from_parts[0].strip().lower()
            
            # WHERE-Teil extrahieren (falls vorhanden)
            where_conditions = []
            if "WHERE" in query_upper:
                where_index = query_upper.find("WHERE")
                # WHERE-Teil bis zum Ende oder bis LIMIT/GROUP BY/ORDER BY
                where_end = len(original_query)
                if "LIMIT" in query_upper:
                    limit_index = query_upper.find("LIMIT")
                    if limit_index > where_index:
                        where_end = limit_index
                if "GROUP BY" in query_upper:
                    group_by_index = query_upper.find("GROUP BY")
                    if group_by_index > where_index and group_by_index < where_end:
                        where_end = group_by_index
                if "ORDER BY" in query_upper:
                    order_by_index = query_upper.find("ORDER BY")
                    if order_by_index > where_index and order_by_index < where_end:
                        where_end = order_by_index
                
                where_part = original_query[where_index + 5:where_end].strip()
                # Einfache WHERE-Bedingungen parsen (nur AND, =)
                where_conditions = self._parse_where_conditions(where_part)
            
            # GROUP BY-Teil extrahieren (falls vorhanden)
            group_by_fields = []
            if "GROUP BY" in query_upper:
                group_by_index = query_upper.find("GROUP BY")
                group_by_part = original_query[group_by_index + 8:].strip()
                # Falls WHERE nach GROUP BY kommt, nur bis WHERE parsen
                if "WHERE" in group_by_part.upper():
                    where_in_group = group_by_part.upper().find("WHERE")
                    group_by_part = group_by_part[:where_in_group].strip()
                group_by_fields = [f.strip() for f in group_by_part.split(",")]
            
            # LIMIT-Teil extrahieren (falls vorhanden)
            limit_count = None
            if "LIMIT" in query_upper:
                limit_index = query_upper.find("LIMIT")
                # LIMIT-Wert extrahieren
                limit_value_part = original_query[limit_index + 5:].strip()
                try:
                    limit_count = int(limit_value_part.split()[0])
                except (ValueError, IndexError):
                    limit_count = None
            
            if table_name in self.data["tables"]:
                # Verwende 'records' als einheitliches Feld
                records = self.data["tables"][table_name].get("records", [])
                
                # JOIN verarbeiten falls vorhanden
                if join_info:
                    records = self._apply_join(records, join_info)
                
                # WHERE-Bedingungen anwenden
                if where_conditions:
                    filtered_records = []
                    for record in records:
                        if self._matches_where_conditions(record, where_conditions):
                            filtered_records.append(record)
                    records = filtered_records
                
                # GROUP BY verarbeiten falls vorhanden
                if group_by_fields:
                    result = self._apply_group_by(records, select_part, group_by_fields)
                else:
                    # Feld-Selektion (ohne GROUP BY)
                    if select_part == "*":
                        result = records
                    elif select_part.strip().upper() == "COUNT(*)":
                        # Spezielle Behandlung für COUNT(*)
                        result = [{"COUNT(*)": len(records)}]
                    else:
                        # Selektionsfelder parsen (unterstützt alias-qualifizierte Spalten und AS-Aliase)
                        def parse_select_fields(select_str):
                            fields_parsed = []  # List[Tuple[field_expr, alias]]
                            raw_fields = [f.strip() for f in select_str.split(",")]
                            for raw in raw_fields:
                                parts_as = raw.split(" as ")
                                if len(parts_as) == 1:
                                    parts_as = raw.split(" AS ")
                                if len(parts_as) == 2:
                                    field_expr = parts_as[0].strip()
                                    out_alias = parts_as[1].strip()
                                else:
                                    field_expr = raw
                                    out_alias = None
                                fields_parsed.append((field_expr, out_alias))
                            return fields_parsed
                        
                        fields = parse_select_fields(select_part)
                        
                        result = []
                        for record in records:
                            filtered_record = {}
                            for field_expr, out_alias in fields:
                                # Alias-Qualifizierer entfernen (b.Kunde -> Kunde)
                                col_name = field_expr.split(".")[-1] if "." in field_expr else field_expr
                                
                                # Verschachtelte Felder werden nicht verwendet; direkte Spalten lesen
                                if col_name in record:
                                    key_name = out_alias or field_expr  # Bevorzugt Alias, sonst Originalausdruck
                                    # Für alias-qualifizierte Namen ohne AS: liefere den kurzen Namen
                                    if out_alias is None and "." in field_expr:
                                        key_name = col_name
                                    filtered_record[key_name] = record[col_name]
                            if filtered_record:  # Nur Records mit gefundenen Feldern hinzufügen
                                result.append(filtered_record)
                
                # LIMIT anwenden (falls vorhanden)
                if limit_count is not None and limit_count > 0:
                    result = result[:limit_count]
                
                return result
        
        return []
    
    def _parse_where_conditions(self, where_part: str) -> List[Dict[str, Any]]:
        """Parst WHERE-Bedingungen"""
        conditions = []
        # Einfache AND-getrennte Bedingungen
        for condition in where_part.split("AND"):
            condition = condition.strip()
            
            # Verschiedene Operatoren unterstützen
            operators = ["=", ">", "<", ">=", "<=", "!="]
            operator = None
            field = None
            value = None
            
            for op in operators:
                if op in condition:
                    parts = condition.split(op, 1)
                    if len(parts) == 2:
                        field = parts[0].strip()
                        value = parts[1].strip()
                        operator = op
                        break
            
            if field and value and operator:
                # Tabellenalias entfernen, falls vorhanden (b.Kunde -> Kunde)
                if "." in field:
                    field = field.split(".")[-1]
                
                # Wert konvertieren (Zahlen, Strings, etc.)
                try:
                    if value.isdigit():
                        value = int(value)
                    elif value.replace(".", "").isdigit():
                        value = float(value)
                    elif value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    else:
                        # String-Wert (Anführungszeichen entfernen)
                        value = value.strip("'\"")
                except:
                    pass
                conditions.append({"field": field, "operator": operator, "value": value})
        
        return conditions
    
    def _apply_group_by(self, records: List[Dict[str, Any]], select_part: str, group_by_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Wendet GROUP BY auf Records an
        
        Args:
            records: Liste der Records
            select_part: SELECT-Teil der Query
            group_by_fields: Felder für GROUP BY
            
        Returns:
            Gruppierte Ergebnisse
        """
        # Gruppiere Records nach GROUP BY Feldern
        groups = {}
        for record in records:
            # Erstelle Group-Key aus GROUP BY Feldern
            group_key = []
            for field in group_by_fields:
                value = record.get(field, None)
                group_key.append(str(value))
            group_key = tuple(group_key)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record)
        
        # Aggregations-Funktionen parsen
        result = []
        for group_key, group_records in groups.items():
            group_result = {}
            
            # GROUP BY Felder hinzufügen
            for i, field in enumerate(group_by_fields):
                group_result[field] = group_records[0].get(field, None)
            
            # Aggregations-Funktionen verarbeiten
            if select_part != "*":
                fields = [f.strip() for f in select_part.split(",")]
                for field in fields:
                    if field not in group_by_fields:  # Nur Aggregations-Felder
                        if "COUNT" in field.upper():
                            # COUNT(*)
                            if "(*)" in field:
                                group_result[field] = len(group_records)
                            else:
                                # COUNT(field)
                                field_name = field[field.find("(")+1:field.find(")")]
                                count = sum(1 for r in group_records if r.get(field_name) is not None)
                                group_result[field] = count
                        elif "SUM" in field.upper():
                            # SUM(field)
                            field_name = field[field.find("(")+1:field.find(")")]
                            try:
                                total = sum(float(r.get(field_name, 0)) for r in group_records)
                                group_result[field] = total
                            except:
                                group_result[field] = 0
                        elif "AVG" in field.upper():
                            # AVG(field)
                            field_name = field[field.find("(")+1:field.find(")")]
                            try:
                                values = [float(r.get(field_name, 0)) for r in group_records]
                                group_result[field] = sum(values) / len(values) if values else 0
                            except:
                                group_result[field] = 0
                        elif "MIN" in field.upper():
                            # MIN(field)
                            field_name = field[field.find("(")+1:field.find(")")]
                            try:
                                values = [float(r.get(field_name, 0)) for r in group_records]
                                group_result[field] = min(values) if values else 0
                            except:
                                group_result[field] = 0
                        elif "MAX" in field.upper():
                            # MAX(field)
                            field_name = field[field.find("(")+1:field.find(")")]
                            try:
                                values = [float(r.get(field_name, 0)) for r in group_records]
                                group_result[field] = max(values) if values else 0
                            except:
                                group_result[field] = 0
                        else:
                            # Einfaches Feld (ersten Wert nehmen)
                            group_result[field] = group_records[0].get(field, None)
            
            result.append(group_result)
        
        return result
    
    def _matches_where_conditions(self, record: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
        """Prüft ob Record WHERE-Bedingungen erfüllt"""
        for condition in conditions:
            field = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]
            
            if field not in record:
                return False
            
            actual_value = record[field]
            
            # Verschiedene Operatoren unterstützen
            if operator == "=":
                if actual_value != expected_value:
                    return False
            elif operator == ">":
                try:
                    # Behandle deutsche Dezimalzahlen (Komma statt Punkt)
                    if isinstance(actual_value, str):
                        actual_value = actual_value.replace(",", ".")
                    if isinstance(expected_value, str):
                        expected_value = expected_value.replace(",", ".")
                    
                    if float(actual_value) <= float(expected_value):
                        return False
                except (ValueError, TypeError):
                    return False
            elif operator == "<":
                try:
                    # Behandle deutsche Dezimalzahlen (Komma statt Punkt)
                    if isinstance(actual_value, str):
                        actual_value = actual_value.replace(",", ".")
                    if isinstance(expected_value, str):
                        expected_value = expected_value.replace(",", ".")
                    
                    if float(actual_value) >= float(expected_value):
                        return False
                except (ValueError, TypeError):
                    return False
            elif operator == ">=":
                try:
                    # Behandle deutsche Dezimalzahlen (Komma statt Punkt)
                    if isinstance(actual_value, str):
                        actual_value = actual_value.replace(",", ".")
                    if isinstance(expected_value, str):
                        expected_value = expected_value.replace(",", ".")
                    
                    if float(actual_value) < float(expected_value):
                        return False
                except (ValueError, TypeError):
                    return False
            elif operator == "<=":
                try:
                    # Behandle deutsche Dezimalzahlen (Komma statt Punkt)
                    if isinstance(actual_value, str):
                        actual_value = actual_value.replace(",", ".")
                    if isinstance(expected_value, str):
                        expected_value = expected_value.replace(",", ".")
                    
                    if float(actual_value) > float(expected_value):
                        return False
                except (ValueError, TypeError):
                    return False
            elif operator == "!=":
                if actual_value == expected_value:
                    return False
        
        return True
    
    def _parse_join_info(self, original_query: str, query_upper: str) -> Optional[Dict[str, Any]]:
        """
        Parst JOIN-Informationen aus der Query
        
        Args:
            original_query: Originale Query
            query_upper: Query in Großbuchstaben
            
        Returns:
            JOIN-Informationen oder None
        """
        # Vereinfachte JOIN-Erkennung
        if "INNER JOIN" not in query_upper and "LEFT JOIN" not in query_upper:
            return None
        
        # JOIN-Teil extrahieren
        join_keyword = "INNER JOIN" if "INNER JOIN" in query_upper else "LEFT JOIN"
        join_start = query_upper.find(join_keyword)
        join_part = original_query[join_start:].strip()
        
        # Einfache Parsing-Logik
        parts = join_part.split()
        if len(parts) < 4:
            return None
        
        # Tabellennamen ist das zweite Wort nach JOIN-Schlüsselwort
        # z.B. [LEFT, JOIN, table, alias, ON, ...] oder [INNER, JOIN, table, alias, ON, ...]
        join_table = parts[2].strip()
        
        # ON-Bedingung finden
        on_index = -1
        for i, part in enumerate(parts):
            if part.upper() == "ON":
                on_index = i
                break
        
        if on_index == -1:
            return None
        
        # ON-Bedingung extrahieren
        on_condition = " ".join(parts[on_index + 1:])
        
        # LIMIT entfernen falls vorhanden
        if "LIMIT" in on_condition.upper():
            limit_index = on_condition.upper().find("LIMIT")
            on_condition = on_condition[:limit_index].strip()
        
        # Einfache Gleichheitsbedingung parsen
        if "=" not in on_condition:
            return None
        
        left_field, right_field = on_condition.split("=", 1)
        left_field = left_field.strip()
        right_field = right_field.strip()
        
        return {
            "join_table": join_table,
            "left_field": left_field,
            "right_field": right_field,
            "join_type": "LEFT" if join_keyword == "LEFT JOIN" else "INNER"
        }
    
    def _apply_join(self, left_records: List[Dict[str, Any]], join_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Wendet JOIN auf Records an
        
        Args:
            left_records: Records der linken Tabelle
            join_info: JOIN-Informationen
            
        Returns:
            Gejointe Records
        """
        join_table = join_info["join_table"]
        left_field = join_info["left_field"]
        right_field = join_info["right_field"]
        join_type = join_info.get("join_type", "INNER")
        
        # Rechte Tabelle laden
        if join_table not in self.data["tables"]:
            return []
        
        right_records = self.data["tables"][join_table].get("records", [])
        
        # JOIN durchführen
        joined_records = []
        
        # Feldnamen ohne Tabellen-Prefix vorbereiten
        left_field_clean = left_field.split('.')[-1] if '.' in left_field else left_field
        right_field_clean = right_field.split('.')[-1] if '.' in right_field else right_field
        
        if join_type == "INNER":
            for left_record in left_records:
                left_value = left_record.get(left_field_clean)
                for right_record in right_records:
                    right_value = right_record.get(right_field_clean)
                    if left_value == right_value:
                        joined_record = left_record.copy()
                        for key, value in right_record.items():
                            if key not in joined_record:
                                joined_record[key] = value
                        joined_records.append(joined_record)
            return joined_records
        else:  # LEFT JOIN
            for left_record in left_records:
                left_value = left_record.get(left_field_clean)
                matched = False
                for right_record in right_records:
                    right_value = right_record.get(right_field_clean)
                    if left_value == right_value:
                        joined_record = left_record.copy()
                        for key, value in right_record.items():
                            if key not in joined_record:
                                joined_record[key] = value
                        joined_records.append(joined_record)
                        matched = True
                if not matched:
                    # Keine rechte Zeile – linke Zeile beibehalten
                    joined_records.append(left_record.copy())
            return joined_records
    
    def get_customer_profile(self, customer_id: int) -> Dict[str, Any]:
        """
        Holt vollständiges Customer-Profil
        
        Args:
            customer_id: Kunden-ID
            
        Returns:
            Vollständiges Customer-Profil
        """
        profile = {"Kunde": customer_id}
        
        # Sammle alle Daten für diesen Kunden
        for table_name, table_data in self.data["tables"].items():
            for record in table_data.get("records", []):
                if record.get("Kunde") == customer_id:
                    profile[table_name] = record
                    break
        
        return profile
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Datenbank-Statistiken zurück"""
        stats = {
            "total_customers": self.data["metadata"]["total_customers"],
            "data_sources": self.data["metadata"]["data_sources"],
            "tables": {}
        }
        
        for table_name, table_data in self.data["tables"].items():
            stats["tables"][table_name] = {
                "record_count": len(table_data.get("records", [])),
                "description": table_data.get("description", ""),
                "source": table_data.get("source", "")
            }
        
        return stats
    
    def create_file_record(self, file_name: str, source_type: str = "unknown") -> int:
        """
        Erstellt einen neuen File-Record
        
        Args:
            file_name: Name der Datei
            source_type: Typ der Datenquelle
            
        Returns:
            File-ID
        """
        # Stelle sicher, dass die Files-Tabelle existiert
        if "files" not in self.data["tables"]:
            self.data["tables"]["files"] = {
                "description": "Datei-Tracking für alle importierten Datenquellen",
                "source": "various",
                "records": []
            }
        
        # Neue File-ID generieren
        existing_ids = [file["id"] for file in self.data["tables"]["files"]["records"]]
        file_id = max(existing_ids) + 1 if existing_ids else 1
        
        file_record = {
            "id": file_id,
            "file_name": file_name,
            "dt_inserted": datetime.now().isoformat(),
            "source_type": source_type
        }
        
        self.data["tables"]["files"]["records"].append(file_record)
        return file_id
    
    def add_file_to_experiment(self, experiment_id: int, file_id: int) -> bool:
        """
        Fügt eine File-ID zu einem Experiment hinzu
        
        Args:
            experiment_id: ID des Experiments
            file_id: ID der Datei
            
        Returns:
            True wenn erfolgreich
        """
        for exp in self.data["tables"]["experiments"]["records"]:
            if exp["experiment_id"] == experiment_id:
                try:
                    exp["file_id"] = int(file_id)
                except Exception:
                    exp["file_id"] = file_id
                # Legacy-Feld sicher entfernen, falls vorhanden
                if "id_files" in exp:
                    try:
                        del exp["id_files"]
                    except Exception:
                        pass
                return True
        return False
    
    def get_experiment_files(self, experiment_id: int) -> List[Dict[str, Any]]:
        """
        Holt alle Dateien für ein Experiment
        
        Args:
            experiment_id: ID des Experiments
            
        Returns:
            Liste der Datei-Records
        """
        files = []
        for exp in self.data["tables"]["experiments"]["records"]:
            if exp["experiment_id"] == experiment_id:
                fid = exp.get("file_id")
                if fid is not None:
                    file_record = self.get_file_by_id(int(fid))
                    if file_record:
                        files.append(file_record)
                break
        return files
    
    def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """
        Holt einen File-Record nach ID
        
        Args:
            file_id: ID der Datei
            
        Returns:
            File-Record oder None
        """
        for file_record in self.data["tables"]["files"]["records"]:
            if file_record["id"] == file_id:
                return file_record
        return None

    # ==============================
    # Views Management (logical views)
    # ==============================
    def list_views(self) -> List[Dict[str, Any]]:
        tbl = self.data.get("tables", {}).get("views", {}).get("records", []) or []
        return list(tbl)

    def add_or_update_view(self, name: str, query: str, description: Optional[str] = None) -> bool:
        name = (name or "").strip()
        query = (query or "").strip()
        if not name or not query:
            return False
        # Nur SELECT/WITH erlauben
        upper = query.strip().upper()
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return False
        # Einfaches Identifier-Pattern
        import re
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return False
        tables = self.data.setdefault("tables", {})
        tbl = tables.setdefault("views", {"records": []}).setdefault("records", [])
        now = datetime.now().isoformat()
        # Update wenn vorhanden
        for rec in tbl:
            if rec.get("name") == name:
                rec["query"] = query
                if description is not None:
                    rec["description"] = description
                rec["updated_at"] = now
                return True
        # Neu anlegen
        tbl.append({
            "name": name,
            "query": query,
            "description": description or "",
            "created_at": now,
            "updated_at": now
        })
        return True

    def delete_view(self, name: str) -> bool:
        name = (name or "").strip()
        tables = self.data.setdefault("tables", {})
        tbl = tables.setdefault("views", {"records": []}).setdefault("records", [])
        new_tbl: List[Dict[str, Any]] = []
        removed = False
        for rec in tbl:
            if rec.get("name") == name:
                removed = True
                continue
            new_tbl.append(rec)
        tables["views"]["records"] = new_tbl
        return removed

    def dedupe_stage0_files(self) -> Dict[str, Any]:
        """
        Konsolidiert Duplikate in der Tabelle 'files' für source_type='stage0_cache' anhand von file_name.
        - Bevorzugt den kleinsten ID als kanonische ID pro file_name.
        - Aktualisiert Referenzen in 'experiments.id_files' und in 'rawdata.records[].id_files'.
        - Entfernt redundante File-Records aus 'files'.
        Returns: Report mit Mapping und Zählwerten.
        """
        report: Dict[str, Any] = {
            "canonical_map": {},  # old_id -> new_id
            "removed_file_ids": [],
            "updated_experiments": 0,
            "updated_raw_records": 0,
        }
        try:
            tables = self.data.setdefault("tables", {})
            files_tbl = tables.setdefault("files", {"records": []}).setdefault("records", [])
            # 1) Gruppiere Stage0-Records nach file_name
            name_to_ids: Dict[str, List[int]] = {}
            for rec in files_tbl:
                try:
                    if (rec.get("source_type") or "").lower() != "stage0_cache":
                        continue
                    name = rec.get("file_name")
                    if not name:
                        continue
                    fid = int(rec.get("id"))
                    name_to_ids.setdefault(name, []).append(fid)
                except Exception:
                    continue

            # 2) Bestimme kanonische IDs und Mapping
            canonical_for_name: Dict[str, int] = {}
            for name, ids in name_to_ids.items():
                if not ids:
                    continue
                canonical = min(ids)
                canonical_for_name[name] = canonical
                for fid in ids:
                    if fid != canonical:
                        report["canonical_map"][fid] = canonical

            if not report["canonical_map"]:
                return report

            # 3) Update experiments.file_id
            exps_tbl = tables.setdefault("experiments", {"records": []}).setdefault("records", [])
            updated_exps = 0
            for exp in exps_tbl:
                try:
                    fid = exp.get("file_id")
                    if fid is None:
                        continue
                    fid_int = int(fid)
                except Exception:
                    continue
                mapped = report["canonical_map"].get(fid_int)
                if mapped is not None and mapped != fid_int:
                    exp["file_id"] = int(mapped)
                    updated_exps += 1
                # Legacy-Feld entfernen, falls noch vorhanden
                if "id_files" in exp:
                    try:
                        del exp["id_files"]
                    except Exception:
                        pass
            report["updated_experiments"] = updated_exps

            # 4) Update rawdata.records[].id_files
            raw_tbl = tables.setdefault("rawdata", {"records": []}).setdefault("records", [])
            updated_raw = 0
            for r in raw_tbl:
                ids_list = r.get("id_files")
                if not isinstance(ids_list, list) or not ids_list:
                    continue
                new_list: List[int] = []
                seen: set = set()
                changed = False
                for fid in ids_list:
                    try:
                        fid_int = int(fid)
                    except Exception:
                        continue
                    mapped = report["canonical_map"].get(fid_int, fid_int)
                    if mapped != fid_int:
                        changed = True
                    if mapped not in seen:
                        new_list.append(mapped)
                        seen.add(mapped)
                if changed:
                    r["id_files"] = new_list
                    updated_raw += 1
            report["updated_raw_records"] = updated_raw

            # 5) Entferne redundante File-Records
            keep_ids: set = set(canonical_for_name.values())
            new_files: List[Dict[str, Any]] = []
            removed: List[int] = []
            for rec in files_tbl:
                try:
                    fid = int(rec.get("id"))
                except Exception:
                    continue
                if (rec.get("source_type") or "").lower() == "stage0_cache":
                    if fid not in keep_ids and fid in report["canonical_map"]:
                        removed.append(fid)
                        continue
                new_files.append(rec)
            tables["files"]["records"] = new_files
            report["removed_file_ids"] = removed

            print(f"✅ Dedupe Stage0 files: removed={len(removed)}, experiments_updated={updated_exps}, raw_updated={updated_raw}")
            return report
        except Exception as e:
            print(f"❌ Fehler bei dedupe_stage0_files: {e}")
            return report

    def create_experiment(self, 
                         experiment_name: str,
                         training_from: str,
                         training_to: str,
                         backtest_from: str,
                         backtest_to: str,
                         model_type: str = "cox",
                         feature_set: str = "standard",
                         hyperparameters: Dict[str, Any] = None,
                         id_file: Optional[int] = None,
                         file_ids: Optional[List[int]] = None) -> int:
        """
        Erstellt ein neues Experiment
        
        Args:
            experiment_name: Name des Experiments
            training_from: Start des Training-Zeitraums (YYYYMM Format - entspricht data_dictionary I_TIMEBASE)
            training_to: Ende des Training-Zeitraums (YYYYMM Format)
            backtest_from: Start des Backtest-Zeitraums (YYYYMM Format)
            backtest_to: Ende des Backtest-Zeitraums (YYYYMM Format)
            model_type: Typ des Modells ('cox', 'binary', 'ensemble')
            feature_set: Feature-Set ('standard', 'enhanced', 'custom')
            hyperparameters: Dictionary mit Hyperparametern
            id_file: Primäre File-ID (FK → files.id). Falls None, wird aus file_ids gewählt oder 1.
            file_ids: (Legacy) Liste von File-IDs – nur zur Migration; es wird die kleinste ID gewählt.
            
        Returns:
            Experiment-ID
        """
        # Stelle sicher, dass die Experiment-Tabellen existieren
        if "experiments" not in self.data["tables"]:
            self.data["tables"]["experiments"] = {
                "description": "Experiment-Tracking für verschiedene Modell-Tests",
                "source": "generated",
                "metadata": {},
                "records": []
            }
        if "experiment_kpis" not in self.data["tables"]:
            self.data["tables"]["experiment_kpis"] = {
                "description": "KPI-Metriken für Experimente",
                "source": "generated",
                "metadata": {},
                "records": []
            }
        
        # Falls keine Hyperparameter übergeben wurden: Pflicht-Snapshot aus algorithm_config_optimized.json
        if not hyperparameters:
            algo_path = ProjectPaths.config_directory() / "algorithm_config_optimized.json"
            if not algo_path.exists():
                raise FileNotFoundError(f"algorithm_config_optimized.json nicht gefunden: {algo_path}")
            with open(algo_path, 'r', encoding='utf-8') as f:
                algo_cfg = _json.load(f)
            hyperparameters = {"algorithm_config": algo_cfg}

        # Neue Experiment-ID generieren
        existing_ids = [exp["experiment_id"] for exp in self.data["tables"]["experiments"]["records"]]
        experiment_id = max(existing_ids) + 1 if existing_ids else 1
        
        # Bestimme primäre File-ID
        if id_file is None:
            primary = self._select_primary_file_id(file_ids)
            id_file = primary if primary is not None else 1
        
        experiment = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "training_from": training_from,
            "training_to": training_to,
            "backtest_from": backtest_from,
            "backtest_to": backtest_to,
            "model_type": model_type,
            "feature_set": feature_set,
            "hyperparameters": hyperparameters or {},
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "file_id": int(id_file)
        }
        
        self.data["tables"]["experiments"]["records"].append(experiment)
        return experiment_id
    
    def get_experiment_by_id(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Holt ein Experiment nach ID
        
        Args:
            experiment_id: ID des Experiments
        
        Returns:
            Experiment-Record oder None
        """
        try:
            for exp in self.data.get("tables", {}).get("experiments", {}).get("records", []):
                if exp.get("experiment_id") == experiment_id:
                    return exp
        except Exception:
            pass
        return None

    def _is_valid_yyyymm(self, value: Any) -> bool:
        """Validiert YYYYMM-Format strikt (YYYY 2000-9999, MM 01-12)."""
        try:
            s = str(value)
            if len(s) != 6 or not s.isdigit():
                return False
            year = int(s[:4])
            month = int(s[4:6])
            if year < 2000 or year > 9999:
                return False
            return 1 <= month <= 12
        except Exception:
            return False

    def update_experiment(self, experiment_id: int, updates: Dict[str, Any]) -> bool:
        """
        Aktualisiert Felder eines Experiments (minimal-invasiv, keine Fallbacks)
        
        Erlaubte Felder:
          - experiment_name (str)
          - training_from (YYYYMM)
          - training_to (YYYYMM)
          - backtest_from (YYYYMM)
          - backtest_to (YYYYMM)
          - model_type (str)
          - feature_set (str)
          - hyperparameters (dict)
          - file_id (int)
        
        Args:
            experiment_id: ID des Experiments
            updates: zu setzende Felder
        
        Returns:
            True bei Erfolg, sonst False
        """
        exp = self.get_experiment_by_id(experiment_id)
        if not exp:
            return False
        if not isinstance(updates, dict) or not updates:
            return False

        # Validierungen vorbereiten
        date_fields = [
            ("training_from", updates.get("training_from")),
            ("training_to", updates.get("training_to")),
            ("backtest_from", updates.get("backtest_from")),
            ("backtest_to", updates.get("backtest_to")),
        ]
        for key, val in date_fields:
            if val is not None and not self._is_valid_yyyymm(val):
                raise ValueError(f"Ungültiges YYYYMM in Feld '{key}': {val}")

        if "hyperparameters" in updates and updates["hyperparameters"] is not None and not isinstance(updates["hyperparameters"], dict):
            raise ValueError("'hyperparameters' muss ein Dictionary sein")

        if "file_id" in updates and updates["file_id"] is not None:
            try:
                updates["file_id"] = int(updates["file_id"])
            except Exception:
                raise ValueError("'file_id' muss ein Integer sein")

        # Nur erlaubte Felder aktualisieren
        allowed_keys = {
            "experiment_name", "training_from", "training_to",
            "backtest_from", "backtest_to", "model_type", "feature_set",
            "hyperparameters", "file_id"
        }
        for k, v in updates.items():
            if k in allowed_keys and v is not None:
                exp[k] = v

        # Legacy-Feld sicher entfernen, falls vorhanden
        if "id_files" in exp:
            try:
                del exp["id_files"]
            except Exception:
                pass

        return True

    def delete_experiment(self, experiment_id: int, cascade: bool = False) -> bool:
        """
        Löscht ein Experiment. Nur mit Cascade (erforderlich).
        
        Entfernt zusätzlich alle referenzierenden Records in anderen Tabellen,
        die über 'id_experiments' oder 'experiment_id' koppeln. 'files' bleibt erhalten.
        """
        if not cascade:
            raise ValueError("Löschen ist nur mit cascade=true zulässig")

        tables = self.data.get("tables", {})
        # 1) Experiments-Record entfernen
        exps = tables.get("experiments", {}).get("records", [])
        before_count = len(exps)
        tables.get("experiments", {})["records"] = [r for r in exps if r.get("experiment_id") != experiment_id]
        removed = before_count != len(tables.get("experiments", {}).get("records", []))

        # 2) Alle abhängigen Tabellen filtern (außer 'files')
        for t_name, t_meta in list(tables.items()):
            if t_name in ("experiments", "files"):
                continue
            recs = t_meta.get("records", []) or []
            if recs:
                t_meta["records"] = [
                    r for r in recs
                    if (r.get("id_experiments") not in (experiment_id,) and r.get("experiment_id") not in (experiment_id,))
                ]

        # Metadaten aktualisieren (Zeitstempel, Kundenanzahl neu berechnen)
        try:
            self._update_metadata("experiments", len(tables.get("experiments", {}).get("records", [])))
        except Exception:
            pass

        return removed
    
    def add_experiment_kpi(self, 
                          experiment_id: int,
                          metric_name: str,
                          metric_value: float,
                          metric_type: str = "backtest") -> bool:
        """
        Fügt eine KPI-Metrik zu einem Experiment hinzu
        
        Args:
            experiment_id: ID des Experiments
            metric_name: Name der Metrik ('auc', 'precision', 'recall', 'f1', 'c_index')
            metric_value: Wert der Metrik
            metric_type: Typ der Metrik ('training', 'validation', 'backtest')
            
        Returns:
            True wenn erfolgreich
        """
        # Stelle sicher, dass die Experiment-Tabellen existieren
        if "experiments" not in self.data["tables"]:
            self.data["tables"]["experiments"] = {
                "description": "Experiment-Tracking für verschiedene Modell-Tests",
                "source": "generated",
                "metadata": {},
                "records": []
            }
        if "experiment_kpis" not in self.data["tables"]:
            self.data["tables"]["experiment_kpis"] = {
                "description": "KPI-Metriken für Experimente",
                "source": "generated",
                "metadata": {},
                "records": []
            }
        
        # Prüfe ob Experiment existiert
        experiment_exists = any(exp["experiment_id"] == experiment_id 
                              for exp in self.data["tables"]["experiments"]["records"])
        
        if not experiment_exists:
            print(f"❌ Experiment mit ID {experiment_id} nicht gefunden")
            return False
        
        # Neue KPI-ID generieren
        existing_kpi_ids = [kpi["kpi_id"] for kpi in self.data["tables"]["experiment_kpis"]["records"]]
        kpi_id = max(existing_kpi_ids) + 1 if existing_kpi_ids else 1
        
        kpi = {
            "kpi_id": kpi_id,
            "experiment_id": experiment_id,
            "metric_name": metric_name,
            "metric_value": round(metric_value, 4),
            "metric_type": metric_type,
            "calculated_at": datetime.now().isoformat()
        }
        
        self.data["tables"]["experiment_kpis"]["records"].append(kpi)
        self._update_metadata("experiment_kpis", len(self.data["tables"]["experiment_kpis"]["records"]))
        
        print(f"✅ KPI '{metric_name}' = {metric_value:.4f} für Experiment {experiment_id} hinzugefügt")
        return True
    
    def add_customer_churn_details_from_backtest(self, backtest_json_path: str, experiment_id: int = None):
        """
        Generiert detaillierte Customer-Daten aus Backtest-JSON und fügt sie zur Datenbank hinzu
        
        Args:
            backtest_json_path: Pfad zur Backtest-JSON-Datei
            experiment_id: ID des Experiments (optional)
        """
        try:
            print(f"🔍 Generiere Customer-Details aus: {backtest_json_path}")
            
            # Lade Backtest-Daten
            with open(backtest_json_path, 'r', encoding='utf-8') as f:
                backtest_data = json.load(f)
            
            # Extrahiere Customer Predictions (verschiedene mögliche Quellen)
            customer_predictions = (
                backtest_data.get('test_period_customers_data')
                or backtest_data.get('all_2020_customers_data', [])
                or backtest_data.get('validation_customers', [])
            )
            
            if not customer_predictions:
                print("❌ Keine Customer Predictions in Backtest-Daten gefunden")
                return False
            
            print(f"📊 Verarbeite {len(customer_predictions)} Customer Predictions...")
            
            # Berechne verschiedene Schwellwerte (verschiedene mögliche Feldnamen)
            y_true = []
            y_pred_proba = []
            
            for pred in customer_predictions:
                # Unterstütze verschiedene Feldnamen
                actual_churn = pred.get('ACTUAL_CHURN') or pred.get('actual_churn', 0)
                churn_prob = pred.get('CHURN_PROBABILITY') or pred.get('churn_probability', 0.0)
                
                y_true.append(actual_churn)
                y_pred_proba.append(churn_prob)
            
            # Optimaler Threshold aus Backtest-Daten
            optimal_threshold = (
                backtest_data.get('optimal_threshold')
                or backtest_data.get('backtest_results', {}).get('optimal_threshold', 0.5)
            )
            
            # Berechne Schwellwerte
            thresholds = self._calculate_multiple_thresholds(y_true, y_pred_proba, optimal_threshold)
            
            # Extrahiere Backtest-Zeiträume
            backtest_period = backtest_data.get('backtest_period', {}) or backtest_data.get('backtest_results', {}).get('backtest_period', {})
            backtest_start = int(backtest_period.get('test_from', '202001'))
            backtest_end = int(backtest_period.get('test_to', '202012'))
            
            # Entfernt: Whitelist/Mapping-Logik. Ab jetzt werden alle gelieferten Feature-Schlüssel übernommen.

            # Erstelle Customer-Details
            customer_details = []
            
            for prediction_data in customer_predictions:
                customer_id = prediction_data.get('Kunde', '')
                if not customer_id:
                    continue
                
                # Unterstütze verschiedene Feldnamen
                churn_prob = prediction_data.get('CHURN_PROBABILITY') or prediction_data.get('churn_probability', 0.0)
                actual_churn = prediction_data.get('ACTUAL_CHURN') or prediction_data.get('actual_churn', 0)
                
                # Bestimme Letzte_Timebase
                if actual_churn == 1:
                    # Regel: letzter YYYYMM, in dem der Kunde in der Datenbasis vorkommt –
                    # bevorzugt der maximale I_TIMEBASE mit I_Alive = 0, sonst maximaler I_TIMEBASE des Kunden.
                    last_timebase = backtest_end
                    try:
                        tables = self.data.get('tables', {})
                        raw = tables.get('rawdata', {}).get('records', []) or []
                        cust_id_int = int(customer_id)
                        def _to_int(v):
                            try:
                                return int(v)
                            except Exception:
                                return None
                        cust_rows = [r for r in raw if _to_int(r.get('Kunde')) == cust_id_int]
                        # bevorzugt Zeilen mit I_Alive == 0 (oder I_ALIVE == 0)
                        zeros = []
                        for r in cust_rows:
                            alive_val = r.get('I_Alive')
                            if alive_val is None:
                                alive_val = r.get('I_ALIVE')
                            is_zero = (alive_val in (0, '0', False))
                            if is_zero:
                                tb = _to_int(r.get('I_TIMEBASE'))
                                if tb is not None:
                                    zeros.append(tb)
                        if zeros:
                            last_timebase = max(zeros)
                        else:
                            all_tb = [_to_int(r.get('I_TIMEBASE')) for r in cust_rows]
                            all_tb = [t for t in all_tb if t is not None]
                            if all_tb:
                                last_timebase = max(all_tb)
                    except Exception:
                        # Bei Fehlern: belasse Backtest-Ende
                        pass
                else:
                    last_timebase = backtest_end
                
                # Erstelle Customer-Detail Record
                customer_detail = {
                    'Kunde': int(customer_id),
                    'Letzte_Timebase': int(last_timebase),
                    'I_ALIVE': int(0 if actual_churn == 1 else 1),  # Explizit Integer: 0=churned, 1=aktiv
                    'Churn_Wahrscheinlichkeit': float(round(churn_prob, 6)),
                    'Threshold_Standard_0.5': float(round(thresholds['standard_0.5'], 6)),
                    'Predicted_Standard_0.5': int(0 if churn_prob > thresholds['standard_0.5'] else 1),  # Explizit Integer
                    'Threshold_Optimal': float(round(thresholds['optimal'], 6)),
                    'Predicted_Optimal': int(0 if churn_prob > thresholds['optimal'] else 1),  # Explizit Integer
                    'Threshold_Elbow': float(round(thresholds['elbow'], 6)),
                    'Predicted_Elbow': int(0 if churn_prob > thresholds['elbow'] else 1),  # Explizit Integer
                    'Threshold_F1_Optimal': float(round(thresholds['f1_optimal'], 6)),
                    'Predicted_F1_Optimal': int(0 if churn_prob > thresholds['f1_optimal'] else 1),  # Explizit Integer
                    'Threshold_Precision_First': float(round(thresholds['precision_first'], 6)),
                    'Predicted_Precision_First': int(0 if churn_prob > thresholds['precision_first'] else 1),  # Explizit Integer
                    'Threshold_Recall_First': float(round(thresholds['recall_first'], 6)),
                    'Predicted_Recall_First': int(0 if churn_prob > thresholds['recall_first'] else 1),  # Explizit Integer
                    'experiment_id': int(experiment_id),
                    'source': 'churn'
                }

                # Falls Backtest-Prediction bereits Feature-Werte mitliefert, übernehme alle Schlüssel
                try:
                    feat_blob = prediction_data.get('engineered_features') or prediction_data.get('features') or {}
                    if isinstance(feat_blob, dict):
                        for fname, fval in feat_blob.items():
                            customer_detail[fname] = fval
                except Exception:
                    pass
                
                customer_details.append(customer_detail)
            
            # Stelle sicher, dass die customer_churn_details Tabelle existiert
            if "customer_churn_details" not in self.data["tables"]:
                self.data["tables"]["customer_churn_details"] = {
                    "description": "Detaillierte Customer-Churn-Daten mit verschiedenen Schwellwerten und Predictions",
                    "source": "generated",
                    "metadata": {},
                    "schema": {
                        "Kunde": {"display_type": "integer", "description": "Kunden-ID"},
                        "Letzte_Timebase": {"display_type": "integer", "description": "Letzter aktiver Monat"},
                        "I_ALIVE": {"display_type": "integer", "description": "Aktiver Status (1=aktiv, 0=churned)"},
                        "Churn_Wahrscheinlichkeit": {"display_type": "decimal", "description": "Churn-Wahrscheinlichkeit"},
                        "Threshold_Standard_0.5": {"display_type": "decimal", "description": "Standard-Schwellwert 0.5"},
                        "Predicted_Standard_0.5": {"display_type": "integer", "description": "Prediction für Standard 0.5 (1=aktiv, 0=churned)"},
                        "Threshold_Optimal": {"display_type": "decimal", "description": "Optimaler Schwellwert"},
                        "Predicted_Optimal": {"display_type": "integer", "description": "Prediction für Optimal (1=aktiv, 0=churned)"},
                        "Threshold_Elbow": {"display_type": "decimal", "description": "Elbow-Schwellwert"},
                        "Predicted_Elbow": {"display_type": "integer", "description": "Prediction für Elbow (1=aktiv, 0=churned)"},
                        "Threshold_F1_Optimal": {"display_type": "decimal", "description": "F1-optimaler Schwellwert"},
                        "Predicted_F1_Optimal": {"display_type": "integer", "description": "Prediction für F1-Optimal (1=aktiv, 0=churned)"},
                        "Threshold_Precision_First": {"display_type": "decimal", "description": "Precision-First Schwellwert"},
                        "Predicted_Precision_First": {"display_type": "integer", "description": "Prediction für Precision-First (1=aktiv, 0=churned)"},
                        "Threshold_Recall_First": {"display_type": "decimal", "description": "Recall-First Schwellwert"},
                        "Predicted_Recall_First": {"display_type": "integer", "description": "Prediction für Recall-First (1=aktiv, 0=churned)"},
                        "experiment_id": {"display_type": "integer", "description": "Verknüpfung zu Experiment"},
                        "Error": {"display_type": "text", "description": "Fehler-Information (falls vorhanden)"}
                    },
                    "records": []
                }
            
            # Füge zur Datenbank hinzu
            existing_records = self.data["tables"]["customer_churn_details"]["records"]
            
            # Entferne nur frühere Churn-Records dieses Experiments (Cox-Records bleiben)
            if experiment_id:
                existing_records = [
                    r for r in existing_records
                    if not (r.get('experiment_id') == experiment_id and (r.get('source') == 'churn'))
                ]
            
            # Füge neue Records hinzu
            existing_records.extend(customer_details)
            self.data["tables"]["customer_churn_details"]["records"] = existing_records
            
            # Update Metadata
            self.data["tables"]["customer_churn_details"]["metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "total_customers": len(customer_details),
                "source_file": backtest_json_path,
                "experiment_id": experiment_id,
                "backtest_period": f"{backtest_start}-{backtest_end}",
                "optimal_threshold": optimal_threshold
            }
            
            print(f"✅ {len(customer_details)} Customer-Details zur Datenbank hinzugefügt")
            print(f"📊 Churn-Rate: {sum(1 for c in customer_details if c['I_ALIVE'] == 0) / len(customer_details) * 100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Generieren der Customer-Details: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_multiple_thresholds(self, y_true, y_pred_proba, optimal_threshold):
        """Berechnet verschiedene Schwellwerte basierend auf y_true und y_pred_proba"""
        try:
            from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
            import numpy as np
            
            thresholds = {}
            
            # Standard Threshold 0.5
            thresholds['standard_0.5'] = 0.5
            
            # Optimal Threshold (aus Backtest-Daten)
            thresholds['optimal'] = optimal_threshold
            
            # Elbow-Methode: Optimaler Punkt in ROC-Kurve
            try:
                fpr, tpr, threshold_candidates = roc_curve(y_true, y_pred_proba)
                # Berechne Distanz zum optimalen Punkt (0,1)
                distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
                elbow_idx = np.argmin(distances)
                thresholds['elbow'] = threshold_candidates[elbow_idx]
            except Exception as e:
                print(f"⚠️ Elbow-Threshold Berechnung fehlgeschlagen: {e}")
                thresholds['elbow'] = 0.5
            
            # F1-Score Optimierung
            try:
                f1_scores = []
                threshold_range = np.arange(0.1, 0.9, 0.01)
                for threshold in threshold_range:
                    y_pred = (np.array(y_pred_proba) > threshold).astype(int)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    f1_scores.append(f1)
                
                best_f1_idx = np.argmax(f1_scores)
                thresholds['f1_optimal'] = threshold_range[best_f1_idx]
            except Exception as e:
                print(f"⚠️ F1-Optimal Threshold Berechnung fehlgeschlagen: {e}")
                thresholds['f1_optimal'] = 0.5
            
            # Precision-First Optimierung (hohe Precision)
            try:
                precision_scores = []
                threshold_range = np.arange(0.1, 0.9, 0.01)
                for threshold in threshold_range:
                    y_pred = (np.array(y_pred_proba) > threshold).astype(int)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    precision_scores.append(precision)
                
                # Wähle Threshold mit Precision > 0.7 (falls verfügbar)
                high_precision_thresholds = [t for t, p in zip(threshold_range, precision_scores) if p > 0.7]
                if high_precision_thresholds:
                    thresholds['precision_first'] = max(high_precision_thresholds)
                else:
                    # Fallback: Höchste Precision
                    best_precision_idx = np.argmax(precision_scores)
                    thresholds['precision_first'] = threshold_range[best_precision_idx]
            except Exception as e:
                print(f"⚠️ Precision-First Threshold Berechnung fehlgeschlagen: {e}")
                thresholds['precision_first'] = 0.5
            
            # Recall-First Optimierung (hoher Recall)
            try:
                recall_scores = []
                threshold_range = np.arange(0.1, 0.9, 0.01)
                for threshold in threshold_range:
                    y_pred = (np.array(y_pred_proba) > threshold).astype(int)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    recall_scores.append(recall)
                
                # Wähle Threshold mit Recall > 0.9 (falls verfügbar)
                high_recall_thresholds = [t for t, r in zip(threshold_range, recall_scores) if r > 0.9]
                if high_recall_thresholds:
                    thresholds['recall_first'] = min(high_recall_thresholds)  # Niedrigster Threshold für hohen Recall
                else:
                    # Fallback: Höchster Recall
                    best_recall_idx = np.argmax(recall_scores)
                    thresholds['recall_first'] = threshold_range[best_recall_idx]
            except Exception as e:
                print(f"⚠️ Recall-First Threshold Berechnung fehlgeschlagen: {e}")
                thresholds['recall_first'] = 0.5
            
            print(f"✅ Schwellwerte berechnet:")
            for name, value in thresholds.items():
                print(f"   {name}: {value:.3f}")
            
            return thresholds
            
        except Exception as e:
            print(f"❌ Fehler bei der Schwellwert-Berechnung: {e}")
            # Fallback-Schwellwerte
            return {
                'standard_0.5': 0.5,
                'optimal': optimal_threshold,
                'elbow': 0.5,
                'f1_optimal': 0.5,
                'precision_first': 0.5,
                'recall_first': 0.5
            }
    
    def get_experiment_kpis(self, experiment_id: int) -> Dict[str, Any]:
        """
        Gibt alle KPIs für ein Experiment zurück
        
        Args:
            experiment_id: ID des Experiments
            
        Returns:
            Dictionary mit KPIs gruppiert nach Typ
        """
        kpis = [kpi for kpi in self.data["tables"]["experiment_kpis"]["records"] 
                if kpi["experiment_id"] == experiment_id]
        
        result = {
            "experiment_id": experiment_id,
            "training": {},
            "validation": {},
            "backtest": {}
        }
        
        for kpi in kpis:
            metric_type = kpi["metric_type"]
            if metric_type in result:
                result[metric_type][kpi["metric_name"]] = kpi["metric_value"]
        
        return result
    
    def compare_experiments(self, experiment_ids: List[int]) -> Dict[str, Any]:
        """
        Vergleicht mehrere Experimente
        
        Args:
            experiment_ids: Liste der Experiment-IDs
            
        Returns:
            Vergleichs-Daten
        """
        comparison = {
            "experiments": [],
            "metrics_comparison": {}
        }
        
        for exp_id in experiment_ids:
            # Experiment-Daten
            experiment = next((exp for exp in self.data["tables"]["experiments"]["records"] 
                             if exp["experiment_id"] == exp_id), None)
            
            if experiment:
                exp_data = {
                    "experiment_id": exp_id,
                    "experiment_name": experiment["experiment_name"],
                    "model_type": experiment["model_type"],
                    "feature_set": experiment["feature_set"],
                    "training_period": f"{experiment['training_from']} - {experiment['training_to']}",
                    "backtest_period": f"{experiment['backtest_from']} - {experiment['backtest_to']}",
                    "status": experiment["status"]
                }
                
                # KPIs
                kpis = self.get_experiment_kpis(exp_id)
                exp_data["kpis"] = kpis
                
                comparison["experiments"].append(exp_data)
                
                # Metriken für Vergleich sammeln
                for metric_type in ["training", "validation", "backtest"]:
                    if metric_type not in comparison["metrics_comparison"]:
                        comparison["metrics_comparison"][metric_type] = {}
                    
                    for metric_name, metric_value in kpis.get(metric_type, {}).items():
                        if metric_name not in comparison["metrics_comparison"][metric_type]:
                            comparison["metrics_comparison"][metric_type][metric_name] = []
                        
                        comparison["metrics_comparison"][metric_type][metric_name].append({
                            "experiment_id": exp_id,
                            "experiment_name": experiment["experiment_name"],
                            "value": metric_value
                        })
        
        return comparison
    
    def add_cox_prioritization_results(self, json_path: str, experiment_id: int = None) -> bool:
        """
        Fügt Cox-Priorisierungsergebnisse zur Datenbank hinzu
        
        Args:
            json_path: Pfad zur JSON-Datei mit Cox-Ergebnissen
            experiment_id: Experiment-ID (optional)
            
        Returns:
            True wenn erfolgreich, False sonst
        """
        try:
            print(f"📊 Lade Cox-Priorisierungsergebnisse: {json_path}")
            
            # JSON-Datei laden
            with open(json_path, 'r', encoding='utf-8') as f:
                cox_data = json.load(f)
            
            # Experiment-ID aus Daten oder Parameter verwenden
            if experiment_id is None:
                experiment_id = cox_data.get('experiment_id')
            
            if experiment_id is None:
                print("❌ Keine Experiment-ID gefunden")
                return False
            
            # Cox-Priorisierungsergebnisse in SQL-Tabellen übertragen
            prioritization_data = cox_data.get('prioritization_data', [])
            
            if not prioritization_data:
                print("❌ Keine Priorisierungsergebnisse gefunden")
                return False
            
            print(f"✅ {len(prioritization_data)} Cox-Priorisierungsergebnisse geladen")
            
            # Cox-Results Tabelle erweitern
            cox_results_table = self.data['tables']['cox_prioritization_results']
            cox_results_data = cox_results_table.get('records', [])
            
            # Neue Cox-Results hinzufügen
            for record in prioritization_data:
                cox_result = {
                    'Kunde': record.get('Kunde'),
                    'cox_score': record.get('PriorityScore', 0.0),
                    'risk_level': self._calculate_risk_level(record.get('PriorityScore', 0.0)),
                    'survival_time': record.get('MonthsToLive_Conditional', 0.0),
                    'event_occurred': record.get('Actual_Event_12m', 0),
                    'experiment_id': experiment_id,
                    'p_event_6m': record.get('P_Event_6m', 0.0),
                    'p_event_12m': record.get('P_Event_12m', 0.0),
                    'rmst_12m': record.get('RMST_12m', 0.0),
                    'rmst_24m': record.get('RMST_24m', 0.0),
                    'months_to_live_unconditional': record.get('MonthsToLive_Unconditional', 0.0),
                    'start_timebase': record.get('StartTimebase'),
                    'last_alive_timebase': record.get('LastAliveTimebase'),
                    'cutoff_exclusive': record.get('CutoffExclusive'),
                    'churn_timebase': record.get('ChurnTimebase'),
                    'lead_months_to_churn': record.get('LeadMonthsToChurn')
                }
                cox_results_data.append(cox_result)
            
            cox_results_table['records'] = cox_results_data
            
            # Cox-Customer-Details Tabelle erweitern
            cox_customer_details_table = self.data['tables']['customer_churn_details']
            cox_customer_details_data = cox_customer_details_table.get('records', [])
            
            # Neue Customer-Details hinzufügen
            for record in prioritization_data:
                customer_detail = {
                    'Kunde': record.get('Kunde'),
                    'experiment_id': experiment_id,
                    'cox_analysis_type': 'enhanced_features',
                    'feature_count': cox_data.get('feature_count', 0),
                    'priority_score': record.get('PriorityScore', 0.0),
                    'risk_category': self._calculate_risk_category(record.get('PriorityScore', 0.0)),
                    'survival_probability_6m': 1.0 - record.get('P_Event_6m', 0.0),
                    'survival_probability_12m': 1.0 - record.get('P_Event_12m', 0.0),
                    'expected_lifetime_months': record.get('MonthsToLive_Conditional', 0.0),
                    'analysis_date': cox_data.get('timestamp', ''),
                    'cutoff_date': record.get('CutoffExclusive')
                }
                cox_customer_details_data.append(customer_detail)
            
            cox_customer_details_table['records'] = cox_customer_details_data
            
            # Datenbank speichern
            self.save()
            
            print(f"✅ Cox-Priorisierungsergebnisse erfolgreich in SQL-Tabellen übertragen")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Cox-Priorisierungsergebnisse: {e}")
            return False
    
    def _calculate_risk_level(self, priority_score: float) -> str:
        """Berechnet Risk-Level basierend auf Priority-Score (C-Index-basiert)"""
        # C-Index 0.758 basierte Schwellwerte (Standardabweichung-basiert)
        # Mean: 0.010549, Std: 0.016003
        if priority_score >= 0.042554:  # +2σ
            return "Sehr Hoch"
        elif priority_score >= 0.026552:  # +1σ
            return "Hoch"
        elif priority_score >= 0.010549:  # Mean
            return "Mittel"
        elif priority_score >= -0.005454:  # -1σ
            return "Niedrig"
        else:
            return "Sehr Niedrig"
    
    def _calculate_risk_category(self, priority_score: float) -> str:
        """Berechnet Risk-Kategorie für Customer-Details"""
        if priority_score >= 0.7:
            return "Kritisch"
        elif priority_score >= 0.5:
            return "Hoch"
        elif priority_score >= 0.3:
            return "Mittel"
        else:
            return "Niedrig"
    
    def add_cox_analysis_metrics(self, json_path: str, experiment_id: int = None) -> bool:
        """
        Fügt Cox-Analyse-Metriken aus JSON-Metadaten-Datei hinzu
        
        Args:
            json_path: Pfad zur Prioritization-JSON-Metadaten-Datei
            experiment_id: ID des zugehörigen Experiments (optional)
            
        Returns:
            True wenn erfolgreich
        """
        try:
            print(f"📊 Lade Cox-Analyse-Metriken: {json_path}")
            
            # JSON-Metadaten laden
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Experiment-ID ermitteln falls nicht angegeben
            if experiment_id is None:
                cutoff_exclusive = metadata.get('cutoff_exclusive')
                if cutoff_exclusive:
                    experiment_id = self._find_cox_experiment_id(cutoff_exclusive)
            
            # Metriken extrahieren
            metrics = [
                ("c_index", metadata.get("c_index"), "performance"),
                ("horizon_max", metadata.get("horizon_max"), "performance"),
                ("num_samples", metadata.get("num_samples"), "data"),
                ("num_active", metadata.get("num_active"), "data"),
                ("feature_count", metadata.get("feature_count"), "data"),
                ("mean_p12", metadata.get("mean_p12"), "prediction"),
                ("runtime_s", metadata.get("runtime_s"), "performance")
            ]
            
            # Records erstellen
            records = []
            metric_id = 1
            for metric_name, metric_value, metric_type in metrics:
                if metric_value is not None:
                    record = {
                        "metric_id": metric_id,
                        "experiment_id": experiment_id,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "metric_type": metric_type,
                        "cutoff_exclusive": metadata.get("cutoff_exclusive"),
                        "feature_count": metadata.get("feature_count"),
                        "c_index": metadata.get("c_index"),
                        "horizon_max": metadata.get("horizon_max"),
                        "num_samples": metadata.get("num_samples"),
                        "num_active": metadata.get("num_active"),
                        "mean_p12": metadata.get("mean_p12"),
                        "runtime_s": metadata.get("runtime_s"),
                        "calculated_at": metadata.get("timestamp", datetime.now().isoformat())
                    }
                    records.append(record)
                    metric_id += 1
            
            # Records zur Tabelle hinzufügen
            if "cox_analysis_metrics" not in self.data["tables"]:
                self.data["tables"]["cox_analysis_metrics"] = {
                    "description": "Cox-Analyse Metriken und Performance-Kennzahlen",
                    "source": "cox_priorization.py",
                    "metadata": {},
                    "schema": {},
                    "records": []
                }
            
            self.data["tables"]["cox_analysis_metrics"]["records"].extend(records)
            
            # Metadaten aktualisieren
            self._update_metadata("cox_analysis_metrics", len(records))
            
            print(f"✅ {len(records)} Cox-Analyse-Metriken zur Datenbank hinzugefügt")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen der Cox-Analyse-Metriken: {e}")
            return False
    
    def create_cox_experiment(self, churn_experiment_id: int, cutoff_exclusive: int) -> int:
        """
        Erstellt ein Cox-Experiment basierend auf einem Churn-Experiment
        
        Args:
            churn_experiment_id: ID des zugehörigen Churn-Experiments
            cutoff_exclusive: Cutoff-Zeitpunkt für Cox-Analyse
            
        Returns:
            ID des erstellten Cox-Experiments
        """
        try:
            # Churn-Experiment finden
            churn_experiment = None
            for exp in self.data["tables"]["experiments"]["records"]:
                if exp["experiment_id"] == churn_experiment_id:
                    churn_experiment = exp
                    break
            
            if not churn_experiment:
                raise ValueError(f"Churn-Experiment mit ID {churn_experiment_id} nicht gefunden")
            
            # Zeitangaben aus Churn-Experiment extrahieren
            training_from = churn_experiment.get("training_from")
            training_to = churn_experiment.get("training_to")
            backtest_from = churn_experiment.get("backtest_from")
            backtest_to = churn_experiment.get("backtest_to")
            
            if not all([training_from, training_to, backtest_from, backtest_to]):
                raise ValueError(f"Churn-Experiment {churn_experiment_id} hat unvollständige Zeitangaben")
            
            # Cox-Experiment erstellen
            cox_experiment_id = self.create_experiment(
                experiment_name=f"Cox Regression Analysis (Cutoff {cutoff_exclusive})",
                training_from=training_from,
                training_to=training_to,
                backtest_from=backtest_from,
                backtest_to=backtest_to,
                model_type="cox_regression",
                feature_set="enhanced_features",
                hyperparameters={
                    "cutoff_exclusive": cutoff_exclusive,
                    "horizons": [6, 12],
                    "rmst_horizons": [12, 24],
                    "weights": [0.6, 0.3, 0.1],
                    "churn_experiment_id": churn_experiment_id
                }
            )
            
            print(f"✅ Cox-Experiment erstellt: ID {cox_experiment_id}")
            return cox_experiment_id
            
        except Exception as e:
            print(f"❌ Fehler beim Erstellen des Cox-Experiments: {e}")
            return None
    
    def _find_cox_experiment_id(self, cutoff_exclusive: int) -> int:
        """
        Findet Cox-Experiment-ID basierend auf Cutoff-Zeitpunkt
        
        Args:
            cutoff_exclusive: Cutoff-Zeitpunkt
            
        Returns:
            Experiment-ID oder None
        """
        for exp in self.data["tables"]["experiments"]["records"]:
            if (exp["model_type"] == "cox_regression" and 
                exp["hyperparameters"].get("cutoff_exclusive") == cutoff_exclusive):
                return exp["experiment_id"]
        return None
    
    # ==============================
    # Artefakte-Registry (Modelle/Backtests/Explain)
    # ==============================
    def ensure_artifacts_registry(self) -> None:
        """Stellt sicher, dass die Tabelle artifacts_registry existiert."""
        tables = self.data.setdefault("tables", {})
        if "artifacts_registry" not in tables:
            tables["artifacts_registry"] = {
                "description": "Registry für Modell-/Backtest-/Explain-Artefakte",
                "source": "managementstudio",
                "metadata": {},
                "schema": {
                    "artifact_id": {"display_type": "integer", "description": "laufende ID"},
                    "category": {"display_type": "text", "description": "model|backtest|explain"},
                    "experiment_id": {"display_type": "integer", "description": "Experiment-ID (optional)"},
                    "file_path": {"display_type": "text", "description": "Pfad auf Dateisystem"},
                    "size": {"display_type": "integer", "description": "Bytes"},
                    "status": {"display_type": "text", "description": "active|obsolete|deleted"},
                    "created_at": {"display_type": "datetime", "description": "Dateizeitpunkt"},
                    "registered_at": {"display_type": "datetime", "description": "Registry-Zeitpunkt"}
                },
                "records": []
            }
    
    def add_artifact_record(self,
                             category: str,
                             experiment_id: Optional[int],
                             file_path: str,
                             status: str = "active",
                             size: Optional[int] = None,
                             created_at: Optional[str] = None) -> int:
        """Fügt einen neuen Artefakt-Eintrag hinzu und gibt artifact_id zurück."""
        from datetime import datetime as _dt
        self.ensure_artifacts_registry()
        tbl = self.data["tables"]["artifacts_registry"]
        recs = tbl.get("records", []) or []
        next_id = 1 + max([int(r.get("artifact_id", 0)) for r in recs] or [0])
        try:
            sz = size
            if sz is None:
                try:
                    import os as _os
                    sz = int(_os.path.getsize(file_path)) if file_path else None
                except Exception:
                    sz = None
        except Exception:
            sz = None
        rec = {
            "artifact_id": next_id,
            "category": str(category or ""),
            "experiment_id": int(experiment_id) if experiment_id is not None else None,
            "file_path": str(file_path or ""),
            "size": sz,
            "status": str(status or "active"),
            "created_at": created_at or _dt.now().isoformat(),
            "registered_at": _dt.now().isoformat()
        }
        recs.append(rec)
        tbl["records"] = recs
        return next_id
    
    def update_artifact_status(self, file_path: str, status: str) -> bool:
        """Aktualisiert den Status eines Artefakts, falls vorhanden."""
        self.ensure_artifacts_registry()
        tbl = self.data["tables"]["artifacts_registry"]
        recs = tbl.get("records", []) or []
        updated = False
        for r in recs:
            if str(r.get("file_path")) == str(file_path):
                r["status"] = str(status)
                updated = True
        tbl["records"] = recs
        return updated

    # ==============================
    # Schema-Migration: id_files (LIST) -> file_id (FK)
    # ==============================
    def _select_primary_file_id(self, id_files_value) -> Optional[int]:
        """Wählt deterministisch ein primäres file_id aus einer Liste/Skalar.
        Regel: kleinstes Integer aus der Liste; bei Skalar -> int(Skalar); bei Fehler -> None."""
        try:
            import collections.abc as _abc
            if id_files_value is None:
                return None
            # DuckDB/JSON kann LIST sein; akzeptiere list/tuple
            if isinstance(id_files_value, (list, tuple)) or (hasattr(id_files_value, '__iter__') and not isinstance(id_files_value, (str, bytes))):
                vals = []
                for x in id_files_value:
                    try:
                        vals.append(int(x))
                    except Exception:
                        continue
                return min(vals) if vals else None
            # Skalar
            try:
                return int(id_files_value)
            except Exception:
                return None
        except Exception:
            return None

    def migrate_files_to_fk(self, remove_id_files: bool = False) -> dict:
        """Migriert Experiments/Rawdata: ergänzt 'file_id' aus bisherigen 'id_files'.
        Args:
            remove_id_files: Wenn True, entfernt 'id_files' nach Migration aus Records.
        Returns: Zählwerte für aktualisierte Datensätze.
        """
        tables = self.data.setdefault("tables", {})
        updated_experiments = 0
        updated_raw = 0

        # Experiments
        exp_tbl = tables.setdefault("experiments", {"records": []})
        exp_recs = exp_tbl.setdefault("records", [])
        for rec in exp_recs:
            file_id = rec.get("file_id")
            if file_id is None:
                primary = self._select_primary_file_id(rec.get("id_files"))
                if primary is not None:
                    rec["file_id"] = int(primary)
                    updated_experiments += 1
            if remove_id_files and "id_files" in rec:
                try:
                    del rec["id_files"]
                except Exception:
                    pass

        # Rawdata
        raw_tbl = tables.setdefault("rawdata", {"records": []})
        raw_recs = raw_tbl.setdefault("records", [])
        for rec in raw_recs:
            file_id = rec.get("file_id")
            if file_id is None:
                primary = self._select_primary_file_id(rec.get("id_files"))
                if primary is not None:
                    rec["file_id"] = int(primary)
                    updated_raw += 1
            if remove_id_files and "id_files" in rec:
                try:
                    del rec["id_files"]
                except Exception:
                    pass

        # Optional: Schema-Hinweise ergänzen (nur Anzeigezwecke)
        try:
            # Experiments schema
            exp_schema = exp_tbl.setdefault("schema", {})
            exp_schema.setdefault("file_id", {"display_type": "integer", "description": "FK → files.id"})
            # Rawdata schema
            raw_schema = raw_tbl.setdefault("schema", {})
            raw_schema.setdefault("file_id", {"display_type": "integer", "description": "FK → files.id"})
        except Exception:
            pass

        return {"experiments_updated": updated_experiments, "rawdata_updated": updated_raw}

    def link_stage0_parent_and_normalize_rawdata(self) -> dict:
        """Verknüpft Stage0-Dateien (files.source_type='stage0_cache') mit ihrem Ursprungs-CSV (parent_file_id)
        und setzt rawdata.file_id auf das CSV file_id (statt Stage0 file_id).
        Mapping erfolgt über den in der Stage0-JSON enthaltenen 'file_path' (Basename).
        Returns: Zählwerte (linked_stage0, rawdata_relinked).
        """
        from pathlib import Path as _Path
        import json as _json
        base_dir = ProjectPaths.dynamic_system_outputs_directory() / "stage0_cache"
        tables = self.data.setdefault("tables", {})
        files_tbl = tables.setdefault("files", {"records": []})
        files_recs = files_tbl.setdefault("records", [])
        # Indexe aufbauen
        by_id = {}
        by_name_input = {}
        for fr in files_recs:
            try:
                fid = int(fr.get("id")) if fr.get("id") is not None else None
            except Exception:
                fid = None
            if fid is not None:
                by_id[fid] = fr
            name = fr.get("file_name")
            st = (fr.get("source_type") or "").lower()
            if name and st == "input_data":
                by_name_input[str(name)] = fid
        linked = 0
        # Stage0 → CSV Parent setzen
        for fr in files_recs:
            st = (fr.get("source_type") or "").lower()
            if st != "stage0_cache":
                continue
            fname = fr.get("file_name")
            if not fname:
                continue
            p = base_dir / str(fname)
            parent_id = None
            try:
                if p.exists():
                    data = _json.loads(p.read_text(encoding="utf-8"))
                    file_path = data.get("file_path") or data.get("input_file") or data.get("original_file_path")
                    if file_path:
                        from os.path import basename as _basename
                        bname = _basename(str(file_path))
                        parent_id = by_name_input.get(bname)
            except Exception:
                parent_id = None
            if parent_id is not None:
                if fr.get("parent_file_id") != parent_id:
                    fr["parent_file_id"] = int(parent_id)
                    linked += 1
        # rawdata.file_id auf CSV parent drehen
        raw_tbl = tables.setdefault("rawdata", {"records": []})
        raw_recs = raw_tbl.setdefault("records", [])
        # Map: stage0 file_id -> parent csv file_id
        stg_to_csv = {}
        for fr in files_recs:
            try:
                fid = int(fr.get("id")) if fr.get("id") is not None else None
            except Exception:
                fid = None
            if fid is None:
                continue
            st = (fr.get("source_type") or "").lower()
            if st == "stage0_cache" and fr.get("parent_file_id") is not None:
                stg_to_csv[fid] = int(fr.get("parent_file_id"))
        relinked = 0
        for r in raw_recs:
            try:
                fid = int(r.get("file_id")) if r.get("file_id") is not None else None
            except Exception:
                fid = None
            if fid is None:
                continue
            csv_parent = stg_to_csv.get(fid)
            if csv_parent is not None and csv_parent != fid:
                r["file_id"] = int(csv_parent)
                relinked += 1
        # Schemahinweis: parent_file_id
        files_tbl.setdefault("schema", {}).setdefault("parent_file_id", {"display_type": "integer", "description": "CSV Parent (files.id)"})
        return {"linked_stage0": linked, "rawdata_relinked": relinked}
    
    def save(self) -> bool:
        """
        Speichert die Datenbank sicher (atomar, mit Lock & optionalem Snapshot).
        """
        try:
            self.migrate_files_to_fk(remove_id_files=True)
        except Exception:
            pass
        return self.safe_save(create_snapshot=False, max_snapshots=10)

    def _seed_views_from_files(self) -> None:
        """Lädt SQL-Views aus dem Verzeichnis json-database/Views in die Tabelle 'views'.
        - Erstellt die Tabelle 'views' bei Bedarf mit Schema/Metadaten
        - Aktualisiert vorhandene Einträge (gleicher Name) oder fügt neue hinzu
        - Speichert nicht automatisch; Aufrufer kann optional save() ausführen
        """
        try:
            base_dir = ProjectPaths.json_database_directory() / "Views"
            if not base_dir.exists():
                return
            tables = self.data.setdefault("tables", {})
            views_tbl = tables.setdefault("views", {
                "description": "Logical Views (Name + SELECT)",
                "source": "managementstudio",
                "metadata": {},
                "schema": {
                    "name": {"display_type": "text", "description": "View-Name"},
                    "query": {"display_type": "text", "description": "SELECT-Statement"},
                    "description": {"display_type": "text", "description": "Beschreibung"},
                    "created_at": {"display_type": "datetime", "description": "Erstellt am"},
                    "updated_at": {"display_type": "datetime", "description": "Aktualisiert am"}
                },
                "records": []
            })
            records = views_tbl.setdefault("records", []) or []
            by_name = {}
            for r in records:
                try:
                    nm = (r.get("name") or "").strip()
                    if nm:
                        by_name[nm] = r
                except Exception:
                    continue
            from datetime import datetime as _dt
            now = _dt.now().isoformat()
            for p in sorted(base_dir.iterdir()):
                if not p.is_file():
                    continue
                name = p.name.strip()
                if not name:
                    continue
                try:
                    sql = p.read_text(encoding="utf-8").strip()
                except Exception:
                    continue
                if not sql:
                    continue
                rec = by_name.get(name)
                if rec:
                    rec["query"] = sql
                    rec["updated_at"] = now
                else:
                    records.append({
                        "name": name,
                        "query": sql,
                        "description": f"Imported from Views/{name}",
                        "created_at": now,
                        "updated_at": now
                    })
        except Exception:
            return

    # ==============================
    # Sichere Speicher-Strategie
    # ==============================
    def _lock_path(self) -> Path:
        return self.db_path.with_suffix(self.db_path.suffix + '.lock')

    def _acquire_lock(self, timeout_seconds: float = 30.0) -> bool:
        import time, os
        lock = self._lock_path()
        start = time.time()
        while True:
            try:
                # Exklusives Lock via Datei-Erzeugung
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode('utf-8'))
                os.close(fd)
                return True
            except FileExistsError:
                if time.time() - start > timeout_seconds:
                    print(f"⚠️ Lock-Datei vorhanden: {lock} – Timeout erreicht")
                    return False
                time.sleep(0.1)

    def _release_lock(self) -> None:
        try:
            self._lock_path().unlink(missing_ok=True)
        except Exception:
            pass

    def _rotate_snapshots(self, max_snapshots: int = 10) -> None:
        try:
            parent = self.db_path.parent
            prefix = self.db_path.stem + '_'
            snaps = sorted(parent.glob(self.db_path.stem + '_*.json'))
            if len(snaps) > max_snapshots:
                to_delete = snaps[:len(snaps) - max_snapshots]
                for p in to_delete:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def safe_save(self, create_snapshot: bool = False, max_snapshots: int = 10) -> bool:
        import os, json
        from datetime import datetime
        try:
            # Verzeichnis sicherstellen
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            if not self._acquire_lock():
                return False
            try:
                # Serialize
                payload = json.dumps(self.data, indent=2, ensure_ascii=False)

                # Temporäre Datei schreiben
                tmp_path = self.db_path.with_suffix('.tmp.json')
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(payload)

                # Validierung: erneut einlesen
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    json.load(f)

                # Optional: Snapshot des alten DB-Files
                if create_snapshot and self.db_path.exists():
                    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
                    snapshot = self.db_path.parent / f"{self.db_path.stem}_{ts}.json"
                    try:
                        os.replace(self.db_path, snapshot)
                        self._rotate_snapshots(max_snapshots)
                    except Exception:
                        # Falls kein altes File vorhanden oder rename fehlschlägt → einfach weiter
                        pass

                # Atomar ersetzen
                os.replace(tmp_path, self.db_path)
                # mtime aktualisieren, damit Auto-Reload andere Prozesse erkennen können
                try:
                    self._last_loaded_mtime = self.db_path.stat().st_mtime
                except Exception:
                    pass

                print(f"✅ Datenbank sicher gespeichert: {self.db_path}")
                return True
            finally:
                # Aufräumen
                try:
                    tmp = self.db_path.with_suffix('.tmp.json')
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                self._release_lock()
        except Exception as e:
            print(f"❌ Fehler beim sicheren Speichern der Datenbank: {e}")
            return False

    # ==============================
    # Threshold-Methoden & Metriken
    # ==============================

    def ensure_threshold_methods_seeded(self) -> None:
        methods_table = self.data["tables"].setdefault("threshold_methods", {"records": []})
        existing = methods_table.get("records", [])
        existing_names = {m.get("method_name") for m in existing}
        seed = [
            (1, "precision_optimal", "Threshold maximiert Precision"),
            (2, "f1_optimal", "Threshold maximiert F1"),
            (3, "elbow", "Elbow-Methode in ROC"),
            (4, "recall_first", "Recall-First"),
            (5, "standard_0_5", "Standard Threshold 0.5")
        ]
        for mid, name, desc in seed:
            if name not in existing_names:
                existing.append({"method_id": mid, "method_name": name, "description": desc})
        methods_table["records"] = existing

    def get_threshold_method_id(self, method_name: str) -> int:
        self.ensure_threshold_methods_seeded()
        records = self.data["tables"]["threshold_methods"]["records"]
        for r in records:
            if r.get("method_name") == method_name:
                return r.get("method_id")
        next_id = max([r.get("method_id", 0) for r in records] or [0]) + 1
        records.append({"method_id": next_id, "method_name": method_name, "description": method_name})
        return next_id

    def add_threshold_metrics(self, experiment_id: int, method_name: str, threshold_value: float,
                               precision: float, recall: float, f1: float, data_split: str = "backtest",
                               is_selected: int = 0) -> bool:
        try:
            method_id = self.get_threshold_method_id(method_name)
            # Schema-Erweiterung sicherstellen (is_selected)
            tbl = self.data["tables"].setdefault("churn_threshold_metrics", {"records": []})
            schema = tbl.setdefault("schema", {})
            if "is_selected" not in schema:
                schema["is_selected"] = {"display_type": "integer", "description": "1 wenn gewählte Methode"}
            records = tbl.setdefault("records", [])
            # Replace-Strategie: Bei neu gewählter Methode (is_selected=1) alte Threshold-Zeilen dieses Experiments entfernen
            try:
                if int(is_selected) == 1 and records:
                    records[:] = [r for r in records if int(r.get("experiment_id", -1)) != int(experiment_id)]
            except Exception:
                pass
            # id_files aus dem Experiment übernehmen
            exp_files = []
            for exp in self.data["tables"].get("experiments", {}).get("records", []):
                if exp.get("experiment_id") == experiment_id:
                    exp_files = exp.get("id_files", [])
                    break
            records.append({
                "experiment_id": experiment_id,
                "method_id": method_id,
                "id_files": exp_files,
                "threshold_value": round(float(threshold_value), 6),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1": round(float(f1), 6),
                "data_split": data_split,
                "is_selected": int(1 if is_selected else 0),
                "calculated_at": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"❌ Fehler beim Persistieren der Threshold-Metriken: {e}")
            return False
    
    def add_churn_model_metrics(self, 
                                experiment_id: int, 
                                metrics: Dict[str, Any], 
                                data_split: str = "backtest",
                                ) -> bool:
        """
        Persistiert Modell-Metriken eines Churn-Runs (Training/Backtest) in die Tabelle 'churn_model_metrics'.
        Es werden nur vorhandene Felder übernommen (keine Fallbacks), 'experiment_id' ist Pflicht.
        """
        try:
            if experiment_id is None:
                raise ValueError("experiment_id ist erforderlich")
            # Tabelle und Schema sicherstellen
            tbl = self.data["tables"].setdefault("churn_model_metrics", {"records": []})
            schema = tbl.setdefault("schema", {})
            # Minimal-Schema ergänzen, ohne bestehende Einträge zu überschreiben
            schema.setdefault("experiment_id", {"display_type": "integer", "description": "Experiment-ID"})
            schema.setdefault("data_split", {"display_type": "text", "description": "training/backtest"})
            for col in ("auc", "precision", "recall", "f1", "threshold_used", 
                        "training_timebase", "prediction_timebase", "model_version", "dt_calculated"):
                schema.setdefault(col, {"display_type": "text", "description": ""})

            # id_files vom Experiment übernehmen
            exp_files = []
            for exp in self.data["tables"].get("experiments", {}).get("records", []):
                if exp.get("experiment_id") == experiment_id:
                    exp_files = exp.get("id_files", [])
                    break

            # Record aufbauen – nur echte Felder setzen
            rec: Dict[str, Any] = {
                "experiment_id": experiment_id,
                "data_split": data_split,
                "id_files": exp_files,
                "dt_calculated": datetime.now().isoformat()
            }
            for key in ("auc", "precision", "recall", "f1", "threshold_used", 
                        "training_timebase", "prediction_timebase", "model_version"):
                if metrics.get(key) is not None:
                    rec[key] = metrics.get(key)

            # Replace-Strategie: vorhandene Zeilen für dasselbe Experiment und data_split entfernen
            try:
                existing = tbl.setdefault("records", [])
                existing[:] = [r for r in existing if not (int(r.get("experiment_id", -1)) == int(experiment_id) and str(r.get("data_split")) == str(data_split))]
                existing.append(rec)
            except Exception:
                tbl.setdefault("records", []).append(rec)
            return True
        except Exception as e:
            print(f"❌ Fehler beim Persistieren der churn_model_metrics: {e}")
            return False

    def add_churn_business_metrics(self, 
                                   experiment_id: int, 
                                   business: Dict[str, Any]) -> bool:
        """
        Persistiert Business-Impact-Metriken eines Churn-Runs in 'churn_business_metrics'.
        Es werden nur vorhandene Felder übernommen (keine Fallbacks).
        """
        try:
            if experiment_id is None:
                raise ValueError("experiment_id ist erforderlich")

            # Tabelle und Schema sicherstellen
            tbl = self.data["tables"].setdefault("churn_business_metrics", {"records": []})
            schema = tbl.setdefault("schema", {})
            schema.setdefault("experiment_id", {"display_type": "integer", "description": "Experiment-ID"})
            for col in ("customers_at_risk", "customers_high_risk", "customers_medium_risk", "customers_low_risk",
                        "total_customers", "potential_revenue_loss", "prevention_cost_estimate", "roi_estimate",
                        "avg_customer_value", "max_customer_value", "prediction_timebase", "dt_calculated"):
                schema.setdefault(col, {"display_type": "text", "description": ""})

            # id_files vom Experiment übernehmen
            exp_files = []
            for exp in self.data["tables"].get("experiments", {}).get("records", []):
                if exp.get("experiment_id") == experiment_id:
                    exp_files = exp.get("id_files", [])
                    break

            rec: Dict[str, Any] = {
                "experiment_id": experiment_id,
                "id_files": exp_files,
                "dt_calculated": datetime.now().isoformat()
            }
            for key in ("customers_at_risk", "customers_high_risk", "customers_medium_risk", "customers_low_risk",
                        "total_customers", "potential_revenue_loss", "prevention_cost_estimate", "roi_estimate",
                        "avg_customer_value", "max_customer_value", "prediction_timebase"):
                if business.get(key) is not None:
                    rec[key] = business.get(key)

            # Replace-Strategie: vorhandene Zeilen für dasselbe Experiment entfernen
            try:
                existing = tbl.setdefault("records", [])
                existing[:] = [r for r in existing if int(r.get("experiment_id", -1)) != int(experiment_id)]
                existing.append(rec)
            except Exception:
                tbl.setdefault("records", []).append(rec)
            return True
        except Exception as e:
            print(f"❌ Fehler beim Persistieren der churn_business_metrics: {e}")
            return False
    
    def add_customer_churn_details(self, records: List[Dict]) -> bool:
        """
        Fügt Churn-spezifische Customer Details hinzu
        
        Args:
            records: Liste von Churn Customer Detail Records
            
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            if "customer_churn_details" not in self.data["tables"]:
                return False  # Tabelle sollte existieren nach Migration
                
            # Records zur Tabelle hinzufügen
            table_records = self.data["tables"]["customer_churn_details"]["records"]
            table_records.extend(records)
            
            # Metadata updaten
            self.data["tables"]["customer_churn_details"]["metadata"]["total_records"] = len(table_records)
            self.data["tables"]["customer_churn_details"]["metadata"]["last_updated"] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen von Churn Customer Details: {e}")
            return False
    
    def add_customer_cox_details(self, records: List[Dict]) -> bool:
        """
        Fügt Cox-spezifische Customer Details hinzu
        
        Args:
            records: Liste von Cox Customer Detail Records
            
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            if "customer_cox_details" not in self.data["tables"]:
                return False  # Tabelle sollte existieren nach Migration
                
            # Records zur Tabelle hinzufügen
            table_records = self.data["tables"]["customer_cox_details"]["records"]
            table_records.extend(records)
            
            # Metadata updaten
            self.data["tables"]["customer_cox_details"]["metadata"]["total_records"] = len(table_records)
            self.data["tables"]["customer_cox_details"]["metadata"]["last_updated"] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            print(f"❌ Fehler beim Hinzufügen von Cox Customer Details: {e}")
            return False
    
    def get_customer_churn_details(self, experiment_id: Optional[int] = None) -> List[Dict]:
        """
        Lädt Churn Customer Details
        
        Args:
            experiment_id: Optional filter by experiment
            
        Returns:
            Liste von Churn Customer Detail Records
        """
        try:
            if "customer_churn_details" not in self.data["tables"]:
                return []
            records = self.data["tables"]["customer_churn_details"].get("records", [])
            if experiment_id:
                records = [r for r in records if r.get('experiment_id') == experiment_id]
            return records
        except Exception as e:
            print(f"❌ Fehler beim Laden von Churn Customer Details: {e}")
            return []

    def get_customer_cox_details(self, experiment_id: Optional[int] = None) -> List[Dict]:
        """
        Lädt Cox Customer Details
        
        Args:
            experiment_id: Optional filter by experiment
            
        Returns:
            Liste von Cox Customer Detail Records
        """
        try:
            if "customer_cox_details" not in self.data["tables"]:
                return []
            records = self.data["tables"]["customer_cox_details"].get("records", [])
            if experiment_id:
                records = [r for r in records if r.get('experiment_id') == experiment_id]
            return records
        except Exception as e:
            print(f"❌ Fehler beim Laden von Cox Customer Details: {e}")
            return []

    def cleanup_redundant_records(self) -> None:
        """
        Entfernt redundante 'records' Felder und verwendet nur 'data'
        """
        print("🧹 Bereinige redundante records-Felder...")
        
        for table_name, table_data in self.data["tables"].items():
            if "records" in table_data and "data" in table_data:
                # Wenn beide Felder existieren und identisch sind, entferne 'data'
                if table_data["records"] == table_data["data"]:
                    del table_data["data"]
                    print(f"✅ Redundante 'data' aus Tabelle '{table_name}' entfernt")
                else:
                    print(f"⚠️  Tabelle '{table_name}' hat unterschiedliche 'records' und 'data'")
        
        self.save()
        print("✅ Datenbank-Bereinigung abgeschlossen")

    # =============================================
    # OUTBOX EXPORTER (CHURN, COX, COUNTERFACTUALS)
    # =============================================

    def export_churn_to_outbox(self, experiment_id: int) -> bool:
        """Exportiert Churn-Reports für ein Experiment in die Outbox."""
        try:
            from config.paths_config import ProjectPaths
            out_dir = ProjectPaths.outbox_churn_experiment_directory(int(experiment_id))
            ProjectPaths.ensure_directory_exists(out_dir)

            tables = self.data.get("tables", {})

            def _write_json(filename: str, records):
                with open(out_dir / filename, 'w', encoding='utf-8') as f:
                    _json.dump(records, f, ensure_ascii=False, indent=2)

            # Backtest-Results
            bt = [r for r in tables.get("backtest_results", {}).get("records", []) if int(r.get("id_experiments", -1)) == int(experiment_id)]
            if bt:
                _write_json("backtest_results.json", bt)

            # Model Metrics
            mm = [r for r in tables.get("churn_model_metrics", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
            if mm:
                _write_json("churn_model_metrics.json", mm)

            # Threshold Metrics
            tm = [r for r in tables.get("churn_threshold_metrics", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
            if tm:
                _write_json("churn_threshold_metrics.json", tm)

            # Business Metrics
            bm = [r for r in tables.get("churn_business_metrics", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
            if bm:
                _write_json("churn_business_metrics.json", bm)

            # Feature Importance (optional)
            fi = [r for r in tables.get("churn_feature_importance", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)] if "churn_feature_importance" in tables else []
            if fi:
                _write_json("churn_feature_importance.json", fi)

            # Customer Details (optional)
            cd = [r for r in tables.get("customer_churn_details", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)] if "customer_churn_details" in tables else []
            if cd:
                _write_json("customer_churn_details.json", cd)

            # KPIs (experiment_kpis)
            kpis = [r for r in tables.get("experiment_kpis", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
            if kpis:
                _write_json("kpis.json", kpis)

            return True
        except Exception as e:
            print(f"❌ Fehler beim Churn-Outbox-Export: {e}")
            return False

    def export_cox_to_outbox(self, experiment_id: int) -> bool:
        """Exportiert Cox-Reports für ein Experiment in die Outbox."""
        try:
            from config.paths_config import ProjectPaths
            out_dir = ProjectPaths.outbox_cox_experiment_directory(int(experiment_id))
            ProjectPaths.ensure_directory_exists(out_dir)

            tables = self.data.get("tables", {})

            def _write_json(filename: str, records):
                with open(out_dir / filename, 'w', encoding='utf-8') as f:
                    _json.dump(records, f, ensure_ascii=False, indent=2)

            # Survival
            surv = [r for r in tables.get("cox_survival", {}).get("records", []) if int(r.get("id_experiments", -1)) == int(experiment_id)]
            if surv:
                _write_json("cox_survival.json", surv)

            # Prioritization
            prio = [r for r in tables.get("cox_prioritization_results", {}).get("records", []) if int(r.get("id_experiments", -1)) == int(experiment_id)]
            if prio:
                _write_json("cox_prioritization.json", prio)

            # Metrics
            metrics = [r for r in tables.get("cox_analysis_metrics", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
            if metrics:
                _write_json("metrics.json", metrics)

            # KPIs
            kpis = [r for r in tables.get("experiment_kpis", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
            if kpis:
                _write_json("kpis.json", kpis)

            return True
        except Exception as e:
            print(f"❌ Fehler beim Cox-Outbox-Export: {e}")
            return False

    def export_counterfactuals_to_outbox(self, experiment_id: int) -> bool:
        """Exportiert Counterfactuals-Reports für ein Experiment in die Outbox."""
        try:
            from config.paths_config import ProjectPaths
            out_dir = ProjectPaths.outbox_counterfactuals_directory()
            out_dir = out_dir / f"experiment_{int(experiment_id)}"
            ProjectPaths.ensure_directory_exists(out_dir)

            tables = self.data.get("tables", {})

            def _write_json(filename: str, records):
                with open(out_dir / filename, 'w', encoding='utf-8') as f:
                    _json.dump(records, f, ensure_ascii=False, indent=2)

            # Core CF Reports
            for name, fname in [
                ("cf_individual", "cf_individual.json"),
                ("cf_aggregate", "cf_aggregate.json"),
                ("cf_individual_raw", "cf_individual_raw.json"),
                ("cf_aggregate_raw", "cf_aggregate_raw.json"),
                ("cf_business_metrics", "cf_business_metrics.json"),
                ("cf_feature_recommendations", "cf_feature_recommendations.json"),
                ("cf_cost_analysis", "cf_cost_analysis.json"),
            ]:
                if name in tables:
                    recs = [r for r in tables[name].get("records", []) if int(r.get("id_experiments", -1)) == int(experiment_id)]
                    if recs:
                        _write_json(fname, recs)

            return True
        except Exception as e:
            print(f"❌ Fehler beim Counterfactuals-Outbox-Export: {e}")
            return False

    def import_from_outbox_counterfactuals(self, experiment_id: int) -> bool:
        """Importiert Counterfactuals-Artefakte aus der Outbox in die JSON-DB."""
        try:
            from config.paths_config import ProjectPaths
            out_dir = ProjectPaths.outbox_counterfactuals_directory() / f"experiment_{int(experiment_id)}"
            if not out_dir.exists():
                print(f"ℹ️ CF-Outbox-Verzeichnis nicht gefunden: {out_dir}")
                return False

            def _read_json(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return _json.load(f)

            def _upsert(name: str, records):
                tbl = self.data["tables"].setdefault(name, {"records": []})
                existing = tbl.get("records", []) or []
                remaining = [r for r in existing if int(r.get('id_experiments', -1)) != int(experiment_id)]
                tbl["records"] = remaining + records
                self.data["tables"][name] = tbl

            mapping = {
                'cf_individual.json': 'cf_individual',
                'cf_aggregate.json': 'cf_aggregate',
                'cf_individual_raw.json': 'cf_individual_raw',
                'cf_aggregate_raw.json': 'cf_aggregate_raw',
                'cf_business_metrics.json': 'cf_business_metrics',
                'cf_feature_recommendations.json': 'cf_feature_recommendations',
                'cf_cost_analysis.json': 'cf_cost_analysis',
            }
            for fname, tname in mapping.items():
                p = out_dir / fname
                if p.exists():
                    recs = _read_json(p)
                    _upsert(tname, recs)

            return True
        except Exception as e:
            print(f"❌ Fehler beim Counterfactuals-Outbox-Import: {e}")
            return False

    def export_all_reports_to_outbox(self) -> int:
        """Exportiert für alle vorhandenen Experimente die Reports aller Module in die Outbox.
        Returns: Anzahl Experimente, für die exportiert wurde
        """
        try:
            tables = self.data.get("tables", {})
            exp_ids = set()
            # Sammle Experiment-IDs aus experiments-Tabelle
            for e in tables.get("experiments", {}).get("records", []):
                if e.get("experiment_id") is not None:
                    exp_ids.add(int(e.get("experiment_id")))
            # Sammle zusätzlich aus vorhandenen Modul-Tabellen
            for tname in (
                "backtest_results", "churn_model_metrics", "churn_threshold_metrics", "churn_business_metrics",
                "churn_feature_importance", "customer_churn_details", "cox_survival", "cox_prioritization_results",
                "cox_analysis_metrics", "experiment_kpis", "cf_individual", "cf_aggregate", "cf_individual_raw",
                "cf_aggregate_raw", "cf_business_metrics", "cf_feature_recommendations", "cf_cost_analysis"
            ):
                if tname in tables:
                    for r in tables[tname].get("records", []):
                        # verschiedene Schlüssel je Tabelle
                        eid = r.get("experiment_id") or r.get("id_experiments")
                        if eid is not None:
                            try:
                                exp_ids.add(int(eid))
                            except Exception:
                                pass

            count = 0
            for exp_id in sorted(exp_ids):
                ok1 = self.export_churn_to_outbox(exp_id)
                ok2 = self.export_cox_to_outbox(exp_id)
                ok3 = self.export_counterfactuals_to_outbox(exp_id)
                if ok1 or ok2 or ok3:
                    count += 1
            return count
        except Exception as e:
            print(f"❌ Fehler beim Outbox-Gesamtexport: {e}")
            return 0


def migrate_to_json_database():
    """
    Migration-Script für alle bestehenden JSON-Dateien
    """
    print("🚀 STARTE MIGRATION ZUR JSON-DATENBANK")
    print("=" * 50)
    
    db = ChurnJSONDatabase()
    
    # Stage0-Daten migrieren (verwendet bereits "Kunde")
    stage0_files = list(ProjectPaths.dynamic_system_outputs_directory().glob("stage0_cache/*.json"))
    if stage0_files:
        latest_stage0 = max(stage0_files, key=lambda p: p.stat().st_mtime)
        db.add_customers_from_stage0(str(latest_stage0))
    
    # Backtest-Ergebnisse migrieren (verwendet bereits "Kunde")
    backtest_files = list(ProjectPaths.models_directory().glob("Enhanced_EarlyWarning_Backtest_*.json"))
    if backtest_files:
        latest_backtest = max(backtest_files, key=lambda p: p.stat().st_mtime)
        db.add_backtest_results(str(latest_backtest))
    
    # Cox-Panel migrieren (konvertiert "customer_id" -> "Kunde")
    cox_files = list(ProjectPaths.dynamic_system_outputs_directory().glob("cox_survival_data/cox_survival_panel_v4_*.json"))
    if cox_files:
        latest_cox = max(cox_files, key=lambda p: p.stat().st_mtime)
        db.add_cox_survival(str(latest_cox))
    
    # Prioritization migrieren (verwendet bereits "Kunde")
    prioritization_files = list(ProjectPaths.dynamic_system_outputs_directory().glob("prioritization/prioritization_*_full.json"))
    if prioritization_files:
        latest_prioritization = max(prioritization_files, key=lambda p: p.stat().st_mtime)
        db.add_prioritization(str(latest_prioritization))
    
    # Bestehende Experimente extrahieren
    extract_existing_experiments(db)
    
    # Datenbank speichern
    if db.save():
        print("\n✅ MIGRATION ERFOLGREICH ABGESCHLOSSEN!")
        
        # Statistiken anzeigen
        stats = db.get_statistics()
        print(f"\n📊 DATENBANK-STATISTIKEN:")
        print(f"   Gesamtanzahl Kunden: {stats['total_customers']}")
        print(f"   Datenquellen: {', '.join(stats['data_sources'])}")
        
        for table_name, table_stats in stats["tables"].items():
            print(f"   {table_name}: {table_stats['record_count']} Records")
        
        return True
    else:
        print("\n❌ MIGRATION FEHLGESCHLAGEN!")
        return False


def extract_existing_experiments(db: ChurnJSONDatabase):
    """
    Extrahiert bestehende Experimente aus vorhandenen Daten
    """
    print("\n🔬 EXTRAHIERE BESTEHENDE EXPERIMENTE")
    print("=" * 40)
    
    # Experiment 1: Enhanced Early Warning Backtest
    backtest_files = list(ProjectPaths.models_directory().glob("Enhanced_EarlyWarning_Backtest_*.json"))
    if backtest_files:
        latest_backtest = max(backtest_files, key=lambda p: p.stat().st_mtime)
        
        # Lade Backtest-Daten für Metadaten
        with open(latest_backtest, 'r', encoding='utf-8') as f:
            backtest_data = json.load(f)
        
        # Erstelle Experiment
        exp_id = db.create_experiment(
            experiment_name="Enhanced Early Warning Backtest",
            training_from="201712",  # YYYYMM Format
            training_to="202407",    # YYYYMM Format
            backtest_from="202408",  # YYYYMM Format
            backtest_to="202412",    # YYYYMM Format
            model_type="binary",
            feature_set="enhanced",
            hyperparameters={
                "threshold": backtest_data.get("optimal_threshold", 0.5),
                "model_type": backtest_data.get("model_type", "unknown")
            }
        )
        
        # Extrahiere KPIs aus Backtest-Daten
        performance = backtest_data.get("backtest_results", {})
        if performance:
            # AUC
            if "auc" in performance:
                db.add_experiment_kpi(exp_id, "auc", performance["auc"], "backtest")
            
            # Precision, Recall, F1
            if "precision" in performance:
                db.add_experiment_kpi(exp_id, "precision", performance["precision"], "backtest")
            if "recall" in performance:
                db.add_experiment_kpi(exp_id, "recall", performance["recall"], "backtest")
            if "f1_score" in performance:
                db.add_experiment_kpi(exp_id, "f1", performance["f1_score"], "backtest")
            elif "f1" in performance:
                db.add_experiment_kpi(exp_id, "f1", performance["f1"], "backtest")
    
    # Experiment 2: Cox Survival Analysis
    cox_files = list(ProjectPaths.dynamic_system_outputs_directory().glob("cox_survival_data/cox_survival_panel_v4_*.json"))
    if cox_files:
        latest_cox = max(cox_files, key=lambda p: p.stat().st_mtime)
        
        # Lade Cox-Daten für Metadaten
        with open(latest_cox, 'r', encoding='utf-8') as f:
            cox_data = json.load(f)
        
        cox_survival_data = cox_data.get("cox_survival_data", {})
        metadata = cox_survival_data.get("metadata", {})
        
        # Erstelle Experiment
        exp_id = db.create_experiment(
            experiment_name="Cox Survival Analysis",
            training_from="201712",  # YYYYMM Format
            training_to="202407",    # YYYYMM Format
            backtest_from="202408",  # YYYYMM Format
            backtest_to="202412",    # YYYYMM Format
            model_type="cox",
            feature_set="standard",
            hyperparameters={
                "cutoff_exclusive": metadata.get("cutoff_exclusive"),
                "feature_columns_count": len(metadata.get("feature_columns", []))
            }
        )
        
        # Cox-spezifische KPIs (falls verfügbar)
        if "c_index" in metadata:
            db.add_experiment_kpi(exp_id, "c_index", metadata["c_index"], "backtest")
        
        # Survival-Statistiken
        survival_records = cox_survival_data.get("survival_records", [])
        if survival_records:
            total_records = len(survival_records)
            churn_events = sum(1 for record in survival_records if record.get("event") == 1)
            churn_rate = churn_events / total_records if total_records > 0 else 0
            
            db.add_experiment_kpi(exp_id, "churn_rate", churn_rate, "backtest")
            db.add_experiment_kpi(exp_id, "total_records", total_records, "backtest")
            db.add_experiment_kpi(exp_id, "churn_events", churn_events, "backtest")
    
    print("✅ Bestehende Experimente extrahiert")


if __name__ == "__main__":
    # Test der Datenbank
    migrate_to_json_database()
