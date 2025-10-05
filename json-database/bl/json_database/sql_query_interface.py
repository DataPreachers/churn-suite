#!/usr/bin/env python3
"""
SQL-ähnliches Query-Interface für Churn JSON-Datenbank
Ermöglicht SQL-ähnliche Abfragen mit Tabellen-Ausgabe
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from bl.json_database.churn_json_database import ChurnJSONDatabase
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate
import json
from datetime import datetime

# Optional: DuckDB für echte SQL-Unterstützung
try:
    import duckdb  # type: ignore
    _DUCKDB_AVAILABLE = True
except Exception:
    duckdb = None  # type: ignore
    _DUCKDB_AVAILABLE = False

# Pfade
from config.paths_config import ProjectPaths  # paths_config verwenden


class SQLQueryInterface:
    """
    SQL-ähnliches Interface für die Churn JSON-Datenbank
    """
    
    def __init__(self):
        """Initialisiert das Query-Interface"""
        self.db = ChurnJSONDatabase()
        self.history = []
        self._data_dictionary = self._load_data_dictionary()
        # Typen aus Data Dictionary nur validieren, nicht erzwingen (kein hartes Casting)
        self.strict_types = False
    
    def _load_data_dictionary(self) -> Dict[str, Any]:
        """Lädt das Data Dictionary aus der Konfiguration."""
        try:
            dd_path = ProjectPaths.config_directory() / "data_dictionary_optimized.json"
            with open(dd_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _infer_pandas_dtype(self, dd_type: str) -> str:
        """Mappt Data-Dictionary Typen auf Pandas-Dtypes."""
        t = (dd_type or "").upper()
        if t in ("INTEGER", "INT", "BINARY"):
            return "Int64"  # nullable integer
        if t in ("BOOLEAN", "BOOL"):
            return "boolean"  # pandas nullable boolean
        if t in ("NUMERIC", "DOUBLE", "FLOAT"):
            return "float64"
        if t in ("CATEGORICAL", "TEXT", "STRING"):
            return "string"
        return "string"
    
    def _cast_dataframe(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Wendet Typen aus dem Data Dictionary (und minimale Heuristiken) auf den DataFrame an."""
        if df is None or df.empty:
            return df
        dd_cols = (self._data_dictionary.get("columns", {}) if isinstance(self._data_dictionary, dict) else {})
        # Ziel-Dtypes pro Spalte sammeln
        dtype_map: Dict[str, str] = {}
        for col in df.columns:
            dd_info = dd_cols.get(col)
            if isinstance(dd_info, dict) and "data_type" in dd_info:
                dtype_map[col] = self._infer_pandas_dtype(dd_info.get("data_type"))
        # Wunsch: Keine Laufzeit-Casts, nur Validierung (Typen am Anfang bestimmen)
        if not self.strict_types:
            return df
        # Keine Heuristiken im Strict Mode
        # Casting anwenden robust
        for col, pd_type in dtype_map.items():
            if col not in df.columns:
                continue
            try:
                if pd_type in ("float64",):
                    # Erzwinge float64 explizit (pd.to_numeric behält sonst int64 bei)
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                elif pd_type == "Int64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif pd_type in ("boolean", "bool"):
                    def _to_bool(v):
                        if v is True or (isinstance(v, (int, float)) and v == 1):
                            return True
                        if v is False or (isinstance(v, (int, float)) and v == 0):
                            return False
                        if isinstance(v, str):
                            s = v.strip().lower()
                            if s in ("true", "1", "yes", "y"): return True
                            if s in ("false", "0", "no", "n"): return False
                        return None
                    df[col] = df[col].map(_to_bool).astype("boolean")
                elif pd_type == "string":
                    df[col] = df[col].astype("string")
            except Exception:
                # im Zweifel Spalte unangetastet lassen
                pass

        # Strict Validation: Stelle sicher, dass erwartete Dtypes erreicht wurden
        errors = []
        for col, pd_type in dtype_map.items():
            try:
                actual = str(df[col].dtype)
            except Exception:
                actual = "<unknown>"
            expected = pd_type
            # pandas repräsentiert nullable integer als 'Int64', boolean als 'boolean'
            if actual != expected:
                # kleine Toleranz: float64 kann als 'float64' erscheinen (ok)
                if not (expected == "float64" and actual == "float64"):
                    errors.append(f"Spalte '{col}': erwartet {expected}, erhalten {actual}")
        if errors and self.strict_types:
            raise ValueError("Typvalidierung fehlgeschlagen – Data Dictionary strikt: " + "; ".join(errors))
        return df
    
    def _execute_with_duckdb(self, query: str) -> List[Dict[str, Any]]:
        """
        Führt eine SQL-Query mit DuckDB gegen die aktuellen JSON-Tabellen aus.
        Registriert alle Tabellen (records -> DataFrame) als Views und führt dann die Query aus.
        """
        # Stelle sicher, dass die JSON-DB vor jeder Query frisch ist (Auto-Reload)
        try:
            self.db.maybe_reload()
        except Exception:
            pass
        if not _DUCKDB_AVAILABLE:
            raise RuntimeError("DuckDB nicht verfügbar. Bitte 'pip install duckdb' ausführen.")
        # In-Memory Connection
        con = duckdb.connect(database=':memory:')
        
        # Alle Tabellen registrieren
        for table_name, table_data in self.db.data.get("tables", {}).items():
            records = table_data.get("records", []) or []
            df = pd.DataFrame(records)
            
            # ✅ BOOLEAN→INTEGER Konvertierung für DuckDB Type-Kompatibilität
            # DuckDB kann keine Mixed Boolean/Double Spalten verarbeiten
            for col in df.columns:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype('Int64')  # Pandas nullable integer
                elif df[col].apply(lambda x: isinstance(x, bool)).any():
                    # Mixed types: konvertiere Booleans zu Integers
                    df[col] = df[col].apply(lambda x: int(x) if isinstance(x, bool) else x)
            
            df = self._cast_dataframe(table_name, df)

            # ✅ Minimal-invasive Typkorrekturen für Kernfelder (ohne strikten Modus)
            # Ziel: BETWEEN/JOIN ohne explizite CASTs in Queries ermöglichen
            try:
                if table_name.lower() == "experiments":
                    for k in ("training_from", "training_to", "backtest_from", "backtest_to"):
                        if k in df.columns:
                            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
                    if "file_id" in df.columns:
                        df["file_id"] = pd.to_numeric(df["file_id"], errors="coerce").astype("Int64")
                elif table_name.lower() == "rawdata":
                    for k in ("I_TIMEBASE", "file_id"):
                        if k in df.columns:
                            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
            except Exception:
                pass
            # Falls DataFrame keine Spalten hat, versuche Schema zu verwenden
            if df.shape[1] == 0:
                schema = table_data.get("schema", {})
                if isinstance(schema, dict) and len(schema.keys()) > 0:
                    df = pd.DataFrame(columns=list(schema.keys()))
            # Wenn immer noch keine Spalten vorhanden, Registrierung überspringen
            if df.shape[1] == 0:
                continue
            con.register(table_name, df)

        # Spezielle Registrierung: rawdata_original (geflachte Originalspalten aus 'features')
        try:
            raw_tbl = self.db.data.get("tables", {}).get("rawdata", {})
            raw_records = raw_tbl.get("records", []) or []
            if raw_records:
                raw_df = pd.DataFrame(raw_records)
                # 'features' kann fehlen oder null sein
                features_series = raw_df.get("features")
                if features_series is not None:
                    try:
                        # json_normalize auf dict-ähnlichen Inhalten, robuste Füllung bei None
                        feat_df = pd.json_normalize([
                            (r if isinstance(r, dict) else {}) for r in features_series.tolist()
                        ])
                    except Exception:
                        feat_df = pd.DataFrame()
                else:
                    feat_df = pd.DataFrame()

                # Zusatz: bringe evtl. Kernfelder (Kunde/I_TIMEBASE) auch dann rein,
                # wenn sie top-level in rawdata liegen sollten
                for col in ("Kunde", "I_TIMEBASE"):
                    if col not in feat_df.columns and col in raw_df.columns:
                        feat_df[col] = raw_df[col]

                # Wenn keine Feature-Spalten ermittelt werden konnten, registriere nicht
                if not feat_df.empty:
                    # Optional: konsistente Typen anwenden
                    feat_df = self._cast_dataframe("rawdata_original", feat_df)
                    con.register("rawdata_original", feat_df)
        except Exception:
            # Registrierung ist optional; bei Fehlern still fortfahren
            pass
        
        # YYYYMM Hilfsfunktionen nach Tabellen-Registrierung
        self._register_yyyymm_functions(con)
        
        # Cox-spezifische Views und Funktionen registrieren
        self._register_cox_functions(con)

        # Counterfactuals-Views (aus Artefakten) registrieren
        self._register_cf_views(con)
        
        # Query ausführen und als DataFrame holen
        out_df: pd.DataFrame = con.execute(query).fetchdf()
        con.close()
        # In Dict-Liste konvertieren
        return out_df.to_dict(orient='records')

    def _build_pivot_case_sql(self,
                               target_yyyymm: int,
                               years: int = 2,
                               month: Optional[int] = 12,
                               scope: str = "same-file",
                               threshold: str = "optimal",
                               base: str = "churned") -> str:
        """
        Baut ein dynamisches SQL (CASE/AGGREGATION statt PIVOT), das für einen Ziel-Backtestmonat
        und eine dynamisch generierte Liste historischer Monate (z. B. Dezember der Vorjahre)
        die Churn-Wahrscheinlichkeit spaltenweise ausgibt.

        Args:
            target_yyyymm: Ziel-Backtestmonat (z. B. 202507)
            years: Anzahl der Vorjahre, die betrachtet werden (nur für month-basierten Rückblick)
            month: Optionaler Monat (z. B. 12 für Dezember). Wenn gesetzt, werden (target_year - i, month) gewählt
            scope: "same-file" (nur Experimente mit gleichen id_files) oder "all"
            threshold: Klassifikations-Flag in customer_details, z. B. "optimal", "std05", "f1", "precision", "recall"

        Returns:
            Vollständiges SQL-Statement als String
        """
        # Threshold-Mapping auf Spaltennamen in customer_details
        threshold_map = {
            "optimal": "Predicted_Optimal",
            "std05": "Predicted_Standard_0.5",
            "f1": "Predicted_F1_Optimal",
            "precision": "Predicted_Precision_First",
            "recall": "Predicted_Recall_First",
        }
        threshold_key = (threshold or "optimal").lower()
        predicted_col = threshold_map.get(threshold_key, "Predicted_Optimal")

        # Ziel- und Vergleichsmonate ermitteln (rein arithmetisch, keine Abfrage nötig)
        try:
            target_year = int(str(int(target_yyyymm))[:4])
        except Exception:
            target_year = target_yyyymm // 100
        months_list = [int(target_yyyymm)]
        if month is not None:
            try:
                month_int = int(month)
            except Exception:
                month_int = 12
            if month_int < 1 or month_int > 12:
                month_int = 12
            for i in range(1, max(0, int(years)) + 1):
                m = target_year - i
                months_list.append(m * 100 + month_int)

        # Eindeutig und sortiert (Ziel zuerst, dann absteigend)
        months_seen = set()
        months_unique: List[int] = []
        for m in months_list:
            if m not in months_seen:
                months_seen.add(m)
                months_unique.append(m)

        # CASE-Spalten generieren
        case_cols = []
        for m in months_unique:
            case_cols.append(
                f"  MAX(CASE WHEN p.yyyymm = {m} THEN p.churn_prob END) AS churnprob_{m}"
            )
        case_cols_sql = ",\n".join(case_cols)

        # Scope-Bedingung: gleiche Files vs. alle (fallback auf all, wenn id_files nicht nutzbar)
        effective_scope = (scope or "same-file").lower()
        if effective_scope == "same-file":
            try:
                exp_records = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
                target_yyyymm_int = int(target_yyyymm)
                def to_int_val(x):
                    try:
                        return int(x)
                    except Exception:
                        return None
                matches = [e for e in exp_records if to_int_val(e.get("backtest_to")) == target_yyyymm_int or e.get("backtest_to_int") == target_yyyymm_int]
                has_ids = any(isinstance(e.get("id_files"), list) and len(e.get("id_files")) > 0 for e in matches)
                if not has_ids:
                    effective_scope = "all"
            except Exception:
                effective_scope = "all"

        if effective_scope == "same-file":
            scope_cte = (
                "target_files AS (\n"
                "  SELECT DISTINCT file_id\n"
                "  FROM target t\n"
                "  JOIN experiments e ON e.experiment_id = t.experiment_id\n"
                "  CROSS JOIN UNNEST(e.id_files) AS f(file_id)\n"
                "),\n"
                "candidate_experiments AS (\n"
                "  SELECT DISTINCT e.experiment_id, e.backtest_to_int\n"
                "  FROM exp e\n"
                "  CROSS JOIN UNNEST(e.id_files) AS f(file_id)\n"
                "  JOIN target_files tf ON tf.file_id = f.file_id\n"
                ")\n"
                ",\n"  # wichtig: nach dem letzten CTE ein Komma für den nächsten Block
            )
        else:
            scope_cte = (
                "candidate_experiments AS (\n"
                "  SELECT DISTINCT e.experiment_id, e.backtest_to_int\n"
                "  FROM exp e\n"
                ")\n"
                ",\n"  # wichtig: nach dem letzten CTE ein Komma für den nächsten Block
            )

        # Basis-Kohorte bestimmen
        base_mode = (base or "churned").lower()
        if base_mode not in ("churned", "all"):
            base_mode = "churned"

        base_cte = ""
        base_filter_clause = ""
        if base_mode == "churned":
            base_cte = (
                "churned_base AS (\n"
                "  SELECT DISTINCT cd.Kunde\n"
                "  FROM customer_churn_details cd\n"
                "  JOIN target t ON cd.experiment_id = t.experiment_id\n"
                  "  WHERE cd.I_ALIVE = 0\n"
                "),\n"
            )
            base_filter_clause = "  WHERE cd.Kunde IN (SELECT Kunde FROM churned_base)\n"

        # Endgültiges SQL zusammensetzen
        sql = f"""
WITH exp AS (
  SELECT 
    e.*,
    CAST(training_from AS INTEGER) AS training_from_int,
    CAST(training_to AS INTEGER) AS training_to_int,
    CAST(backtest_from AS INTEGER) AS backtest_from_int,
    CAST(backtest_to AS INTEGER) AS backtest_to_int
  FROM experiments e
),
target AS (
  SELECT e.experiment_id, e.backtest_to_int
  FROM exp e
  WHERE e.backtest_to_int = {int(target_yyyymm)}
  LIMIT 1
),
{scope_cte}
{base_cte}
probs AS (
  SELECT
    cd.Kunde,
    ce.backtest_to_int AS yyyymm,
    cd.Churn_Wahrscheinlichkeit AS churn_prob
  FROM customer_churn_details cd
  JOIN candidate_experiments ce ON ce.experiment_id = cd.experiment_id
{base_filter_clause})
SELECT
  p.Kunde,
{case_cols_sql}
FROM probs p
GROUP BY p.Kunde
ORDER BY p.Kunde;
"""
        return sql

    def _record_cli_run(self, procedure: str, params: Dict[str, Any], table_name: str) -> None:
        """Protokolliert einen CLI-Run in der JSON-DB Tabelle 'cli'."""
        try:
            cli_tbl = self.db.data["tables"].setdefault("cli", {"records": []})
            records: List[Dict[str, Any]] = cli_tbl.setdefault("records", [])
            max_id = max([r.get("run_id", 0) for r in records] or [0])
            records.append({
                "run_id": max_id + 1,
                "procedure": procedure,
                "params": params,
                "table_name": table_name,
                "description": "Gespeicherte Prozedur-Ausführung",
                "created_at": datetime.now().isoformat()
            })
            cli_tbl["records"] = records
            self.db.save()
        except Exception:
            pass
    
    def _register_yyyymm_functions(self, con):
        """
        Registriert YYYYMM Hilfsfunktionen in DuckDB entsprechend data_dictionary I_TIMEBASE Format
        """
        # Robust: experiments_timebase als DataFrame registrieren (kein SQL-View),
        # damit fehlende Spalten in 'experiments' nicht zu Binder-Fehlern führen.
        try:
            import pandas as pd  # type: ignore
            exp_records = self.db.data.get("tables", {}).get("experiments", {}).get("records", []) or []
            exp_df = pd.DataFrame(exp_records)
            if exp_df is None or exp_df.empty:
                exp_timebase_df = pd.DataFrame(columns=[
                    "training_from","training_to","backtest_from","backtest_to",
                    "training_from_int","training_to_int","backtest_from_int","backtest_to_int",
                    "training_year","training_month","backtest_year","backtest_month",
                    "training_duration_months"
                ])
            else:
                # Hilfsfunktionen lokal
                def to_int_safe(x):
                    try:
                        return int(x)
                    except Exception:
                        return None
                # Spalten sicherstellen
                for col in ("training_from","training_to","backtest_from","backtest_to"):
                    if col not in exp_df.columns:
                        exp_df[col] = None
                exp_df["training_from_int"] = exp_df["training_from"].apply(to_int_safe)
                exp_df["training_to_int"] = exp_df["training_to"].apply(to_int_safe)
                exp_df["backtest_from_int"] = exp_df["backtest_from"].apply(to_int_safe)
                exp_df["backtest_to_int"] = exp_df["backtest_to"].apply(to_int_safe)
                # Jahr/Monat nur wenn möglich
                def year_part(v):
                    try:
                        v = int(v)
                        return v // 100
                    except Exception:
                        return None
                def month_part(v):
                    try:
                        v = int(v)
                        return v % 100
                    except Exception:
                        return None
                exp_df["training_year"] = exp_df["training_from_int"].apply(year_part)
                exp_df["training_month"] = exp_df["training_from_int"].apply(month_part)
                exp_df["backtest_year"] = exp_df["backtest_from_int"].apply(year_part)
                exp_df["backtest_month"] = exp_df["backtest_from_int"].apply(month_part)
                def duration_months(bf, tf):
                    try:
                        by, bm = int(bf)//100, int(bf)%100
                        ty, tm = int(tf)//100, int(tf)%100
                        return (by - ty) * 12 + (bm - tm)
                    except Exception:
                        return None
                exp_df["training_duration_months"] = exp_df.apply(lambda r: duration_months(r.get("backtest_from_int"), r.get("training_from_int")), axis=1)
                exp_timebase_df = exp_df
            con.register("experiments_timebase", exp_timebase_df)
        except Exception:
            # Fallback: leeres Schema registrieren
            try:
                import pandas as pd  # type: ignore
                con.register("experiments_timebase", pd.DataFrame(columns=[
                    "training_from","training_to","backtest_from","backtest_to",
                    "training_from_int","training_to_int","backtest_from_int","backtest_to_int",
                    "training_year","training_month","backtest_year","backtest_month",
                    "training_duration_months"
                ]))
            except Exception:
                pass
        
        # YYYYMM Macro Functions
        con.execute("""
        CREATE OR REPLACE MACRO yyyymm_to_year(yyyymm) AS CAST(CAST(yyyymm AS BIGINT) / 100 AS BIGINT);
        """)
        
        con.execute("""
        CREATE OR REPLACE MACRO yyyymm_to_month(yyyymm) AS CAST(CAST(yyyymm AS BIGINT) % 100 AS BIGINT);
        """)
        
        # Robuste Monate-Differenz mit ganzzahliger Division: diff(a,b) = months(a) - months(b)
        con.execute("""
        CREATE OR REPLACE MACRO yyyymm_total_months(x) AS 
            CAST(CAST(x AS BIGINT) / 100 AS BIGINT) * 12 + CAST(CAST(x AS BIGINT) % 100 AS BIGINT);
        """)
        con.execute("""
        CREATE OR REPLACE MACRO yyyymm_diff_months(a, b) AS 
            yyyymm_total_months(a) - yyyymm_total_months(b);
        """)
        
        con.execute("""
        CREATE OR REPLACE MACRO yyyymm_between(yyyymm, from_yyyymm, to_yyyymm) AS 
        CAST(yyyymm AS INTEGER) BETWEEN CAST(from_yyyymm AS INTEGER) AND CAST(to_yyyymm AS INTEGER);
        """)
        
        print("📅 YYYYMM Hilfsfunktionen registriert (entsprechend data_dictionary I_TIMEBASE)")
        print("   - experiments_timebase VIEW (mit Jahr/Monat/Dauer-Feldern)")
        print("   - yyyymm_to_year(yyyymm) -> Jahr")
        print("   - yyyymm_to_month(yyyymm) -> Monat")  
        print("   - yyyymm_diff_months(from, to) -> Monate zwischen")
        print("   - yyyymm_between(yyyymm, from, to) -> Boolean")
    
    def _register_cox_functions(self, con):
        """
        Registriert Cox-spezifische Views und Funktionen für Survival-Analysis
        Erweiterte Horizonte: 6, 12, 18, 24 Monate
        """
        # Wenn keine Cox-Daten vorhanden sind, überspringen
        try:
            tables = self.db.data.get("tables", {})
            has_survival = bool(tables.get("cox_survival", {}).get("records", []))
            has_prior = bool(tables.get("cox_prioritization_results", {}).get("records", []))
            if not (has_survival and has_prior):
                return
        except Exception:
            return

        # Cox Survival Analysis View (erweitert mit mehr Horizonten)
        try:
            con.execute("""
            CREATE OR REPLACE VIEW cox_survival_enhanced AS 
            SELECT c.*,
                -- Überlebenszeit in Monaten (gerundet)
                ROUND(c.duration, 2) as survival_months,
                
                -- Risk-Kategorien basierend auf Cox-Prioritization-Results
                CASE 
                    WHEN cp.PriorityScore >= 70 THEN 'Sehr Hoch'
                    WHEN cp.PriorityScore >= 50 THEN 'Hoch'  
                    WHEN cp.PriorityScore >= 30 THEN 'Mittel'
                    WHEN cp.PriorityScore >= 15 THEN 'Niedrig'
                    ELSE 'Sehr Niedrig'
                END as risk_category,
                
                -- Survival-Wahrscheinlichkeiten für verschiedene Horizonte (aus vorhandenen Feldern)
                COALESCE(cp.P_Event_6m, 0.0) as p_event_6m,
                COALESCE(cp.P_Event_12m, 0.0) as p_event_12m,
                -- 18m/24m: Intelligente Approximation basierend auf vorhandenen Daten
                CASE 
                    WHEN cp.P_Event_12m > 0 THEN LEAST(cp.P_Event_12m * 1.3, 1.0)  
                    ELSE 0.0 
                END as p_event_18m,
                CASE 
                    WHEN cp.P_Event_12m > 0 THEN LEAST(cp.P_Event_12m * 1.5, 1.0)  
                    ELSE 0.0 
                END as p_event_24m,
                
                -- Überlebens-Wahrscheinlichkeiten (1 - Event-Wahrscheinlichkeit) 
                (1.0 - COALESCE(cp.P_Event_6m, 0.0)) as p_survival_6m,
                (1.0 - COALESCE(cp.P_Event_12m, 0.0)) as p_survival_12m,
                (1.0 - CASE WHEN cp.P_Event_12m > 0 THEN LEAST(cp.P_Event_12m * 1.3, 1.0) ELSE 0.0 END) as p_survival_18m,
                (1.0 - CASE WHEN cp.P_Event_12m > 0 THEN LEAST(cp.P_Event_12m * 1.5, 1.0) ELSE 0.0 END) as p_survival_24m,
                
                -- RMST (Restricted Mean Survival Time) 
                COALESCE(cp.RMST_12m, 0.0) as rmst_12m,
                COALESCE(cp.RMST_24m, 0.0) as rmst_24m,
                
                -- Prioritäts-Score und erwartete Lebensdauer
                COALESCE(cp.PriorityScore, 0.0) as priority_score,
                COALESCE(cp.MonthsToLive_Conditional, 0.0) as expected_lifetime_months,
                COALESCE(cp.MonthsToLive_Unconditional, 0.0) as expected_lifetime_unconditional,
                
                -- Experiment-Information
                e.experiment_name,
                e.model_type,
                e.created_at as experiment_date
                
            FROM cox_survival c
            LEFT JOIN cox_prioritization_results cp ON c.Kunde = cp.Kunde AND c.id_experiments = cp.id_experiments
            LEFT JOIN experiments e ON c.id_experiments = e.experiment_id
            """)
            
            # Customer Risk Profiling View  
            con.execute("""
            CREATE OR REPLACE VIEW customer_risk_profile AS
            SELECT 
                Kunde,
                id_experiments as experiment_id,
                
                -- Risiko-Kategorisierung
                risk_category,
                priority_score,
                
                -- Survival-Wahrscheinlichkeiten in %
                ROUND(p_survival_6m * 100, 1) as survival_6m_percent,
                ROUND(p_survival_12m * 100, 1) as survival_12m_percent, 
                ROUND(p_survival_18m * 100, 1) as survival_18m_percent,
                ROUND(p_survival_24m * 100, 1) as survival_24m_percent,
                
                -- Churn-Wahrscheinlichkeiten in %
                ROUND(p_event_6m * 100, 1) as churn_risk_6m_percent,
                ROUND(p_event_12m * 100, 1) as churn_risk_12m_percent,
                ROUND(p_event_18m * 100, 1) as churn_risk_18m_percent,
                ROUND(p_event_24m * 100, 1) as churn_risk_24m_percent,
                
                -- Erwartete Lebensdauer
                ROUND(expected_lifetime_months, 1) as expected_months_remaining,
                
                -- Status
                CASE WHEN event = 1 THEN 'Churned' ELSE 'Active' END as customer_status,
                
                -- Experiment-Info
                experiment_name,
                experiment_date
                
            FROM cox_survival_enhanced
            """)
            
            # Cox Performance Summary View
            con.execute("""
            CREATE OR REPLACE VIEW cox_performance_summary AS
            SELECT 
                e.experiment_id,
                e.experiment_name,
                e.model_type,
                e.created_at,
                
                -- Performance-Metriken aus experiment_kpis
                MAX(CASE WHEN ek.metric_name = 'c_index' THEN ek.metric_value END) as c_index,
                MAX(CASE WHEN ek.metric_name = 'events' THEN ek.metric_value END) as total_events,
                MAX(CASE WHEN ek.metric_name = 'runtime_seconds' THEN ek.metric_value END) as runtime_seconds,
                
                -- Daten-Statistiken
                COUNT(cs.Kunde) as total_customers,
                SUM(cs.event) as actual_events,
                COUNT(cs.Kunde) - SUM(cs.event) as censored_customers,
                ROUND(AVG(cs.duration), 2) as avg_survival_months,
                
                -- Risiko-Verteilung (aus cox_survival_enhanced)
                COUNT(CASE WHEN cse.risk_category = 'Sehr Hoch' THEN 1 END) as customers_very_high_risk,
                COUNT(CASE WHEN cse.risk_category = 'Hoch' THEN 1 END) as customers_high_risk,
                COUNT(CASE WHEN cse.risk_category = 'Mittel' THEN 1 END) as customers_medium_risk,
                COUNT(CASE WHEN cse.risk_category = 'Niedrig' THEN 1 END) as customers_low_risk,
                COUNT(CASE WHEN cse.risk_category = 'Sehr Niedrig' THEN 1 END) as customers_very_low_risk
                
            FROM experiments e
            LEFT JOIN experiment_kpis ek ON e.experiment_id = ek.experiment_id
            LEFT JOIN cox_survival cs ON e.experiment_id = cs.id_experiments  
            LEFT JOIN cox_survival_enhanced cse ON e.experiment_id = cse.id_experiments AND cs.Kunde = cse.Kunde
            WHERE e.model_type LIKE '%cox%'
            GROUP BY e.experiment_id, e.experiment_name, e.model_type, e.created_at
            """)
            
            print("🔬 Cox Survival Analysis Funktionen registriert:")
            print("   - cox_survival_enhanced VIEW (mit 6/12/18/24-Monats-Horizonten)")
            print("   - customer_risk_profile VIEW (Customer-Risk-Profiling)")  
            print("   - cox_performance_summary VIEW (Performance-Übersicht)")
            print("   📊 Survival-Wahrscheinlichkeiten für 6, 12, 18, 24 Monate verfügbar")

        except Exception as e:
            print(f"⚠️ Cox-Funktionen-Registrierung teilweise fehlgeschlagen: {e}")
            # Weiter fortfahren, auch wenn einzelne Views fehlschlagen
    
    def _register_cf_views(self, con) -> None:
        """
        Registriert Counterfactuals (CF) Artefakte als SQL-Views:
        - cf_individual, cf_aggregate, cf_individual_raw, cf_aggregate_raw

        Quelle bevorzugt: JSON-Database Tabellen (falls vorhanden):
          cf_individual, cf_aggregate, cf_individual_raw, cf_aggregate_raw
        Fallback-Quelle: artifacts/<experiment_id>/counterfactuals/*.json
        Anreicherung (Fallback): 'experiment_id' wird pro Record gesetzt
        """
        try:
            from pathlib import Path
            import pandas as pd  # type: ignore
            import json

            # 1) Bevorzugt: Aus JSON-DB Tabellen registrieren
            try:
                tables = self.db.data.get("tables", {})
                db_map = {
                    "cf_individual": tables.get("cf_individual", {}).get("records", []) or [],
                    "cf_aggregate": tables.get("cf_aggregate", {}).get("records", []) or [],
                    "cf_individual_raw": tables.get("cf_individual_raw", {}).get("records", []) or [],
                    "cf_aggregate_raw": tables.get("cf_aggregate_raw", {}).get("records", []) or [],
                }
                if any(len(v) > 0 for v in db_map.values()):
                    for view_name, recs in db_map.items():
                        if not recs:
                            continue
                        try:
                            df = pd.DataFrame(recs)
                            con.register(view_name, df)
                        except Exception:
                            continue
                    print("🧩 Counterfactuals-Views aus JSON-DB registriert: cf_individual, cf_aggregate, cf_individual_raw, cf_aggregate_raw")
                    return  # DB-first, kein Fallback nötig
            except Exception:
                pass

            # 2) Fallback: Artefakte aus dem artifacts/-Verzeichnis
            base = ProjectPaths.artifacts_directory()
            if not base.exists():
                return

            def load_cf_for_exp(exp_id: int) -> dict:
                cf_dir = base / str(int(exp_id)) / "counterfactuals"
                out = {}
                if cf_dir.exists():
                    for name in ("cf_individual.json", "cf_aggregate.json", "cf_individual_raw.json", "cf_aggregate_raw.json"):
                        p = cf_dir / name
                        if p.exists():
                            try:
                                recs = json.loads(p.read_text(encoding="utf-8"))
                                if isinstance(recs, list):
                                    # experiment_id anreichern
                                    for r in recs:
                                        if isinstance(r, dict):
                                            r.setdefault("experiment_id", int(exp_id))
                                    out[name] = recs
                            except Exception:
                                pass
                return out

            # Kandidaten-Experimente aus JSON-DB (falls vorhanden) oder aus Verzeichnis ableiten
            exp_ids = []
            try:
                exp_tbl = self.db.data.get("tables", {}).get("experiments", {}).get("records", []) or []
                exp_ids = sorted({int(e.get("experiment_id")) for e in exp_tbl if e.get("experiment_id") is not None})
            except Exception:
                exp_ids = []
            if not exp_ids:
                try:
                    exp_ids = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
                except Exception:
                    exp_ids = []

            # Akkumulieren
            acc = {
                "cf_individual.json": [],
                "cf_aggregate.json": [],
                "cf_individual_raw.json": [],
                "cf_aggregate_raw.json": []
            }
            for eid in exp_ids:
                loaded = load_cf_for_exp(eid)
                for k in acc.keys():
                    acc[k].extend(loaded.get(k, []))

            # DataFrames registrieren (nur wenn Daten vorhanden)
            reg_map = {
                "cf_individual": acc["cf_individual.json"],
                "cf_aggregate": acc["cf_aggregate.json"],
                "cf_individual_raw": acc["cf_individual_raw.json"],
                "cf_aggregate_raw": acc["cf_aggregate_raw.json"],
            }
            for view_name, recs in reg_map.items():
                if not recs:
                    continue
                try:
                    df = pd.DataFrame(recs)
                    con.register(view_name, df)
                except Exception:
                    continue

            # Hinweis für UI-Log
            if any(len(v) > 0 for v in reg_map.values()):
                print("🧩 Counterfactuals-Views registriert: cf_individual, cf_aggregate, cf_individual_raw, cf_aggregate_raw")
        except Exception:
            # CF-Registrierung ist optional – bei Fehlern still weitermachen
            pass

    def execute_query(self, query: str, output_format: str = "dict") -> Any:
        """
        Führt eine SQL-Query aus (erzwingt DuckDB, kein Fallback)
        """
        # Query zur Historie hinzufügen
        self.history.append(query)
        
        if not _DUCKDB_AVAILABLE:
            return "❌ DuckDB nicht verfügbar. Bitte 'pip install duckdb' ausführen."
        
        try:
            result: List[Dict[str, Any]] = self._execute_with_duckdb(query)
        except Exception as e:
            return f"❌ DuckDB-Fehler: {str(e)}"
        
        # Ergebnis formatieren
        if output_format == "table":
            return self._format_as_table(result)
        elif output_format == "json":
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif output_format == "dict":
            return result  # Direkte Rückgabe der strukturierten Daten
        elif output_format == "pandas":
            return pd.DataFrame(result)
        else:  # raw
            return result
    
    def _format_as_table(self, data: List[Dict[str, Any]]) -> str:
        """
        Formatiert Daten als schöne Tabelle
        
        Args:
            data: Liste von Dictionaries
            
        Returns:
            Formatierte Tabelle als String
        """
        if not data:
            return "📭 Keine Daten gefunden"
        
        # DataFrame erstellen
        df = pd.DataFrame(data)
        
        # Spalten basierend auf display_type formatieren
        for col in df.columns:
            display_type = self._get_display_type(col)
            
            if display_type == "integer":
                df[col] = df[col].apply(lambda x: f"{int(x):,}".replace(",", ".") if pd.notna(x) else "")
            elif display_type == "decimal":
                try:
                    df[col] = df[col].astype(float)
                except Exception:
                    pass
                df[col] = df[col].apply(lambda x: f"{x:.4f}".replace(".", ",") if pd.notna(x) else "")
                # Als String konvertieren, damit tabulate die Formatierung nicht überschreibt
                df[col] = df[col].astype(str)
                # Pandas object dtype erzwingen
                df[col] = df[col].astype('object')
            elif display_type == "datetime":
                df[col] = df[col].apply(lambda x: str(x)[:19] if pd.notna(x) else "")  # Nur Datum+Zeit
            elif display_type == "json":
                df[col] = df[col].apply(lambda x: str(x)[:30] + "..." if pd.notna(x) and len(str(x)) > 30 else str(x))
            elif display_type == "list":
                df[col] = df[col].apply(lambda x: str(x)[:30] + "..." if pd.notna(x) and len(str(x)) > 30 else str(x))
            else:  # text oder unbekannt
                def _safe_to_str(x):
                    try:
                        import numpy as np  # type: ignore
                        if x is None:
                            return ""
                        if isinstance(x, (list, dict)):
                            return str(x)
                        if isinstance(x, np.ndarray):
                            return str(x.tolist())
                        return str(x)
                    except Exception:
                        return str(x)
                df[col] = df[col].apply(_safe_to_str)
        
        # Tabelle formatieren mit expliziten Spaltennamen
        table = tabulate(
            df,  # Vollständiger DataFrame
            headers=df.columns.tolist(),  # Explizite Spaltennamen
            tablefmt='grid',
            showindex=False,
            numalign='right',
            stralign='left',
            disable_numparse=True  # Verhindert, dass tabulate Zahlen als Float interpretiert
        )
        
        # Statistiken hinzufügen
        stats = f"\n📊 Statistiken: {len(data)} Zeilen, {len(df.columns)} Spalten"
        
        return table + stats
    
    def profile_rawdata(self) -> List[Dict[str, Any]]:
        """
        Erstellt ein einfaches Spaltenprofil der Tabelle 'rawdata': Spaltenname, Nicht-Null-Zähler,
        Gesamtzeilen, Missing-Quote, abgeleiteter Pandas-Datentyp.
        """
        try:
            tbl = self.db.data.get("tables", {}).get("rawdata", {})
            records = tbl.get("records", []) or []
            if not records:
                return []
            import pandas as pd  # type: ignore
            df = pd.DataFrame(records)
            # 'features' ist JSON – nicht mitprofilieren
            if "features" in df.columns:
                try:
                    df = df.drop(columns=["features"])
                except Exception:
                    pass
            total = len(df)
            out: List[Dict[str, Any]] = []
            for col in df.columns:
                non_null = int(df[col].notna().sum())
                missing = total - non_null
                missing_rate = round((missing / total) * 100.0, 2) if total > 0 else 0.0
                out.append({
                    "column": col,
                    "non_null": non_null,
                    "rows": total,
                    "missing_%": missing_rate,
                    "dtype": str(df[col].dtype)
                })
            # Sortiere: erst systematische Felder nach vorne
            priority = {"Kunde": 0, "I_TIMEBASE": 1, "id_files": 2, "dt_inserted": 3}
            out.sort(key=lambda r: (priority.get(str(r.get("column")), 9999), str(r.get("column"))))
            return out
        except Exception:
            return []
    
    def show_tables(self) -> str:
        """Zeigt alle verfügbaren Tabellen"""
        tables = list(self.db.data["tables"].keys())
        
        table_info = []
        for table_name in tables:
            table_data = self.db.data["tables"][table_name]
            # Verwende nur 'data' als einheitliches Feld
            record_count = len(table_data.get("records", []))
            description = table_data.get("description", "")
            
            table_info.append({
                "Tabelle": table_name,
                "Records": record_count,
                "Beschreibung": description
            })
        
        return self._format_as_table(table_info)
    
    def describe_table(self, table_name: str) -> str:
        """Zeigt Schema einer Tabelle"""
        if table_name not in self.db.data["tables"]:
            return f"❌ Tabelle '{table_name}' nicht gefunden"
        
        # Verwende nur 'records' als einheitliches Feld
        table_data = self.db.data["tables"][table_name]
        records = table_data.get("records", [])
        
        if not records:
            return f"📭 Tabelle '{table_name}' ist leer"
        
        # Schema aus erstem Record ableiten
        sample_record = records[0]
        schema = []
        
        for field, value in sample_record.items():
            field_type = type(value).__name__
            if isinstance(value, (int, float)):
                field_type = "NUMERIC"
            elif isinstance(value, str):
                field_type = "TEXT"
            elif isinstance(value, bool):
                field_type = "BOOLEAN"
            elif isinstance(value, dict):
                field_type = "JSON"
            
            schema.append({
                "Feld": field,
                "Typ": field_type,
                "Beispiel": str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            })
        
        return self._format_as_table(schema)
    
    def get_query_history(self) -> List[str]:
        """Gibt Query-Historie zurück"""
        return self.history
    
    def clear_history(self):
        """Löscht Query-Historie"""
        self.history = []
    
    def _get_display_type(self, column_name: str) -> str:
        """
        Ermittelt den display_type für eine Spalte
        
        Args:
            column_name: Name der Spalte
            
        Returns:
            display_type oder "text" als Fallback
        """
        # Durch alle Tabellen gehen und nach der Spalte suchen
        for table_name, table_data in self.db.data["tables"].items():
            schema = table_data.get("schema", {})
            if column_name in schema:
                return schema[column_name].get("display_type", "text")
        
        # Fallback: Basierend auf Spaltenname raten
        if column_name.lower() in ['id', 'kunde', 'experiment_id', 'kpi_id']:
            return "integer"
        elif column_name.lower() in ['probability', 'value', 'score']:
            return "decimal"
        elif column_name.lower() in ['date', 'time', 'dt_']:
            return "datetime"
        else:
            return "text"


def interactive_query_session():
    """
    Interaktive Query-Session
    """
    interface = SQLQueryInterface()
    
    print("🚀 SQL-ähnliches Query-Interface für Churn JSON-Datenbank")
    print("=" * 60)
    print("Verfügbare Befehle:")
    print("  - SQL-Query eingeben (z.B. SELECT * FROM backtest_results)")
    print("  - \\tables - Zeigt alle Tabellen")
    print("  - \\describe <table> - Zeigt Tabellen-Schema")
    print("  - \\history - Zeigt Query-Historie")
    print("  - \\clear - Löscht Historie")
    print("  - \\quit - Beendet Session")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
                
            # Spezielle Befehle
            if query.lower() == "\\quit":
                print("👋 Session beendet")
                break
            elif query.lower() == "\\tables":
                print(interface.show_tables())
                continue
            elif query.lower() == "\\history":
                history = interface.get_query_history()
                if history:
                    print("\n📜 Query-Historie:")
                    for i, q in enumerate(history, 1):
                        print(f"  {i}. {q}")
                else:
                    print("📭 Keine Queries in Historie")
                continue
            elif query.lower() == "\\clear":
                interface.clear_history()
                print("🗑️ Historie gelöscht")
                continue
            elif query.lower().startswith("\\describe "):
                table_name = query[10:].strip()
                print(interface.describe_table(table_name))
                continue
            elif query.lower() == "\\raw_profile":
                prof = interface.profile_rawdata()
                print(interface._format_as_table(prof))
                continue
            elif query.lower().startswith("\\pivot_case "):
                # Syntax: \pivot_case <target_yyyymm> [--years N] [--month MM] [--scope same-file|all] [--threshold optimal|std05|f1|precision|recall] [--base churned|all] [--save-table NAME]
                try:
                    parts = query.split()
                    # Mindestargumente prüfen
                    if len(parts) < 2:
                        print("❌ Verwendung: \\pivot_case <target_yyyymm> [--years N] [--month MM] [--scope same-file|all] [--threshold optimal|std05|f1|precision|recall] [--base churned|all] [--save-table NAME]")
                        continue
                    # Defaults
                    target_yyyymm = int(parts[1])
                    years = 2
                    month = 12
                    scope = "same-file"
                    threshold = "optimal"
                    save_table: Optional[str] = None
                    base = "churned"
                    # Flags parsen
                    i = 2
                    while i < len(parts):
                        token = parts[i]
                        if token == "--years" and i + 1 < len(parts):
                            try:
                                years = int(parts[i + 1])
                            except Exception:
                                pass
                            i += 2
                            continue
                        if token == "--month" and i + 1 < len(parts):
                            try:
                                month = int(parts[i + 1])
                            except Exception:
                                pass
                            i += 2
                            continue
                        if token == "--scope" and i + 1 < len(parts):
                            val = parts[i + 1].lower()
                            if val in ("same-file", "all"):
                                scope = val
                            i += 2
                            continue
                        if token == "--threshold" and i + 1 < len(parts):
                            val = parts[i + 1].lower()
                            if val in ("optimal", "std05", "f1", "precision", "recall"):
                                threshold = val
                            i += 2
                            continue
                        if token == "--save-table" and i + 1 < len(parts):
                            save_table = parts[i + 1]
                            i += 2
                            continue
                        if token == "--base" and i + 1 < len(parts):
                            val = parts[i + 1].lower()
                            if val in ("churned", "all"):
                                base = val
                            i += 2
                            continue
                        i += 1
                    # SQL erzeugen
                    sql = interface._build_pivot_case_sql(
                        target_yyyymm=target_yyyymm,
                        years=years,
                        month=month,
                        scope=scope,
                        threshold=threshold,
                        base=base
                    )
                    print("\n🧩 Generiertes SQL:\n" + sql)
                    # Ausführen (roh für Persistenz)
                    try:
                        raw_records = interface._execute_with_duckdb(sql)
                    except Exception as e:
                        print(f"❌ DuckDB-Fehler: {e}")
                        continue
                    # Standard: als Tabelle in JSON-DB speichern (für Management Studio sichtbar)
                    if not save_table:
                        save_table = f"pivot_case_{target_yyyymm}"
                    if save_table:
                        # Schema heuristisch ableiten
                        schema: Dict[str, Dict[str, str]] = {}
                        if raw_records:
                            sample = raw_records[0]
                            for k, v in sample.items():
                                if isinstance(v, bool):
                                    display_type = "text"
                                elif isinstance(v, int):
                                    display_type = "integer"
                                elif isinstance(v, float):
                                    display_type = "decimal"
                                else:
                                    display_type = "text"
                                schema[k] = {"display_type": display_type, "description": ""}
                        interface.db.data["tables"][save_table] = {
                            "description": "Pivot CASE Ergebnisse (dynamisch generiert)",
                            "source": "sql_query_interface",
                            "metadata": {
                                "generated_by": "pivot_case",
                                "target_yyyymm": target_yyyymm,
                                "years": years,
                                "month": month,
                                "scope": scope,
                                "threshold": threshold,
                                "created_at": datetime.now().isoformat()
                            },
                            "schema": schema,
                            "records": raw_records
                        }
                        interface.db.save()
                        print(f"💾 Ergebnis als Tabelle '{save_table}' gespeichert (Management Studio sichtbar)")
                        # CLI-Run protokollieren
                        try:
                            interface._record_cli_run(
                                procedure="pivot_case",
                                params={
                                    "target_yyyymm": target_yyyymm,
                                    "years": years,
                                    "month": month,
                                    "scope": scope,
                                    "threshold": threshold,
                                    "base": base
                                },
                                table_name=save_table
                            )
                        except Exception:
                            pass
                    # Ausgabe im Editor hübsch formatiert
                    print(interface._format_as_table(raw_records))
                except Exception as e:
                    print(f"❌ Fehler bei \\pivot_case: {e}")
                continue
            
            # Normale Query ausführen
            print("\n" + "=" * 60)
            result = interface.execute_query(query)
            print(result)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Session beendet")
            break
        except Exception as e:
            print(f"❌ Fehler: {str(e)}")


if __name__ == "__main__":
    interactive_query_session()
