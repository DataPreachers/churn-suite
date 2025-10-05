#!/usr/bin/env python3
"""
🔧 Churn Data Loader - Extrahiert aus enhanced_early_warning.py

Verantwortlichkeiten:
- Data Loading & Preprocessing
- CSV Metadata Management
- DataAccessLayer Integration
- Data Quality Validation
- Training/Test Split Management

REENGINEERING: Erste Extraktion aus 2957-Zeilen Monster-Modul
BASELINE: QS-validiert gegen Experiment ID 1 (AUC 0.994, 4,742 Predictions)
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Project imports - korrigiertes Path Setup
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.paths_config import ProjectPaths
from config.data_access_layer import get_data_access
from config.data_schema import get_data_schema

class ChurnDataLoader:
    """
    Data Loading & Management für Churn Prediction System
    
    Extrahiert aus enhanced_early_warning.py für bessere Modularität
    """
    
    def __init__(self, experiment_id: Optional[int] = None):
        self.paths = ProjectPaths()
        self.experiment_id = experiment_id
        
        # DataSchema Architecture Integration
        self.dal = get_data_access()
        self.schema = get_data_schema()
        
        # Cache für geladene Daten
        self.df = None
        self.data_dictionary = None
        
        print("📊 Churn Data Loader initialisiert")
        
    def load_stage0_data(self) -> pd.DataFrame:
        """
        Lädt Stage0-Daten über DataAccessLayer
        
        MIGRATION: Verwendet DataAccessLayer statt direkter CSV/JSON-Zugriffe
        
        Returns:
            pd.DataFrame: Geladene und validierte Daten
        """
        print("🔧 Lade Daten über DataAccessLayer...")
        
        try:
            # Verwende DataAccessLayer für automatisch validierte Daten
            self.df = self.dal.load_stage0_data()
            print(f"✅ Daten über DataAccessLayer geladen: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")
            
            # Data Dictionary laden
            self.data_dictionary = self.schema.data_dictionary
            print(f"✅ Data Dictionary über DataSchema geladen: {len(self.data_dictionary.get('columns', {}))} Features")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Fehler beim Laden über DataAccessLayer: {e}")
            raise ValueError(f"Datenladeung über DataSchema fehlgeschlagen: {e}")
    
    def load_csv_with_metadata(self, csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Lädt CSV-Datei mit Metadaten-Tracking
        
        Args:
            csv_path: Pfad zur CSV-Datei (optional, nutzt ProjectPaths als Default)
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (DataFrame, CSV-Metadaten)
        """
        if csv_path is None:
            csv_path = ProjectPaths.get_input_data_path()
        
        print(f"📊 Lade CSV-Datei: {csv_path}")
        
        # CSV-Metadaten erfassen
        metadata = self.get_file_metadata(csv_path)
        if not metadata:
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")
        
        # CSV laden
        try:
            # Versuche verschiedene Separatoren
            try:
                df = pd.read_csv(csv_path, sep=';')
            except:
                df = pd.read_csv(csv_path, sep=',')
                
            print(f"✅ CSV geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            
            # Cache aktualisieren
            self.df = df
            
            return df, metadata
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der CSV-Datei: {e}")
            raise ValueError(f"CSV-Datei konnte nicht geladen werden: {e}")
    
    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Ermittelt Metadaten einer Datei für Tracking und Versioning
        
        Args:
            file_path: Pfad zur Datei
        
        Returns:
            Dict mit Metadaten oder None falls Datei nicht existiert
        """
        if not os.path.exists(file_path):
            return None
            
        stat = os.stat(file_path)
        return {
            'filename': os.path.basename(file_path),
            'file_path': file_path,
            'creation_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'file_size': stat.st_size,
            'last_accessed': stat.st_atime
        }
    
    def csv_has_changed(self, stored_metadata: Dict[str, Any], 
                       current_path: Optional[str] = None) -> bool:
        """
        Prüft ob sich eine CSV-Datei seit dem letzten Training geändert hat
        
        Args:
            stored_metadata: Gespeicherte Metadaten aus vorherigem Training
            current_path: Aktueller CSV-Pfad (optional)
        
        Returns:
            bool: True wenn sich Datei geändert hat oder Metadaten fehlen
        """
        if not stored_metadata:
            return True  # Keine Metadaten = neu laden
        
        if current_path is None:
            current_path = ProjectPaths.get_input_data_path()
        
        current_metadata = self.get_file_metadata(current_path)
        if not current_metadata:
            return True  # Datei nicht gefunden = neu laden
        
        # Vergleiche kritische Metadaten
        size_changed = current_metadata['file_size'] != stored_metadata.get('file_size', 0)
        time_changed = current_metadata['modified_time'] != stored_metadata.get('modified_time', 0)
        
        if size_changed or time_changed:
            print(f"📊 CSV-Datei hat sich geändert:")
            if size_changed:
                print(f"   Größe: {stored_metadata.get('file_size')} → {current_metadata['file_size']}")
            if time_changed:
                old_time = datetime.fromtimestamp(stored_metadata.get('modified_time', 0))
                new_time = datetime.fromtimestamp(current_metadata['modified_time'])
                print(f"   Änderungszeit: {old_time} → {new_time}")
            return True
        
        return False
    
    def calculate_data_hash(self, df: Optional[pd.DataFrame] = None) -> str:
        """
        Berechnet Hash für Daten-Konsistenz-Prüfung
        
        Args:
            df: DataFrame (optional, nutzt self.df als Default)
        
        Returns:
            str: SHA256-Hash der Datenstruktur
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("Keine Daten für Hash-Berechnung verfügbar")
        
        try:
            # Sample für Hash-Berechnung (erste 1000 Zeilen)
            df_sample = df.head(1000)
            
            # Struktur-Info erstellen
            structure_info = {
                'columns': list(df_sample.columns),
                'dtypes': {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
                'shape': df_sample.shape,
                'sample_values': df_sample.head(5).to_dict()
            }
            
            # Hash berechnen
            structure_str = json.dumps(structure_info, sort_keys=True)
            file_hash = hashlib.sha256(structure_str.encode()).hexdigest()
            
            return file_hash
            
        except Exception as e:
            print(f"❌ Fehler bei Hash-Berechnung: {e}")
            return "unknown_hash"
    
    def get_target_column_name(self) -> str:
        """
        Ermittelt Target-Spaltenname aus Data Dictionary
        
        Returns:
            str: Name der Target-Spalte ('I_Alive' als Fallback)
        """
        if self.data_dictionary and 'columns' in self.data_dictionary:
            for col_name, col_info in self.data_dictionary['columns'].items():
                if col_info.get('role') == 'TARGET':
                    return col_name
        
        # Fallback für Rückwärtskompatibilität
        return 'I_Alive'
    
    def get_dynamic_column_names(self) -> Dict[str, str]:
        """
        Ermittelt dynamische Spaltennamen aus Data Dictionary
        
        Returns:
            Dict: Mapping von Rollen zu Spaltennamen
        """
        if not self.data_dictionary or 'columns' not in self.data_dictionary:
            return {}
        
        dynamic_columns = {}
        
        for col_name, col_info in self.data_dictionary['columns'].items():
            role = col_info.get('role')
            if role:
                dynamic_columns[role] = col_name
        
        return dynamic_columns
    
    def validate_data_quality(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Führt Data Quality Checks durch
        
        Args:
            df: DataFrame zu validieren (optional)
        
        Returns:
            Dict: Validation-Ergebnisse
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("Keine Daten für Validation verfügbar")
        
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'duplicates': 0,
            'validation_status': 'PASS'
        }
        
        try:
            # Missing Values Check
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                validation_results['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
            
            # Data Types
            validation_results['data_types'] = {
                col: str(dtype) for col, dtype in df.dtypes.items()
            }
            
            # Duplicates
            validation_results['duplicates'] = int(df.duplicated().sum())
            
            # Target Column Check
            target_col = self.get_target_column_name()
            if target_col not in df.columns:
                validation_results['validation_status'] = 'FAIL'
                validation_results['errors'] = [f"Target-Spalte '{target_col}' nicht gefunden"]
            
            print(f"📊 Data Quality Check: {validation_results['validation_status']}")
            print(f"   Zeilen: {validation_results['total_rows']}")
            print(f"   Spalten: {validation_results['total_columns']}")
            print(f"   Duplikate: {validation_results['duplicates']}")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Fehler bei Data Quality Check: {e}")
            validation_results['validation_status'] = 'ERROR'
            validation_results['error_message'] = str(e)
            return validation_results
    
    def get_training_test_split(self, training_from: str, training_to: str,
                               test_from: str, test_to: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Erstellt zeitbasierte Training/Test-Trennung
        
        Args:
            training_from: Start Training (YYYYMM)
            training_to: Ende Training (YYYYMM)
            test_from: Start Test (YYYYMM)
            test_to: Ende Test (YYYYMM)
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (Training-Daten, Test-Daten)
        """
        if self.df is None:
            raise ValueError("Daten müssen zuerst geladen werden")
        
        # Timebase-Spalte ermitteln
        dynamic_columns = self.get_dynamic_column_names()
        timebase_col = dynamic_columns.get('TIMEBASE', 'I_TIMEBASE')
        
        if timebase_col not in self.df.columns:
            raise ValueError(f"Timebase-Spalte '{timebase_col}' nicht gefunden")
        
        # Training-Daten filtern
        training_mask = (
            (self.df[timebase_col] >= int(training_from)) & 
            (self.df[timebase_col] <= int(training_to))
        )
        training_df = self.df[training_mask].copy()
        
        # Test-Daten filtern
        test_mask = (
            (self.df[timebase_col] >= int(test_from)) & 
            (self.df[timebase_col] <= int(test_to))
        )
        test_df = self.df[test_mask].copy()
        
        print(f"📊 Zeitbasierte Trennung:")
        print(f"   Training: {training_from}-{training_to} → {len(training_df)} Zeilen")
        print(f"   Test: {test_from}-{test_to} → {len(test_df)} Zeilen")
        
        return training_df, test_df
    
    def load_step0_cache(self) -> Optional[pd.DataFrame]:
        """
        Lädt Daten aus Step0 Cache falls verfügbar
        
        Returns:
            pd.DataFrame oder None falls Cache nicht verfügbar
        """
        try:
            step0_dir = self.paths.dynamic_system_outputs_directory() / "stage0_cache"
            if not step0_dir.exists():
                print("📊 Step0 Cache nicht gefunden")
                return None
                
            json_files = list(step0_dir.glob("*.json"))
            if not json_files:
                print("📊 Keine Step0 JSON-Dateien gefunden")
                return None
            
            # Verwende die neueste Datei
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Lade Step0 Cache: {latest_file.name}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                step0_data = json.load(f)
            
            # Extrahiere Records
            records = step0_data.get('records', [])
            if not records:
                print("❌ Keine Records in Step0 Cache gefunden")
                return None
            
            df = pd.DataFrame(records)
            print(f"✅ Step0 Cache geladen: {len(df)} Zeilen")
            
            return df
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Step0 Cache: {e}")
            return None
    
    def create_integrated_data_dictionary(self) -> Dict[str, Any]:
        """
        Erstellt integriertes Data Dictionary für Feature Engineering
        
        Returns:
            Dict: Erweiterte Data Dictionary mit Feature-Klassifikationen
        """
        if not self.data_dictionary:
            raise ValueError("Data Dictionary nicht verfügbar")
        
        integrated_dict = self.data_dictionary.copy()
        
        # Stelle sicher, dass alle Spalten klassifiziert sind
        if self.df is not None:
            for col in self.df.columns:
                if col not in integrated_dict.get('columns', {}):
                    # Automatische Klassifikation für fehlende Spalten
                    feature_type, data_type = self._intelligently_classify_feature(col)
                    
                    integrated_dict.setdefault('columns', {})[col] = {
                        'feature_type': feature_type,
                        'data_type': data_type,
                        'auto_classified': True
                    }
        
        return integrated_dict
    
    def _intelligently_classify_feature(self, col: str) -> Tuple[str, str]:
        """
        Intelligente Feature-Klassifikation basierend auf Spaltenname und Datentyp
        
        Args:
            col: Spaltenname
        
        Returns:
            Tuple[str, str]: (feature_type, data_type)
        """
        if self.df is None:
            return 'numerical', 'unknown'
        
        # Basis-Klassifikation anhand Spaltenname
        col_lower = col.lower()
        
        if col_lower.startswith(('i_', 'n_')):
            feature_type = 'numerical'
        elif col_lower.startswith('t_'):
            feature_type = 'categorical'
        elif 'date' in col_lower or 'time' in col_lower:
            feature_type = 'temporal'
        else:
            # Fallback basierend auf Datentyp
            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                feature_type = 'numerical'
            else:
                feature_type = 'categorical'
        
        # Data Type bestimmung
        dtype = self.df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            data_type = 'integer'
        elif pd.api.types.is_float_dtype(dtype):
            data_type = 'float'
        elif pd.api.types.is_bool_dtype(dtype):
            data_type = 'boolean'
        else:
            data_type = 'object'
        
        return feature_type, data_type
    
    def get_baseline_validation_data(self, experiment_id: int = 1) -> Dict[str, Any]:
        """
        Lädt Baseline-Daten für QS-Validation
        
        Args:
            experiment_id: Experiment ID für Baseline (default: 1)
        
        Returns:
            Dict: Baseline-Konfiguration für Validation
        """
        return {
            'experiment_id': experiment_id,
            'training_from': '201712',
            'training_to': '202407',
            'backtest_from': '202408',
            'backtest_to': '202412',
            'expected_predictions': 4742,
            'baseline_metrics': {
                'auc': 0.994,
                'precision': 0.5163,
                'recall': 1.0,
                'f1_score': 0.681
            }
        }

def main():
    """Test des ChurnDataLoader"""
    print("🔧 === CHURN DATA LOADER TEST ===")
    
    loader = ChurnDataLoader()
    
    # Test 1: Stage0 Data Loading
    try:
        df = loader.load_stage0_data()
        print(f"✅ Stage0 Daten geladen: {len(df)} Zeilen")
    except Exception as e:
        print(f"❌ Stage0 Loading fehlgeschlagen: {e}")
    
    # Test 2: Data Quality Validation
    try:
        validation = loader.validate_data_quality()
        print(f"✅ Data Quality Check: {validation['validation_status']}")
    except Exception as e:
        print(f"❌ Data Quality Check fehlgeschlagen: {e}")
    
    # Test 3: Baseline Validation Data
    baseline = loader.get_baseline_validation_data()
    print(f"✅ Baseline Validation Config: {baseline['expected_predictions']} Predictions")

if __name__ == "__main__":
    main()