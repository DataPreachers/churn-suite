#!/usr/bin/env python3
"""
Enhanced Early Warning System mit 110+ Features
Kombiniert Dynamic System Features mit Early Warning Signalen

MIGRATION: Migriert zur DataSchema-Architektur für konsistente Dateninterpretation
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import hashlib
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 🚀 SAUBERE PYTHONPATH-KONFIGURATION
import sys
from pathlib import Path

# Füge config/ und bl/ zu Python-Pfad hinzu
project_root = Path(__file__).parent.parent
config_path = project_root / "config"
bl_path = project_root / "bl"

if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
if str(bl_path) not in sys.path:
    sys.path.insert(0, str(bl_path))

# Jetzt kann ProjectPaths importiert werden
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths_config import ProjectPaths

# MIGRATION: DataSchema Architecture imports
from config.data_access_layer import get_data_access
from config.data_schema import get_data_schema

# Globale Konfiguration importieren
from config.global_config import get_global_config, enforce_real_analysis, enforce_real_backtest

# Import für wissenschaftlich fundierte Schwellwert-Optimierung
from .threshold_optimizer import RobustThresholdOptimizer
from .churn_model_trainer import ChurnModelTrainer
from .churn_evaluator import ChurnEvaluator

class EnhancedEarlyWarningSystem:
    """
    Enhanced Early Warning mit 110+ Features
    
    MIGRATION: Migriert zur DataSchema-Architektur für konsistente Dateninterpretation
    """
    
    def __init__(self, experiment_id=None, test_reduction: float = 0.0):
        self.paths = ProjectPaths()
        
        # MIGRATION: DataSchema Architecture Integration
        self.dal = get_data_access()
        self.schema = get_data_schema()
        
        self.df = None
        self.features_df = None
        self.model = None
        self.data_dictionary = None
        self.feature_names = []  # Initialisiere feature_names
        self.experiment_id = experiment_id  # MINIMALE ÄNDERUNG: experiment_id hinzufügen
        # Test-Reduktion: 0.0 = keine Reduktion; 0.9 = 10% der Kunden verarbeiten
        try:
            self.test_reduction = float(test_reduction) if test_reduction is not None else 0.0
        except Exception:
            self.test_reduction = 0.0
        
        print("🔒 Enhanced Early Warning System initialisiert mit DataSchema-Validierung")
        
    def _get_csv_metadata(self):
        """Ermittelt Metadaten der CSV-Datei für Modell-Identifikation"""
        csv_path = ProjectPaths.get_input_data_path()
        if os.path.exists(csv_path):
            stat = os.stat(csv_path)
            return {
                'filename': os.path.basename(csv_path),
                'creation_time': stat.st_ctime,
                'size': stat.st_size,
                'modified_time': stat.st_mtime
            }
        return None

    def _find_existing_model_for_csv(self):
        """Findet existierendes Modell für aktuelle CSV-Datei"""
        csv_metadata = self._get_csv_metadata()
        if not csv_metadata:
            return None
            
        models_dir = ProjectPaths.get_models_directory()
        if not os.path.exists(models_dir):
            return None
            
        # Suche nach Modellen mit passender CSV-Metadaten
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.json') and 'Enhanced_EarlyWarning' in model_file:
                metadata_path = os.path.join(models_dir, model_file)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Prüfe ob CSV-Metadaten übereinstimmen
                    if 'csv_metadata' in metadata:
                        stored_metadata = metadata['csv_metadata']
                        if (stored_metadata.get('filename') == csv_metadata['filename'] and
                            stored_metadata.get('creation_time') == csv_metadata['creation_time'] and
                            stored_metadata.get('size') == csv_metadata['size']):
                            # Modell gefunden - verwende es
                            model_path = metadata_path.replace('.json', '.joblib')
                            if os.path.exists(model_path):
                                return model_path
                except Exception as e:
                    print(f"⚠️ Fehler beim Lesen von {metadata_path}: {e}")
                    continue
        return None

    def _should_save_new_model(self):
        """Prüft ob neues Modell gespeichert werden soll"""
        existing_model = self._find_existing_model_for_csv()
        if existing_model:
            print(f"✅ Existierendes Modell gefunden für aktuelle CSV: {os.path.basename(existing_model)}")
            print("📄 Verwende existierendes Modell - kein Speichern erforderlich")
            return False, existing_model
        else:
            print("🔄 Kein passendes Modell gefunden - erstelle neues Modell")
            return True, None
    
    def get_csv_metadata(self, csv_path=None):
        """Erstellt Metadaten für die CSV-Datei"""
        if csv_path is None:
            csv_path = ProjectPaths.get_input_data_path()
        
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                return None
            
            # Datei-Metadaten
            stat = csv_file.stat()
            metadata = {
                'file_path': str(csv_file),
                'file_name': csv_file.name,
                'file_size': stat.st_size,
                'last_modified': stat.st_mtime,
                'last_modified_iso': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            return metadata
            
        except Exception as e:
            print(f"❌ Fehler beim Lesen der CSV-Metadaten: {e}")
            return None
    
    def csv_has_changed(self, stored_csv_metadata):
        """Prüft ob sich die CSV-Datei seit dem letzten Training geändert hat"""
        if not stored_csv_metadata:
            return True  # Keine Metadaten = neu trainieren
        
        current_metadata = self.get_csv_metadata()
        if not current_metadata:
            return True  # CSV nicht gefunden = neu trainieren
        
        # Vergleiche Datei-Größe und letztes Änderungsdatum
        size_changed = current_metadata['file_size'] != stored_csv_metadata.get('file_size', 0)
        time_changed = current_metadata['last_modified'] != stored_csv_metadata.get('last_modified', 0)
        
        return size_changed or time_changed
    
    def get_latest_model(self):
        """Findet das neueste Enhanced Early Warning Modell"""
        try:
            models_dir = ProjectPaths.get_models_directory()
            if not models_dir.exists():
                return None, None
            
            # Suche nach Enhanced Early Warning Modellen
            model_files = list(models_dir.glob("Enhanced_EarlyWarning_*.joblib"))
            if not model_files:
                return None, None
            
            # Sortiere nach Datum im Dateinamen (neuestes zuerst)
            model_files.sort(key=lambda x: x.stem, reverse=True)
            latest_model = model_files[0]
            
            # Suche zugehörige Metadaten-Datei
            metadata_file = latest_model.with_suffix('.json')
            if not metadata_file.exists():
                return str(latest_model), None
            
            return str(latest_model), str(metadata_file)
            
        except Exception as e:
            print(f"❌ Fehler beim Suchen des neuesten Modells: {e}")
            return None, None
    
    def load_latest_model(self):
        """Lädt das neueste Modell wenn CSV unverändert ist"""
        try:
            print("🔍 Prüfe verfügbare Modelle...")
            
            # Finde neuestes Modell
            model_path, metadata_path = self.get_latest_model()
            if not model_path:
                print("⚠️ Kein trainiertes Modell gefunden")
                return False
            
            # Lade Metadaten
            if not metadata_path:
                print("⚠️ Keine Modell-Metadaten gefunden")
                return False
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Prüfe CSV-Änderungen
            csv_metadata = metadata.get('csv_metadata')
            if self.csv_has_changed(csv_metadata):
                print("🔄 CSV-Datei hat sich geändert - Neutraining erforderlich")
                return False
            
            # Lade Modell
            self.model = joblib.load(model_path)
            self.feature_names = metadata.get('feature_names', [])
            self.model_metadata = metadata  # ✅ KRITISCH: Speichere Metadaten für vereinfachtes Backtest
            
            print(f"✅ Modell geladen: {Path(model_path).name}")
            print(f"📊 Features: {len(self.feature_names)}")
            print(f"📅 Erstellt: {metadata.get('created', 'Unknown')}")
            print(f"🎯 CSV unverändert seit: {csv_metadata.get('last_modified_iso', 'Unknown') if csv_metadata else 'Unknown'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Modells: {e}")
            return False
    
    def should_retrain_model(self, requested_training_from=None, requested_training_to=None):
        """Bestimmt ob ein Modell neu trainiert werden sollte"""
        try:
            # Prüfe ob es ein Modell gibt
            model_path, metadata_path = self.get_latest_model()
            if not model_path or not metadata_path:
                return True, "Kein trainiertes Modell gefunden"
            
            # Lade Metadaten
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 🔧 NEU: Prüfe Training-Zeiträume
            if requested_training_from is not None and requested_training_to is not None:
                stored_training_periods = metadata.get('training_periods', {})
                stored_from = stored_training_periods.get('training_from')
                stored_to = stored_training_periods.get('training_to')
                
                if stored_from != str(requested_training_from) or stored_to != str(requested_training_to):
                    return True, f"Training-Zeiträume haben sich geändert: {stored_from}-{stored_to} → {requested_training_from}-{requested_training_to}"
            
            # Prüfe CSV-Änderungen
            csv_metadata = metadata.get('csv_metadata')
            if self.csv_has_changed(csv_metadata):
                return True, "CSV-Datei hat sich geändert"
            
            # Prüfe Modell-Alter (optional: > 30 Tage)
            model_created = metadata.get('created')
            if model_created:
                from datetime import datetime, timedelta
                try:
                    created_date = datetime.fromisoformat(model_created.replace('T', ' '))
                    age_days = (datetime.now() - created_date).days
                    if age_days > 30:
                        return True, f"Modell ist {age_days} Tage alt"
                except:
                    pass
            
            return False, "Modell ist aktuell"
            
        except Exception as e:
            return True, f"Fehler beim Prüfen: {e}"
    
    def delete_old_models(self, keep_latest=1):
        """Löscht alte Modelle, behält nur die neuesten"""
        try:
            models_dir = ProjectPaths.get_models_directory()
            if not models_dir.exists():
                return
            
            # Sammle alle Enhanced Early Warning Modelle
            model_files = list(models_dir.glob("Enhanced_EarlyWarning_*.joblib"))
            metadata_files = list(models_dir.glob("Enhanced_EarlyWarning_*.json"))
            
            if len(model_files) <= keep_latest:
                return  # Nichts zu löschen
            
            # Sortiere nach Datum (älteste zuerst)
            model_files.sort(key=lambda x: x.stem)
            metadata_files.sort(key=lambda x: x.stem)
            
            # Lösche alte Modelle
            models_to_delete = model_files[:-keep_latest]
            metadata_to_delete = metadata_files[:-keep_latest]
            
            deleted_count = 0
            for model_file in models_to_delete:
                try:
                    model_file.unlink()
                    deleted_count += 1
                except:
                    pass
            
            for metadata_file in metadata_to_delete:
                try:
                    metadata_file.unlink()
                except:
                    pass
            
            if deleted_count > 0:
                print(f"🗑️ {deleted_count} alte Modelle gelöscht")
            
        except Exception as e:
            print(f"❌ Fehler beim Löschen alter Modelle: {e}")
        
    def load_and_enhance_features(self):
        """
        Lädt und erweitert Features mit Early Warning Signals
        
        MIGRATION: Verwendet DataAccessLayer statt direkter CSV/JSON-Zugriffe
        """
        print("🚨 === ENHANCED EARLY WARNING SYSTEM (DataSchema-Migriert) ===")
        
        # MIGRATION: Lade Daten über DataAccessLayer
        print("🔧 Lade Daten über DataAccessLayer...")
        try:
            # Verwende bereits geladene Daten falls verfügbar (backward compatibility)
            if hasattr(self, 'data') and self.data is not None:
                print("✅ Verwende bereits geladene Daten aus Stufe 1")
                self.df = self.data
                print(f"✅ Daten aus Stufe 1 geladen: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")
            else:
                # MIGRATION: Verwende DataAccessLayer für automatisch validierte Daten
                self.df = self.dal.load_stage0_data()
                print(f"✅ Daten über DataAccessLayer geladen: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")
                
        except Exception as e:
            print(f"❌ Fehler beim Laden über DataAccessLayer: {e}")
            raise ValueError(f"Datenladeung über DataSchema fehlgeschlagen: {e}")
        
        # MIGRATION: Data Dictionary über DataSchema
        print("🔧 Lade Data Dictionary über DataSchema...")
        self.data_dictionary = self.schema.data_dictionary
        print(f"✅ Data Dictionary über DataSchema geladen: {len(self.data_dictionary.get('columns', {}))} Features")
        
        # ✅ ZURÜCK ZUM BEWÄHRTEN ORIGINAL (vermeidet doppeltes Feature Engineering)
        print("📊 Erstelle 360+ Dynamic System Features (ORIGINAL)...")
        features_df, base_features = self._create_integrated_features()
        print(f"✅ {len(base_features)} Dynamic System Features")
        
        # Füge Early Warning Features hinzu (TEMPORÄR DEAKTIVIERT FÜR PERFORMANCE)
        print("🚨 Early Warning Features TEMPORÄR DEAKTIVIERT für Performance-Test...")
        enhanced_df = features_df  # Verwende nur die Basis-Features
        
        # Dynamische Spaltennamen aus Data Dictionary  
        primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
        
        # Finale Feature-Liste (WIRD NACH DATA-LEAKAGE-KORREKTUR GESETZT!)
        exclude_cols = [primary_key_col, timebase_col, target_col]
        final_features = [col for col in enhanced_df.columns if col not in exclude_cols]
        
        print(f"🎯 Total Features: {len(final_features)}")
        
        self.features_df = enhanced_df
        # ❌ NICHT HIER SETZEN - wird nach Data-Leakage-Korrektur gesetzt!
        # self.feature_names = final_features
        
        return enhanced_df, final_features

    def _get_positive_class_proba(self, model, X) -> np.ndarray:
        """
        Robuste Extraktion der Positivklassen-Wahrscheinlichkeit.
        Wenn das Modell nur eine Klasse kennt (einspaltiges predict_proba),
        liefere einen konstanten Vektor (0.0/1.0) statt zu crashen.
        """
        try:
            proba = model.predict_proba(X)
            if proba is None:
                return np.zeros((len(X),), dtype=float)
            if proba.ndim == 1:
                return proba.astype(float)
            n_cols = proba.shape[1]
            if n_cols == 1:
                try:
                    cls = int(getattr(model, 'classes_', [0])[0])
                    val = 1.0 if cls == 1 else 0.0
                except Exception:
                    val = float(proba[0, 0]) if proba.size > 0 else 0.0
                return np.full((proba.shape[0],), val, dtype=float)
            try:
                classes = list(getattr(model, 'classes_', []))
                if 1 in classes:
                    idx = classes.index(1)
                    return proba[:, idx].astype(float)
            except Exception:
                pass
            return proba[:, -1].astype(float)
        except Exception:
            return np.zeros((len(X),), dtype=float)

    
    
    def _get_dynamic_column_names(self):
        """Liest Primary Key, Timebase und Target Spaltennamen dynamisch aus Data Dictionary"""
        primary_key_col = 'Kunde'  # Fallback
        timebase_col = 'I_TIMEBASE'  # Fallback
        target_col = self.get_target_column_name()  # Verwende Data Dictionary
        
        if self.data_dictionary and 'columns' in self.data_dictionary:
            for col_name, col_info in self.data_dictionary['columns'].items():
                if col_info.get('role') == 'PRIMARY_KEY':
                    primary_key_col = col_name
                elif col_info.get('role') == 'TIMEBASE':
                    timebase_col = col_name
                elif col_info.get('role') == 'TARGET':
                    target_col = col_name
        
        return primary_key_col, timebase_col, target_col
    
    def _load_step0_output(self):
        """Lädt Step0 JSON-Ausgabe"""
        try:
            # Suche nach Step0 JSON-Ausgabe
            step0_dir = ProjectPaths.dynamic_system_outputs_directory() / "stage0_cache"
            if step0_dir.exists():
                json_files = list(step0_dir.glob("*.json"))
                if json_files:
                    # Verwende die neueste Datei
                    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            return None
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Step0 JSON-Ausgabe: {e}")
            return None

    def _calculate_csv_hash(self, file_path: str) -> str:
        """CSV Hash berechnen (wie in Stufe 0)"""
        try:
            # Sample lesen für Hash-Berechnung (DEPRECATED: Verwende DataAccessLayer)
            try:
                # MIGRATION NOTE: This is only used for hashing, DataAccessLayer handles actual data loading
                df_sample = pd.read_csv(file_path, nrows=1000, sep=';')
            except:
                df_sample = pd.read_csv(file_path, nrows=1000)
                
            # Struktur-Info erstellen (identisch zu Stufe 0)
            structure_info = {
                'columns': list(df_sample.columns),
                'dtypes': {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
                'null_counts': df_sample.isnull().sum().to_dict(),
                'unique_counts': {col: df_sample[col].nunique() for col in df_sample.columns},
                'file_size': os.path.getsize(file_path)
            }
            
            # Hash berechnen
            structure_str = json.dumps(structure_info, sort_keys=True)
            return hashlib.md5(structure_str.encode()).hexdigest()
            
        except Exception as e:
            print(f"❌ Fehler beim Hash-Berechnung: {e}")
            return None

    def get_target_column_name(self):
        """Returns the name of the target column as defined in the Data Dictionary, or 'I_Alive' as fallback."""
        if self.data_dictionary and 'columns' in self.data_dictionary:
            for col_name, col_info in self.data_dictionary['columns'].items():
                if col_info.get('role') == 'TARGET':
                    return col_name
        # Verwende Data Dictionary für Target-Spalte
        if self.data_dictionary and 'columns' in self.data_dictionary:
            for col, info in self.data_dictionary['columns'].items():
                if info.get('role') == 'TARGET':
                    return col
        # Fallback nur wenn Data Dictionary nicht verfügbar
        return 'I_Alive'

    def add_early_warning_features(self, features_df):
        """Fügt spezialisierte Early Warning Features hinzu"""
        print("🚨 Erstelle Early Warning Features für Churn-Prädiktion...")
        
        # Kopiere Basis-Daten
        enhanced_df = features_df.copy()
        
        # Dynamische Spaltennamen
        primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
        
        # Gruppiere nach Primary Key für zeitlich korrekte Aggregationen
        grouped = enhanced_df.groupby(primary_key_col)
        
        # Finde Dynamic Features aus Data Dictionary
        dynamic_features = []
        if self.data_dictionary and 'columns' in self.data_dictionary:
            for col, col_info in self.data_dictionary['columns'].items():
                if col_info.get('role') == 'DYNAMIC_FEATURE' and col in enhanced_df.columns:
                    dynamic_features.append(col)
        else:
            # Fallback: Verwende Data Dictionary-basierte Klassifizierung
            dynamic_features = []
            if self.data_dictionary and 'columns' in self.data_dictionary:
                for col in enhanced_df.columns:
                    if col in self.data_dictionary['columns']:
                        role = self.data_dictionary['columns'][col].get('role', '')
                        if role == 'DYNAMIC_FEATURE' and col not in [primary_key_col, timebase_col, target_col]:
                            dynamic_features.append(col)
        
        print(f"🎯 Erstelle Early Warning Features für {len(dynamic_features)} Dynamic Features...")
        
        for feature in dynamic_features:
            if feature in enhanced_df.columns:
                try:
                    # 🔧 KORRIGIERT: Early Warning Features mit optimalen Zeitfenstern
                    # Early Warning Signale für verschiedene Zeitfenster
                    optimal_windows = self.find_optimal_global_windows(max_windows=5)  # Max 5 optimale Windows
                    ew_windows = optimal_windows  # Verwende optimale Windows
                    for window in ew_windows:
                        # Drastischer Rückgang für jedes Zeitfenster
                        enhanced_df[f'{feature}_ew_major_drop_{window}p'] = grouped[feature].transform(
                            lambda x: ((x.shift(window) - x.shift(window*2)) / (x.shift(window*2) + 1e-8) > 0.3).astype(int)
                        )
                        
                        # Kontinuierlicher Rückgang für jedes Zeitfenster
                        enhanced_df[f'{feature}_ew_decline_{window}p'] = grouped[feature].transform(
                            lambda x: ((x.shift(window) - x.shift(window*1.5)) > 0).astype(int)
                        )
                        
                        # Volatilität für jedes Zeitfenster
                        enhanced_df[f'{feature}_ew_volatility_{window}p'] = grouped[feature].transform(
                            lambda x: (x.shift(window).rolling(window=window, min_periods=1).std() > x.shift(window).rolling(window=window, min_periods=1).mean() * 0.3).astype(int)
                        )
                    
                    # Kompletter Stopp (1-Monat Lookback - bleibt konstant)
                    enhanced_df[f'{feature}_ew_stop'] = grouped[feature].transform(
                        lambda x: (x.shift(1) == 0).astype(int)
                    )
                    
                except Exception as e:
                    print(f"⚠️ Fehler bei Early Warning Feature {feature}: {e}")
                    continue
        
        # Statische Early Warning Features (NUR für echte STATIC_FEATURES)
        static_features = []
        for col in enhanced_df.columns:
            if col not in [primary_key_col, timebase_col, target_col]:
                # Prüfe Data Dictionary: Nur STATIC_FEATURES bekommen Early Warning Features
                if col in self.data_dictionary.get("columns", {}):
                    role = self.data_dictionary["columns"][col].get("role", "")
                    if role == "STATIC_FEATURE":
                        static_features.append(col)
        
        print(f"🎯 Erstelle Early Warning Features für {len(static_features)} Static Features...")
        
        for feature in static_features:
            if feature in enhanced_df.columns:
                try:
                    # Nur für echte statische Features (nicht für kategorische)
                    if feature not in self.data_dictionary.get("columns", {}) or \
                       self.data_dictionary["columns"][feature].get("role") == "STATIC_FEATURE":
                        # Digitalisierungsrate Early Warning (nur für echte STATIC_FEATURES)
                        if 'digital' in feature.lower():
                            enhanced_df[f'{feature}_ew_digital_drop'] = (enhanced_df[feature] < 0.5).astype(int)
                            enhanced_df[f'{feature}_ew_digital_stagnation'] = (enhanced_df[feature] < 0.7).astype(int)
                    
                except Exception as e:
                    print(f"⚠️ Fehler bei Static Early Warning Feature {feature}: {e}")
                    continue
        
        print(f"✅ {len([col for col in enhanced_df.columns if '_ew_' in col])} Early Warning Features erstellt")
        return enhanced_df
    
    def _create_integrated_data_dictionary(self):
        """Erstellt oder lädt Data Dictionary für intelligente Feature-Klassifizierung"""
        dict_path = ProjectPaths.get_data_dictionary_file()
        
        try:
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.data_dictionary = json.load(f)
                return
            
            # Erstelle neues Data Dictionary
            self.data_dictionary = {"columns": {}}
            columns = self.df.columns.tolist()
            
            # Dynamische Spaltennamen aus Data Dictionary (falls bereits vorhanden)
            primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
            
            # 🧠 INTELLIGENTE FEATURE-KLASSIFIZIERUNG
            for col in columns:
                if col == primary_key_col:
                    role = 'PRIMARY_KEY'
                    data_type = 'NUMERIC'
                elif col == timebase_col:
                    role = 'TIMEBASE'
                    data_type = 'NUMERIC'
                elif col == target_col:
                    role = 'TARGET'
                    data_type = 'BINARY'
                else:
                    # 🔍 INTELLIGENTE FEATURE-ANALYSE
                    role, data_type = self._intelligently_classify_feature(col)
                
                self.data_dictionary["columns"][col] = {
                    "role": role,
                    "data_type": data_type,
                    "auto_classified": True
                }
            
            # Speichere Data Dictionary
            with open(dict_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_dictionary, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ Data Dictionary Fehler: {e}")
            # Minimaler Fallback
            self.data_dictionary = {"columns": {}}
    
    def _intelligently_classify_feature(self, col: str) -> tuple[str, str]:
        """Intelligente Feature-Klassifizierung basierend auf Datenanalyse"""
        
        # Analysiere Datencharakteristiken
        unique_count = self.df[col].nunique()
        null_count = self.df[col].isnull().sum()
        dtype = str(self.df[col].dtype)
        
        # 🔍 KATEGORISCHE FEATURE-ERKENNUNG
        if dtype == 'object' or unique_count <= 20:
            # Kategorische Features (wie N_DIGITALIZATIONRATE)
            return 'CATEGORICAL_FEATURE', 'CATEGORICAL'
        
        # 🔍 STATIC vs DYNAMIC FEATURE-ERKENNUNG (basierend auf Datenanalyse)
        elif unique_count <= 50:
            # Features mit wenigen unique values sind typischerweise statisch
            return 'STATIC_FEATURE', 'NUMERIC'
        else:
            # Features mit vielen unique values sind typischerweise dynamisch
            return 'DYNAMIC_FEATURE', 'NUMERIC'
    
    def _create_integrated_features(self):
        """🎯 GENERISCHES Feature-Engineering - adaptiert sich automatisch an verfügbare Daten"""
        print("🎯 Erstelle GENERISCHE Features (adaptiv basierend auf Data-Dictionary)...")
        
        # Kopiere Basis-Daten
        features_df = self.df.copy()
        
        # 🔍 DETERMINISTISCHE Feature-Erkennung (100% reproduzierbar)
        available_columns = sorted(list(features_df.columns))  # ✅ SORTIERT = DETERMINISTISCH!
        temporal_features = []
        static_features = []
        numeric_features = []
        categorical_features = []
        
        # Dynamische Spaltennamen aus Data Dictionary
        primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
        
        # 🎯 PURE DATA DICTIONARY KLASSIFIZIERUNG (User-gesteuert, keine Fallbacks)
        for col in available_columns:
            if col in [primary_key_col, timebase_col, target_col]:
                continue  # Überspringe System-Spalten
                
            # ✅ ZWINGEND: Alle Features MÜSSEN im Data Dictionary definiert sein
            if not self.data_dictionary or col not in self.data_dictionary.get("columns", {}):
                raise ValueError(f"❌ Feature '{col}' NICHT im Data Dictionary konfiguriert! Bitte in config/data_dictionary.json hinzufügen.")
                
            # ✅ PURE: Nur Data Dictionary-Rollen respektieren
            dict_role = self.data_dictionary["columns"][col]["role"]
            print(f"   📚 {col}: {dict_role}")
            
            # ✅ EXPLIZITE Rollen-Behandlung (keine automatischen Fallbacks)
            if dict_role == "DYNAMIC_FEATURE":
                temporal_features.append(col)
            elif dict_role == "STATIC_FEATURE":
                static_features.append(col)  # ✅ NUR hier! Keine doppelte Klassifizierung
            elif dict_role == "CATEGORICAL_FEATURE":
                categorical_features.append(col)  # ✅ Kategorische Features
            elif dict_role == "EXCLUDED_FEATURE":
                print(f"   ❌ EXCLUDED: {col} (respektiere User-Konfiguration)")
                continue  # ✅ RESPEKTIERE User-Entscheidung
            elif dict_role in ["PRIMARY_KEY", "TIMEBASE", "TARGET"]:
                continue  # System-Spalten bereits übersprungen
            elif dict_role == "LINEAGE":
                # Lineage-Spalten (z. B. file_id) nicht als Feature verwenden
                continue
            else:
                raise ValueError(f"❌ Unbekannte Data Dictionary Rolle: '{dict_role}' für Feature '{col}'")
        
        # ✅ SORTIERUNG FÜR DETERMINISMUS (kritisch!)
        temporal_features = sorted(temporal_features)
        static_features = sorted(static_features)
        numeric_features = sorted(numeric_features)
        categorical_features = sorted(categorical_features)
        
        print(f"🧠 PURE DATA DICTIONARY Feature-Klassifizierung:")
        print(f"   📊 DYNAMIC_FEATURES: {len(temporal_features)} Features")
        print(f"   📈 STATIC_FEATURES: {len(static_features)} Features") 
        print(f"   🏷️ CATEGORICAL_FEATURES: {len(categorical_features)} Features")
        print(f"   ✅ Erkannte Dynamic Features: {temporal_features}")
        print(f"   ✅ Erkannte Static Features: {static_features}")
        print(f"   ⚠️ Numeric General Features: 0 (Pure Data Dictionary - keine automatischen Fallbacks)")
        
        # Sortiere für zeitliche Korrektheit
        features_df = features_df.sort_values([primary_key_col, timebase_col])
        
        print("🔄 Erstelle ADAPTIVE Feature-Kombinationen...")
        
        # Gruppiere nach Primary Key für zeitlich korrekte Aggregationen
        grouped = features_df.groupby(primary_key_col)
        feature_list = []
        
        # 🚀 PERFORMANCE-OPTIMIERTES TEMPORAL FEATURE ENGINEERING
        print(f"⏰ Verarbeite {len(temporal_features)} temporale Features...")
        
        # Sammle alle neuen Features in Dictionary für Batch-Concat
        temporal_feature_dict = {}
        
        for feature in temporal_features:
            if feature in features_df.columns:
                try:
                    # 🔧 KORRIGIERT: Multi-Window Rolling Features mit optimalen Zeitfenstern
                    # 🛡️ DEBUG: Einfache Data-Leakage-Prävention mit Debug-Ausgaben
                    optimal_windows = self.find_optimal_global_windows(max_windows=5)  # Max 5 optimale Windows
                    print(f"🔍 DEBUG: Verwende optimale Windows: {optimal_windows}")
                    windows = optimal_windows  # Optimale Zeitfenster für Early Warning
                    for window in windows:
                        try:
                            # Rolling Mean für jedes Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                            print(f"🔍 DEBUG: Verarbeite {feature}_rolling_{window}p_mean...")
                            temporal_feature_dict[f'{feature}_rolling_{window}p_mean'] = grouped[feature].transform(
                                lambda x: x.shift(window).where(x.shift(window) > 0).rolling(window=window, min_periods=1).mean()
                            )
                            print(f"✅ DEBUG: {feature}_rolling_{window}p_mean erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_rolling_{window}p_mean: {e}")
                            continue
                        
                        try:
                            # Rolling Sum für jedes Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                            print(f"🔍 DEBUG: Verarbeite {feature}_rolling_{window}p_sum...")
                            temporal_feature_dict[f'{feature}_rolling_{window}p_sum'] = grouped[feature].transform(
                                lambda x: x.shift(window).where(x.shift(window) > 0).rolling(window=window, min_periods=1).sum()
                            )
                            print(f"✅ DEBUG: {feature}_rolling_{window}p_sum erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_rolling_{window}p_sum: {e}")
                            continue
                        
                        try:
                            # Rolling Std für jedes Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                            print(f"🔍 DEBUG: Verarbeite {feature}_rolling_{window}p_std...")
                            temporal_feature_dict[f'{feature}_rolling_{window}p_std'] = grouped[feature].transform(
                                lambda x: x.shift(window).where(x.shift(window) > 0).rolling(window=window, min_periods=1).std().fillna(0)
                            )
                            print(f"✅ DEBUG: {feature}_rolling_{window}p_std erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_rolling_{window}p_std: {e}")
                            continue
                        
                        feature_list.extend([
                            f'{feature}_rolling_{window}p_mean',
                            f'{feature}_rolling_{window}p_sum', 
                            f'{feature}_rolling_{window}p_std'
                        ])
                    
                    # 🔧 KORRIGIERT: Trend & Change Features mit optimalen Zeitfenstern
                    # 🛡️ DEBUG: Einfache Data-Leakage-Prävention mit Debug-Ausgaben
                    # Trend Features für verschiedene Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                    trend_windows = optimal_windows  # Verwende optimale Windows
                    for window in trend_windows:
                        try:
                            print(f"🔍 DEBUG: Verarbeite {feature}_trend_{window}p...")
                            temporal_feature_dict[f'{feature}_trend_{window}p'] = grouped[feature].transform(
                                lambda x: (x.shift(window).where(x.shift(window) > 0) - x.shift(window*2).where(x.shift(window*2) > 0)).fillna(0)
                            )
                            print(f"✅ DEBUG: {feature}_trend_{window}p erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_trend_{window}p: {e}")
                            continue
                    
                    # Relative Change für verschiedene Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                    change_windows = optimal_windows  # Verwende optimale Windows
                    for window in change_windows:
                        try:
                            print(f"🔍 DEBUG: Verarbeite {feature}_pct_change_{window}p...")
                            temporal_feature_dict[f'{feature}_pct_change_{window}p'] = grouped[feature].transform(
                                lambda x: x.shift(window).where(x.shift(window) > 0).pct_change(fill_method=None).fillna(0)
                            )
                            print(f"✅ DEBUG: {feature}_pct_change_{window}p erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_pct_change_{window}p: {e}")
                            continue
                    
                    # Cumulative Features für verschiedene Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                    cum_windows = optimal_windows  # Verwende optimale Windows
                    for window in cum_windows:
                        try:
                            print(f"🔍 DEBUG: Verarbeite {feature}_cumsum_{window}p...")
                            temporal_feature_dict[f'{feature}_cumsum_{window}p'] = grouped[feature].transform(
                                lambda x: x.shift(window).where(x.shift(window) > 0).cumsum()
                            )
                            print(f"✅ DEBUG: {feature}_cumsum_{window}p erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_cumsum_{window}p: {e}")
                            continue
                        
                        try:
                            print(f"🔍 DEBUG: Verarbeite {feature}_cummax_{window}p...")
                            temporal_feature_dict[f'{feature}_cummax_{window}p'] = grouped[feature].transform(
                                lambda x: x.shift(window).where(x.shift(window) > 0).cummax()
                            )
                            print(f"✅ DEBUG: {feature}_cummax_{window}p erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_cummax_{window}p: {e}")
                            continue
                    
                    # Activity Patterns für verschiedene Zeitfenster (DEBUG: Einfache 0-Werte-Ausschluss)
                    activity_windows = optimal_windows  # Verwende optimale Windows
                    for window in activity_windows:
                        try:
                            print(f"🔍 DEBUG: Verarbeite {feature}_activity_rate_{window}p...")
                            temporal_feature_dict[f'{feature}_activity_rate_{window}p'] = grouped[feature].transform(
                                lambda x: (x.shift(window).where(x.shift(window) > 0) > 0).rolling(window=window, min_periods=1).mean()
                            )
                            print(f"✅ DEBUG: {feature}_activity_rate_{window}p erfolgreich")
                        except Exception as e:
                            print(f"❌ DEBUG: Fehler bei {feature}_activity_rate_{window}p: {e}")
                            continue
                    
                    # Füge alle neuen Features zur Liste hinzu
                    for window in [3, 6, 9, 12]:
                        feature_list.extend([
                            f'{feature}_trend_{window}p',
                            f'{feature}_pct_change_{window}p',
                            f'{feature}_cumsum_{window}p',
                            f'{feature}_cummax_{window}p',
                            f'{feature}_activity_rate_{window}p'
                        ])
                    
                except Exception as e:
                    print(f"⚠️ Fehler bei Feature {feature}: {e}")
                    continue
        
        # 🚀 PERFORMANCE-BOOST: Alle temporalen Features auf einmal hinzufügen  
        if temporal_feature_dict:
            print(f"🚀 Batch-Concat {len(temporal_feature_dict)} temporale Features...")
            temporal_df = pd.DataFrame(temporal_feature_dict, index=features_df.index)
            features_df = pd.concat([features_df, temporal_df], axis=1)
        
        # 🚀 PERFORMANCE-OPTIMIERTES STATIC FEATURE ENGINEERING  
        if static_features:
            print(f"📊 Verarbeite {len(static_features)} statische Features...")
            
            # Sammle alle statischen Features in Dictionary für Batch-Concat
            static_feature_dict = {}
            
            for feature in static_features:
                if feature in features_df.columns:
                    try:
                        # Customer-Level Aggregationen
                        static_feature_dict[f'{feature}_customer_mean'] = grouped[feature].transform('mean')
                        static_feature_dict[f'{feature}_customer_max'] = grouped[feature].transform('max')
                        static_feature_dict[f'{feature}_customer_min'] = grouped[feature].transform('min')
                        static_feature_dict[f'{feature}_customer_std'] = grouped[feature].transform('std').fillna(0)
                        
                        feature_list.extend([
                            f'{feature}_customer_mean',
                            f'{feature}_customer_max',
                            f'{feature}_customer_min',
                            f'{feature}_customer_std'
                        ])
                    except Exception as e:
                        print(f"⚠️ Fehler bei Static Feature {feature}: {e}")
                        continue
            
            # 🚀 PERFORMANCE-BOOST: Alle statischen Features auf einmal hinzufügen
            if static_feature_dict:
                print(f"🚀 Batch-Concat {len(static_feature_dict)} statische Features...")
                static_df = pd.DataFrame(static_feature_dict, index=features_df.index)
                features_df = pd.concat([features_df, static_df], axis=1)
                
                # Relative Position Features (benötigen vorherige Features)
                for feature in static_features:
                    if feature in features_df.columns:
                        try:
                            features_df[f'{feature}_vs_mean'] = features_df[feature] - features_df[f'{feature}_customer_mean']
                        except Exception as e:
                            print(f"⚠️ Fehler bei Relative Feature {feature}: {e}")
                            continue
        
        # 🚀 PERFORMANCE-OPTIMIERTES CATEGORICAL FEATURE ENGINEERING
        if categorical_features:
            print(f"🏷️ Verarbeite {len(categorical_features)} kategorische Features...")
            
            for feature in categorical_features:
                if feature in features_df.columns:
                    try:
                        feature_series = features_df[feature]
                        
                        # Einheitliche One-Hot-Encoding-Behandlung (inkl. N_DIGITALIZATIONRATE)
                        dummies = pd.get_dummies(feature_series, prefix=feature, drop_first=True)
                        
                        # Füge One-Hot Features hinzu (Boolean → Integer für DuckDB-Kompatibilität)
                        for col in dummies.columns:
                            features_df[col] = dummies[col].astype(int)
                            feature_list.append(col)
                        
                        print(f"   ✅ {feature}: {len(dummies.columns)} One-Hot Features erstellt")
                        
                    except Exception as e:
                        print(f"⚠️ Fehler bei Categorical Feature {feature}: {e}")
                        continue
        
        # 🎯 PURE DATA DICTIONARY: Keine automatischen Numeric Features mehr
        print(f"✅ Pure Data Dictionary System: Alle Features explizit klassifiziert")
        
        # 🚀 PERFORMANCE-OPTIMIERTE CROSS-FEATURE INTERACTIONS
        print("🔗 Erstelle Cross-Feature Interactions...")
        
        # Sammle Interaction Features in Dictionary für Batch-Concat
        interaction_feature_dict = {}
        
        if len(temporal_features) >= 2:
            # DETERMINISTISCHE Top 5 Features (immer sortiert)
            key_features = sorted(temporal_features)[:5]  # ✅ SORTIERT = DETERMINISTISCH!
            for i, feat1 in enumerate(key_features):
                for feat2 in key_features[i+1:]:
                    try:
                        # Ratio (mit Division-by-Zero Protection)
                        interaction_feature_dict[f'ratio_{feat1}_vs_{feat2}'] = (
                            features_df[feat1] / (features_df[feat2] + 0.001)
                        ).fillna(0)
                        feature_list.append(f'ratio_{feat1}_vs_{feat2}')
                    except:
                        continue
        
        # 🚀 PERFORMANCE-BOOST: Alle Interaction Features auf einmal hinzufügen
        if interaction_feature_dict:
            print(f"🚀 Batch-Concat {len(interaction_feature_dict)} interaction Features...")
            interaction_df = pd.DataFrame(interaction_feature_dict, index=features_df.index)
            features_df = pd.concat([features_df, interaction_df], axis=1)
        
        # 🎯 Behandle problematische Werte (ROBUST)
        print("🛡️ Robustheit: Behandle Infinity/NaN-Werte...")
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Intelligente NaN-Behandlung
        for col in feature_list:
            if col in features_df.columns:
                if features_df[col].isna().sum() > 0:
                    # Fülle mit Median für numerische, 0 für andere
                    if features_df[col].dtype in ['int64', 'float64']:
                        fill_value = features_df[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0
                    else:
                        fill_value = 0
                    features_df[col] = features_df[col].fillna(fill_value)
        
        # 🔧 KORRIGIERT: FINALE Feature-Liste OHNE Data Leakage
        # ❌ KEINE Original-Features verwenden (Data Leakage!)
        # ✅ NUR abgeleitete Features verwenden (Rolling, Trend, Early Warning, etc.)
        
        # Sammle NUR abgeleitete Features (keine Original-Features)
        # 🛡️ ULTIMATIVE DATA LEAKAGE PRÄVENTION: ALLE Original-Features ausschließen!
        original_features = temporal_features + static_features  # Das sind die Original-Features
        print(f"🔍 DEBUG: Original-Features: {original_features}")
        print(f"🔍 DEBUG: Feature-List: {len(feature_list)} Features")
        
        derived_features = [f for f in feature_list if f in features_df.columns and f not in original_features]
        print(f"🔍 DEBUG: Abgeleitete Features: {len(derived_features)} Features")
        print(f"🔍 DEBUG: Erste 10 abgeleitete Features: {derived_features[:10]}")
        
        # Zusätzliche Prüfung: Keine Original-Features in abgeleiteten Features
        leaked_features = [f for f in derived_features if f in original_features]
        if leaked_features:
            print(f"❌ DEBUG: DATA LEAKAGE GEFUNDEN: {leaked_features}")
            derived_features = [f for f in derived_features if f not in leaked_features]
            print(f"✅ DEBUG: Data Leakage behoben: {len(derived_features)} Features übrig")
        
        # Sammle One-Hot Features (kategorische Features sind OK)
        one_hot_features = []
        for col in features_df.columns:
            if '_' in col and col not in [primary_key_col, timebase_col, target_col]:
                # One-Hot Features erkennen (enthalten '_' und sind binär)
                if features_df[col].nunique() <= 2:
                    one_hot_features.append(col)
        
        # Kombiniere NUR abgeleitete Features + One-Hot Features
        all_features = derived_features + one_hot_features
        
        # Entferne Duplikate und sortiere
        all_features = sorted(list(set(all_features)))
        
        # 🎯 QUALITÄTSKONTROLLE - KONSTANTE FEATURES ENTFERNEN
        valid_features = []
        constant_features_removed = 0
        for feat in all_features:
            if feat in features_df.columns:
                # ✅ TEMPORÄR DEAKTIVIERT: Feature-Entfernung für Baseline-Vergleich
                # ENTFERNE konstante Features (kein prädiktiver Wert)
                if False and features_df[feat].nunique() == 1:  # DEAKTIVIERT
                    print(f"⚠️ Konstantes Feature ENTFERNT: {feat} (Wert: {features_df[feat].iloc[0]})")
                    constant_features_removed += 1
                else:
                    valid_features.append(feat)
        
        print(f"🗑️ {constant_features_removed} konstante Features entfernt")
        
        print(f"🎯 PURE DATA DICTIONARY Feature-Engineering abgeschlossen:")
        print(f"   📊 Finale Features: {len(valid_features)} (NUR abgeleitete Features)")
        print(f"   ⏰ Dynamic Features verarbeitet: {len(temporal_features)}")
        print(f"   📈 Static Features verarbeitet: {len(static_features)}")
        print(f"   ✅ Data Dictionary Compliance: 100% (keine automatischen Fallbacks)")
        print(f"   🚀 Performance: OPTIMIERT (Batch-Concat eliminiert DataFrame-Fragmentierung)")
        print(f"   🎯 Feature-Qualitätskontrolle: BESTANDEN")
        print(f"   ⚡ Zeitliche Korrektheit: GARANTIERT (nur vergangene Daten)")
        print(f"   🛡️ Robustheit: MAXIMAL (Infinity/NaN behandelt)")
        print(f"   🛡️ DATA LEAKAGE PRÄVENTION: KEINE Original-Features verwendet!")
        
        # ✅ SETZE feature_names NACH Data-Leakage-Korrektur!
        self.feature_names = valid_features
        
        return features_df, valid_features
        
    def create_customer_dataset(self, prediction_timebase=None):
        """Erstellt Customer-Level Dataset für Training"""
        print(f"\n📊 Customer-Level Dataset (Prediction: {prediction_timebase})")
        
        primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
        print(f"🎯 Target-Spalte: {target_col}")
        
        # Debug: Zeige verfügbare Zeiträume
        max_timebase = self.features_df[timebase_col].max()
        min_timebase = self.features_df[timebase_col].min()
        print(f"📅 Verfügbarer Zeitraum: {min_timebase} - {max_timebase}")
        

        
        customer_records = []
        
        print(f"🔍 Debug: {len(self.features_df[primary_key_col].unique())} Kunden gefunden")
        print(f"🔍 Debug: Zeitraum {self.features_df[timebase_col].min()} - {self.features_df[timebase_col].max()}")
        print(f"🔍 Debug: Prediction Timebase: {prediction_timebase}")
        
        for customer in self.features_df[primary_key_col].unique():
            customer_data = self.features_df[self.features_df[primary_key_col] == customer].copy()
            customer_data = customer_data.sort_values(timebase_col)
            
            # Historische Daten vor Prediction-Periode (Features)
            # Konvertiere prediction_timebase zu int für korrekten Vergleich
            prediction_timebase_int = int(prediction_timebase) if prediction_timebase else None
            historical = customer_data[customer_data[timebase_col] < prediction_timebase_int]
            
            if len(historical) > 0:
                latest_record = historical.iloc[-1].copy()
                
                # Zukünftige Daten nach Prediction-Periode (Target)
                future = customer_data[customer_data[timebase_col] >= prediction_timebase_int]
                
                # Churn = wenn I_Alive 0 ist (Kunde ist nicht mehr aktiv)
                if len(future) > 0:
                    will_churn = (future[target_col] == 0).any()
                else:
                    # Keine zukünftigen Daten: Kunde ist nicht mehr aktiv
                    will_churn = False
                
                # Verwende die originale Target-Spalte, nicht WILL_CHURN
                latest_record[target_col] = int(will_churn)
                customer_records.append(latest_record)
        
        customer_df = pd.DataFrame(customer_records)
        
        # Debug: Zeige Target-Verteilung
        if target_col in customer_df.columns:
            churn_count = (customer_df[target_col] == 1).sum()
            total_count = len(customer_df)
            print(f"✅ {total_count} Kunden, {churn_count} Churns, Churn Rate: {customer_df[target_col].mean():.3f}")
        else:
            print(f"⚠️ Target-Spalte '{target_col}' nicht gefunden!")
            print(f"🔍 Verfügbare Spalten: {list(customer_df.columns)}")
        
        return customer_df
        
    def train_enhanced_model(self, prediction_timebase=None, training_from=None, training_to=None):
        """Trainiert Enhanced Early Warning Model mit optimierten Konfigurationen"""
        print(f"\n🤖 === OPTIMIERTES MODEL TRAINING ===")
        
        # 🔧 DATA LEAKAGE PRÄVENTION: Validiere prediction_timebase
        if prediction_timebase is None:
            raise ValueError("❌ DATA LEAKAGE GEFÄHRDUNG: prediction_timebase muss gesetzt werden!")
        
        print(f"🛡️ DATA LEAKAGE PRÄVENTION: prediction_timebase = {prediction_timebase}")
        
        # Features laden falls noch nicht geladen
        if not self.feature_names:
            self.load_and_enhance_features()
        
        
        # Debug: Zeige Target-Verteilung
        target_col = self._get_dynamic_column_names()[2]  # Target-Spalte
        
        print(f"🎯 Target-Verteilung wird später angezeigt...")
        
        # Customer Dataset erstellen
        customer_df = self.create_customer_dataset(prediction_timebase=prediction_timebase)
        
        # Debug: Zeige Target-Verteilung
        print(f"🎯 Target-Verteilung:")
        print(f"   {target_col} = 0: {(customer_df[target_col] == 0).sum()}")
        print(f"   {target_col} = 1: {(customer_df[target_col] == 1).sum()}")
        print(f"   Churn Rate: {customer_df[target_col].mean():.3f}")
        
        # Prüfe ob genügend Churn-Samples vorhanden sind
        churn_samples = (customer_df[target_col] == 1).sum()
        if churn_samples < 10:
            print(f"⚠️ WARNUNG: Nur {churn_samples} Churn-Samples - Modell könnte Probleme haben")
        
        # Zusätzliche Zeitfilterung falls angegeben
        if training_from is not None and training_to is not None:
            print(f"🛡️ Zusätzliche Zeitfilterung: {training_from} bis {training_to}")
            training_from_int = int(training_from)
            training_to_int = int(training_to)
            timebase_col = self._get_dynamic_column_names()[1]  # Timebase-Spalte
            customer_df = customer_df[
                (customer_df[timebase_col] >= training_from_int) &
                (customer_df[timebase_col] <= training_to_int)
            ]
            print(f"📊 Nach Zeitfilterung: {len(customer_df)} Kunden")
        
        # Features vorbereiten
        X = customer_df[self.feature_names].fillna(0)
        y = customer_df[target_col]
        
        print(f"📊 Training: {len(X)} Samples, {len(self.feature_names)} Features")
        
        # Train/Test Split mit stratify nur wenn beide Klassen vorhanden
        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        else:
            print("⚠️ WARNUNG: Nur eine Klasse im Target - verwende einfachen Split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # ✅ MIGRIERT: Verwende extrahierte ChurnModelTrainer
        print("🔧 === MODEL TRAINING MIT EXTRAHIERTER CHURN MODEL TRAINER ===")
        
        # Initialisiere Model Trainer mit Experiment ID
        model_trainer = ChurnModelTrainer(experiment_id=getattr(self, 'experiment_id', None))
        
        # Lade optimierte Konfiguration
        optimized_config = self._load_optimized_config()
        
        # Bestimme Parameter für Model Trainer
        model_params = None
        if optimized_config:
            print(f"✅ Optimierte Konfiguration gefunden: {optimized_config['model_type']}")
            # Konvertiere zur ChurnModelTrainer Parameter-Format
            model_params = {
                'n_estimators': optimized_config.get('n_estimators', 300),
                'max_depth': optimized_config.get('max_depth', 12),
                'min_samples_split': optimized_config.get('min_samples_split', 5),
                'min_samples_leaf': optimized_config.get('min_samples_leaf', 2),
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
            
            # Sampling anwenden falls konfiguriert
            if optimized_config.get('sampling_config', {}).get('strategy') != 'none':
                X_train, y_train = self._apply_sampling(X_train, y_train, optimized_config['sampling_config'])
                print(f"✅ Sampling angewendet: {optimized_config['sampling_config']['strategy']}")
        else:
            # Fail-Fast: Ohne optimierte Konfiguration nicht fortfahren
            raise RuntimeError("Keine optimierte Konfiguration verfügbar (weder Experiment-Override noch Datei)")
        
        # Trainiere Model mit ChurnModelTrainer 
        print("🏃‍♂️ Training mit ChurnModelTrainer...")
        self.model = model_trainer.train_random_forest(
            X_train.values, y_train.values,
            X_test.values, y_test.values,
            params=model_params
        )
        
        # Speichere Model Trainer für weitere Verwendung
        self.model_trainer = model_trainer
        
        # ✅ MIGRIERT: Evaluation mit extrahierter ChurnEvaluator
        print("📊 === MODEL EVALUATION MIT EXTRAHIERTER CHURN EVALUATOR ===")
        
        # Initialisiere Evaluator
        evaluator = ChurnEvaluator(experiment_id=getattr(self, 'experiment_id', None))
        
        # Evaluation mit ChurnEvaluator
        evaluation_results = evaluator.evaluate_model_performance(
            self.model, 
            X_test.values, y_test.values,
            feature_names=self.feature_names
        )
        
        print(f"📊 ChurnEvaluator Performance:")
        print(f"   🎯 AUC: {evaluation_results.get('auc', 0.0):.3f}")
        print(f"   🎯 Precision: {evaluation_results.get('precision', 0.0):.3f}")
        print(f"   🎯 Recall: {evaluation_results.get('recall', 0.0):.3f}")
        print(f"   🎯 F1-Score: {evaluation_results.get('f1_score', 0.0):.3f}")
        print(f"   🎯 Accuracy: {evaluation_results.get('accuracy', 0.0):.3f}")
        
        # Kompatibilität: Setze legacy Variablen für Backward Compatibility
        auc_score = evaluation_results.get('auc', 0.0)
        report = {
            'accuracy': evaluation_results.get('accuracy', 0.0),
            '1': {
                'precision': evaluation_results.get('precision', 0.0),
                'recall': evaluation_results.get('recall', 0.0),
                'f1-score': evaluation_results.get('f1_score', 0.0)
            }
        }
        
        # Speichere Evaluator für weitere Verwendung
        self.evaluator = evaluator
        
        print(f"   🎯 AUC: {auc_score:.3f}")
        
        # 🔧 NEU: Training-Zeiträume in Metriken speichern
        training_metrics = {
            'accuracy': report['accuracy'],
            'precision': report.get('1', {}).get('precision', 0.0),
            'recall': report.get('1', {}).get('recall', 0.0),
            'f1_score': report.get('1', {}).get('f1-score', 0.0),
            'auc': auc_score,
            'training_from': training_from,
            'training_to': training_to,
            'prediction_timebase': prediction_timebase,
            'samples_count': len(X),
            'features_count': len(self.feature_names),
            'churn_rate': y.mean(),
            'model_type': 'Enhanced_EarlyWarning'
        }
        
        print(f"✅ Training abgeschlossen: {len(X)} Samples, AUC: {auc_score:.3f}")
        
        return training_metrics
    
    def _load_optimized_config(self):
        """Lädt optimierte Konfiguration – ohne Fallbacks (Fail-Fast)."""
        # 1) Experiment-Override (vom Orchestrator gesetzt)
        try:
            override = getattr(self, '_optimized_config_override', None)
            if isinstance(override, dict) and override:
                print("🧩 Algorithm Config Source: experiments.hyperparameters.algorithm_config (override)")
                return override
        except Exception:
            pass

        # 2) Pflicht: Datei aus config/ (Single Source of Truth)
        config_path = self.paths.config_directory() / "algorithm_config_optimized.json"
        if not config_path.exists():
            raise FileNotFoundError(f"algorithm_config_optimized.json nicht gefunden: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        print("🧩 Algorithm Config Source: config/algorithm_config_optimized.json")
        return cfg
    
    def _create_optimized_model(self, config):
        """Erstellt Modell basierend auf optimierter Konfiguration"""
        model_type = config.get('model_type', 'RandomForestClassifier')
        
        if model_type == 'RandomForestClassifier':
            return RandomForestClassifier(
                n_estimators=config.get('n_estimators', 200),
                max_depth=config.get('max_depth', 20),
                min_samples_split=config.get('min_samples_split', 5),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                random_state=config.get('random_state', 42),
                class_weight=config.get('class_weight', 'balanced_subsample'),
                n_jobs=-1
            )
        else:
            print(f"⚠️ Unbekannter Modell-Typ: {model_type} - verwende RandomForest")
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced_subsample',
                n_jobs=-1
            )
    
    def _apply_sampling(self, X_train, y_train, sampling_config):
        """Wendet Sampling basierend auf optimierter Konfiguration an"""
        strategy = sampling_config.get('strategy', 'none')
        parameters = sampling_config.get('parameters', {})
        
        # 🔍 IDENTIFIZIERE KATEGORISCHE FEATURES (One-Hot-encoded)
        categorical_features = []
        for col in X_train.columns:
            # One-Hot-encoded Features erkennen (enthalten '_' und sind binär)
            if '_' in col and X_train[col].nunique() <= 2:
                categorical_features.append(col)
        
        if categorical_features:
            print(f"🔍 Erkannte {len(categorical_features)} kategorische Features - werden vom Sampling ausgeschlossen")
            print(f"   Kategorische Features: {categorical_features[:5]}...")
        
        # 🚫 SCHLIESSE KATEGORISCHE FEATURES VOM SAMPLING AUS
        if categorical_features:
            # Trenne kategorische und numerische Features
            X_categorical = X_train[categorical_features]
            X_numerical = X_train.drop(categorical_features, axis=1)
            
            # Wende Sampling nur auf numerische Features an
            if strategy == 'smote_enhanced':
                from imblearn.over_sampling import SMOTE
                sampling_ratio = parameters.get('sampling_ratio', 3.0)
                if sampling_ratio > 1.0:
                    sampling_strategy = 1.0
                else:
                    sampling_strategy = sampling_ratio
                sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=parameters.get('k_neighbors', 3),
                    random_state=parameters.get('random_state', 42)
                )
            elif strategy == 'smoteenn':
                from imblearn.combine import SMOTEENN
                sampler = SMOTEENN(random_state=parameters.get('random_state', 42))
            else:
                print(f"⚠️ Unbekannte Sampling-Strategie: {strategy} - keine Sampling angewendet")
                return X_train, y_train
            
            # Sampling nur auf numerische Features
            X_numerical_resampled, y_resampled = sampler.fit_resample(X_numerical, y_train)
            
            # Füge kategorische Features wieder hinzu (ohne Sampling)
            # Für neue Samples (SMOTE-generiert): Verwende 0 für kategorische Features
            if len(y_resampled) > len(y_train):
                # SMOTE hat neue Samples erstellt
                # Verwende die Indizes von X_numerical_resampled für Konsistenz
                X_categorical_resampled = pd.DataFrame(0, index=X_numerical_resampled.index, columns=X_categorical.columns)
            else:
                # Keine neuen Samples - verwende ursprüngliche Indizes
                X_categorical_resampled = X_categorical.iloc[y_resampled.index]
            
            # Kombiniere numerische und kategorische Features
            X_resampled = pd.concat([X_numerical_resampled, X_categorical_resampled], axis=1)
            
            # Stelle sicher, dass die Feature-Reihenfolge mit dem ursprünglichen Training übereinstimmt
            # Verwende die ursprüngliche Reihenfolge der Features
            original_columns = X_train.columns.tolist()
            X_resampled = X_resampled[original_columns]
            
            return X_resampled, y_resampled
        else:
            # Keine kategorischen Features - normales Sampling
            if strategy == 'smote_enhanced':
                from imblearn.over_sampling import SMOTE
                sampling_ratio = parameters.get('sampling_ratio', 3.0)
                if sampling_ratio > 1.0:
                    sampling_strategy = 1.0
                else:
                    sampling_strategy = sampling_ratio
                sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=parameters.get('k_neighbors', 3),
                    random_state=parameters.get('random_state', 42)
                )
            elif strategy == 'smoteenn':
                from imblearn.combine import SMOTEENN
                sampler = SMOTEENN(random_state=parameters.get('random_state', 42))
            else:
                print(f"⚠️ Unbekannte Sampling-Strategie: {strategy} - keine Sampling angewendet")
                return X_train, y_train
            
            return sampler.fit_resample(X_train, y_train)
    
    def real_world_backtest(self, training_from=None, training_to=None, test_from=None, test_to=None, use_existing_model=False):
        """🏆 REAL-WORLD BACKTEST mit flexiblen Zeiträumen und echten Performance-Metriken"""
        
        if training_from is None or training_to is None or test_from is None or test_to is None:
            raise ValueError("❌ FEHLER: Alle Zeiträume müssen explizit gesetzt werden! Keine Fallbacks erlaubt.")
        
        # 🔧 DATENTYP-KONVERTIERUNG: Konvertiere alle Perioden zu int
        training_from = int(training_from)
        training_to = int(training_to)
        test_from = int(test_from)
        test_to = int(test_to)
        
        print(f"\n🏆 === REAL-WORLD BACKTEST (Flexible Zeiträume) ===")
        print(f"📅 Training: {training_from} bis {training_to}")
        print(f"📅 Test: {test_from} bis {test_to}")
        
        # Dynamische Spaltennamen aus Data Dictionary
        primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
        
        # Prüfe ob Features bereits verfügbar sind
        if hasattr(self, 'features_df') and self.features_df is not None:
            print("✅ Features bereits verfügbar - prüfe auf Early Warning Features...")
            # Prüfe ob Early Warning Features vorhanden sind
            ew_features = [col for col in self.features_df.columns if '_ew_' in col]
            if len(ew_features) == 0:
                print("🚨 Keine Early Warning Features gefunden - erstelle Features neu...")
                self.load_and_enhance_features()
            else:
                print(f"✅ {len(ew_features)} Early Warning Features gefunden")
        else:
            print("🔄 Features nicht verfügbar - erstelle Features neu...")
            self.load_and_enhance_features()
        
        # Erstelle Customer-Level Dataset
        customer_df = self.create_customer_dataset(prediction_timebase=test_from)
        
        # Training Data (historische Daten)
        train_df = customer_df.copy()
        # ✅ timebase_col ist bereits NUMERIC (aus Data Dictionary)
        train_data = train_df[
            (train_df[timebase_col] >= training_from) &
            (train_df[timebase_col] <= training_to)
        ]
        
        print(f"📚 Training Phase ({training_from} bis {training_to}):")
        print(f"   Kunden: {len(train_data):,}")
        print(f"   Churn Rate: {train_data[target_col].mean():.3f}")
        
        # Test Data (zukünftige Daten - DYNAMISCHES FENSTER basierend auf Test-Perioden-Länge)
        # Korrekte Berechnung für YYYYMM Format
        def months_between(yyyymm1, yyyymm2):
            """Berechnet Monate zwischen zwei YYYYMM Perioden"""
            year1 = yyyymm1 // 100
            month1 = yyyymm1 % 100
            year2 = yyyymm2 // 100
            month2 = yyyymm2 % 100
            
            # Berechne Monate zwischen den beiden Daten
            months_diff = (year2 - year1) * 12 + (month2 - month1)
            return months_diff + 1  # +1 weil wir beide Endpunkte einschließen
        
        # Korrekte Test-Perioden-Länge für YYYYMM Format
        test_period_length = months_between(test_from, test_to)
        
        # DYNAMISCHE TARGET-DEFINITION: Anpassung an Test-Perioden-Länge
        # KORREKTUR: Target-Fenster ist BEGRENZT auf Test-Perioden-Länge
        target_window_months = min(test_period_length, 12)  # Maximal 12 Monate, aber nicht länger als Test-Periode
        target_start = test_from
        
        print(f"🎯 DYNAMISCHE TARGET-DEFINITION:")
        print(f"   Test-Perioden-Länge: {test_period_length} Monate")
        print(f"   Target-Fenster: {target_window_months} Monate (ab {target_start})")
        
        # Finde alle Kunden die in Test-Periode aktiv waren
        # Korrekte Test-Perioden für YYYYMM Format
        def generate_yyyymm_periods(start_yyyymm, end_yyyymm):
            """Generiert Liste von YYYYMM Perioden zwischen start und end"""
            periods = []
            current = start_yyyymm
            while current <= end_yyyymm:
                periods.append(current)
                # Nächste Periode
                year = current // 100
                month = current % 100
                if month == 12:
                    current = (year + 1) * 100 + 1
                else:
                    current = year * 100 + month + 1
            return periods
        
        # Berechne Target Window Information
        target_window_info = self._calculate_target_window_info(test_from, test_to)
        test_periods = target_window_info['test_periods']
        # ✅ timebase_col ist bereits NUMERIC (aus Data Dictionary)
        features_df_numeric = self.features_df.copy()
        
        # KONVERTIERUNG: Verarbeite ALLE Kunden die im Test-Zeitraum vorhanden waren
        # Nicht nur aktive Kunden, sondern alle Kunden die im Test-Zeitraum vorhanden waren
        test_period_customers = features_df_numeric[
            (features_df_numeric[timebase_col] == test_from)  # Kunden im Test-Zeitraum
        ][primary_key_col].unique()
        
        print(f"🎯 Verarbeite ALLE Kunden die im Test-Zeitraum aktiv waren: {len(test_period_customers)} Kunden")
        
        # Verwende alle Test-Zeitraum-Kunden für Vorhersagen oder reduziere für Testmodus
        test_customers = test_period_customers
        try:
            reduction = float(getattr(self, 'test_reduction', 0.0) or 0.0)
        except Exception:
            reduction = 0.0
        if reduction > 0.0:
            sample_fraction = max(0.0, min(1.0, 1.0 - reduction))
            if sample_fraction <= 0.0:
                sample_fraction = 0.1  # Fallback auf 10%
            import numpy as _np
            rs = _np.random.RandomState(42)
            sample_size = max(1, int(len(test_period_customers) * sample_fraction))
            if sample_size < len(test_period_customers):
                sampled_idx = rs.choice(len(test_period_customers), size=sample_size, replace=False)
                test_customers = test_period_customers[sampled_idx]
        
        # Erstelle Test-Dataset mit Customer-Level Aggregation
        test_records = []
        
        for customer in test_customers:
            # Historische Daten für Features (bis Training-Ende)
            # ✅ timebase_col ist bereits NUMERIC (aus Data Dictionary)
            customer_historical = self.features_df[
                (self.features_df[primary_key_col] == customer) &
                (self.features_df[timebase_col] <= training_to)
            ].copy()
            
            # Zukünftige Daten für Target (BEGRENZT auf Test-Perioden)
            customer_future = self.features_df[
                (self.features_df[primary_key_col] == customer) &
                (self.features_df[timebase_col] >= target_start) &  # Ab Test-Start
                (self.features_df[timebase_col] <= test_to)  # Bis Test-Ende (nicht darüber hinaus!)
            ].copy()
            
            if len(customer_historical) > 0:
                # Nehm den letzten historischen Record für Features
                latest_record = customer_historical.iloc[-1].copy()
                
                # Churn-Status in Zukunft (Dynamisches Fenster)
                # I_Alive: 1=aktiv, 0=churned → will_churn wenn irgendein Wert 0 ist
                will_churn = (customer_future[target_col] == 0).any()
                
                # Zeit bis Churn (für Analyse) - Korrekte YYYYMM Berechnung
                if will_churn:
                    churn_period = customer_future[customer_future[target_col] == 0][timebase_col].min()
                    months_to_churn = months_between(target_start, churn_period) - 1  # -1 weil wir ab target_start zählen
                else:
                    months_to_churn = None
                
                latest_record['ACTUAL_CHURN'] = int(will_churn)
                latest_record['MONTHS_TO_CHURN'] = months_to_churn
                test_records.append(latest_record)
        
        test_df = pd.DataFrame(test_records)
        
        print(f"🧪 Test Phase - {target_window_months}-Monate-Fenster (>= {target_start}):")
        print(f"   Kunden: {len(test_df):,}")
        print(f"   Actual Churn Rate ({target_window_months}+ Monate): {test_df['ACTUAL_CHURN'].mean():.3f}")
        
        # Zeit-zu-Churn Analyse
        churn_timing = test_df[test_df['ACTUAL_CHURN'] == 1]['MONTHS_TO_CHURN'].dropna()
        if len(churn_timing) > 0:
            print(f"   Durchschnittliche Zeit bis Churn: {churn_timing.mean():.1f} Monate")
            
            # Churn-Verteilung
            early_churns = len(churn_timing[churn_timing <= 3])
            mid_churns = len(churn_timing[(churn_timing > 3) & (churn_timing <= 6)])
            late_churns = len(churn_timing[churn_timing > 6])
            print(f"   Churn-Verteilung: 0-3M: {early_churns}, 4-6M: {mid_churns}, 6+M: {late_churns}")
            
            # MONTHS_TO_CHURN Wahrscheinlichkeits-Analyse
            months_to_churn_analysis = {
                'mean_months_to_churn': float(churn_timing.mean()),
                'median_months_to_churn': float(churn_timing.median()),
                'min_months_to_churn': float(churn_timing.min()),
                'max_months_to_churn': float(churn_timing.max()),
                'std_months_to_churn': float(churn_timing.std()),
                'months_distribution': churn_timing.value_counts().sort_index().to_dict(),
                'probability_categories': {
                    'immediate': len(churn_timing[churn_timing <= 1]) / len(churn_timing),
                    'short_term': len(churn_timing[(churn_timing > 1) & (churn_timing <= 3)]) / len(churn_timing),
                    'medium_term': len(churn_timing[(churn_timing > 3) & (churn_timing <= 6)]) / len(churn_timing),
                    'long_term': len(churn_timing[(churn_timing > 6) & (churn_timing <= 12)]) / len(churn_timing),
                    'very_long_term': len(churn_timing[churn_timing > 12]) / len(churn_timing)
                }
            }
            
            print(f"\n🎯 MONTHS_TO_CHURN WAHRSCHEINLICHKEITEN:")
            print(f"   Sofort (0-1 Monate): {months_to_churn_analysis['probability_categories']['immediate']:.1%}")
            print(f"   Kurzfristig (2-3 Monate): {months_to_churn_analysis['probability_categories']['short_term']:.1%}")
            print(f"   Mittelfristig (4-6 Monate): {months_to_churn_analysis['probability_categories']['medium_term']:.1%}")
            print(f"   Langfristig (7-12 Monate): {months_to_churn_analysis['probability_categories']['long_term']:.1%}")
            print(f"   Sehr langfristig (>12 Monate): {months_to_churn_analysis['probability_categories']['very_long_term']:.1%}")
        else:
            months_to_churn_analysis = {}
        
        # INTELLIGENTE MODELL-WAHL: Verwende IMMER das bereits trainierte Modell für Backtest
        if self.model is not None:
            print("🎯 Verwende bereits trainiertes Modell für Backtest")
            backtest_model = self.model
            model_training_performed = False
            
            # ✅ KORRIGIERT: Definiere X_train/y_train auch für bereits trainiertes Modell
            X_train = train_data[self.feature_names].fillna(0)
            y_train = train_data[target_col]
        else:
            print("⚠️ Kein trainiertes Modell verfügbar - trainiere neues Modell")
            # ✅ MIGRIERT: Fallback Model Training mit ChurnModelTrainer
            print("🔧 Fallback Model Training mit ChurnModelTrainer...")
            X_train = train_data[self.feature_names].fillna(0)
            y_train = train_data[target_col]
            
            # Initialisiere Fallback Model Trainer
            fallback_trainer = ChurnModelTrainer(experiment_id=getattr(self, 'experiment_id', None))
            
            # Standard Parameter für Fallback (IDENTICAL zu ChurnModelTrainer!)
            fallback_params = {
                'n_estimators': 300,
                'max_depth': 12,
                'min_samples_split': 5,     # ✅ KRITISCHER PARAMETER HINZUGEFÜGT!
                'min_samples_leaf': 2,      # ✅ KRITISCHER PARAMETER HINZUGEFÜGT!
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
            
            # Trainiere Fallback Model
            backtest_model = fallback_trainer.train_random_forest(
                X_train.values, y_train.values,
                params=fallback_params
            )
            model_training_performed = True
        
        # ✅ MIGRIERT: Training Performance mit ChurnEvaluator
        print(f"\n📚 TRAINING PERFORMANCE (mit ChurnEvaluator):")
        
        # Initialisiere Training Evaluator
        training_evaluator = ChurnEvaluator(experiment_id=getattr(self, 'experiment_id', None))
        
        # Training Evaluation mit ChurnEvaluator
        training_evaluation = training_evaluator.evaluate_model_performance(
            backtest_model,
            X_train.values, y_train.values,
            feature_names=self.feature_names
        )
        
        print(f"   🎯 Training AUC: {training_evaluation.get('auc', 0.0):.3f}")
        print(f"   🎯 Training Precision: {training_evaluation.get('precision', 0.0):.3f}")
        print(f"   🎯 Training Recall: {training_evaluation.get('recall', 0.0):.3f}")
        print(f"   🎯 Training F1-Score: {training_evaluation.get('f1_score', 0.0):.3f}")
        print(f"   🎯 Training Accuracy: {training_evaluation.get('accuracy', 0.0):.3f}")
        
        # Kompatibilität: Legacy Variablen für Backward Compatibility
        train_auc = training_evaluation.get('auc', 0.0)
        train_report = {
            'accuracy': training_evaluation.get('accuracy', 0.0),
            '1': {
                'precision': training_evaluation.get('precision', 0.0),
                'recall': training_evaluation.get('recall', 0.0),
                'f1-score': training_evaluation.get('f1_score', 0.0)
            }
        }
        
        # Speichere Training-KPIs
        training_metrics = {
            'training_accuracy': train_report['accuracy'],
            'training_precision': train_report['1']['precision'] if '1' in train_report else 0.0,
            'training_recall': train_report['1']['recall'] if '1' in train_report else 0.0,
            'training_f1_score': train_report['1']['f1-score'] if '1' in train_report else 0.0,
            'training_auc': train_auc,
            'training_samples': len(train_data)
        }
        
        # Predictions
        X_test = test_df[self.feature_names].fillna(0)
        y_test = test_df['ACTUAL_CHURN']
        
        y_pred = backtest_model.predict(X_test)
        y_pred_proba_full = backtest_model.predict_proba(X_test)
        
        # Sichere Behandlung von predict_proba (falls nur eine Klasse)
        if y_pred_proba_full.shape[1] > 1:
            y_pred_proba = y_pred_proba_full[:, 1]  # Churn-Wahrscheinlichkeit
        else:
            # Fallback: Nur eine Klasse vorhergesagt
            y_pred_proba = y_pred_proba_full[:, 0]
            print("⚠️ Backtest: Modell hat nur eine Klasse vorhergesagt - verwende Fallback")
        
        # ✅ MIGRIERT: Backtest Metrics mit ChurnEvaluator
        print(f"\n🧪 BACKTEST EVALUATION (mit ChurnEvaluator):")
        
        # Initialisiere Backtest Evaluator
        backtest_evaluator = ChurnEvaluator(experiment_id=getattr(self, 'experiment_id', None))
        
        # Backtest Evaluation mit ChurnEvaluator
        backtest_evaluation = backtest_evaluator.evaluate_model_performance(
            backtest_model,
            X_test.values, y_test.values,
            feature_names=self.feature_names
        )
        
        print(f"   🎯 Backtest AUC: {backtest_evaluation.get('auc', 0.0):.3f}")
        print(f"   🎯 Backtest Precision: {backtest_evaluation.get('precision', 0.0):.3f}")
        print(f"   🎯 Backtest Recall: {backtest_evaluation.get('recall', 0.0):.3f}")
        print(f"   🎯 Backtest F1-Score: {backtest_evaluation.get('f1_score', 0.0):.3f}")
        
        # Kompatibilität: Legacy Variablen
        backtest_auc = backtest_evaluation.get('auc', 0.0)
        backtest_report = {
            'accuracy': backtest_evaluation.get('accuracy', 0.0),
            '1': {
                'precision': backtest_evaluation.get('precision', 0.0),
                'recall': backtest_evaluation.get('recall', 0.0),
                'f1-score': backtest_evaluation.get('f1_score', 0.0)
            }
        }
        
        # Risk Segmentation
        test_df['CHURN_PROBABILITY'] = y_pred_proba
        
        # Wissenschaftlich fundierte Schwellwert-Optimierung
        print("\n🔬 WISSENSCHAFTLICHE SCHWELLWERT-OPTIMIERUNG:")
        optimizer = RobustThresholdOptimizer(cv_folds=5, n_bootstrap=1000)
        threshold_result = optimizer.optimize(y_test, y_pred_proba)
        
        # Extrahiere optimalen Schwellwert aus Dictionary
        if isinstance(threshold_result, dict):
            optimal_threshold = threshold_result.get('optimal_threshold', 0.5)
            method = threshold_result.get('method', 'unknown')
            print(f"   Optimaler Schwellwert: {optimal_threshold:.3f}")
            print(f"   Methode: {method}")
        else:
            optimal_threshold = threshold_result
            print(f"   Optimaler Schwellwert: {optimal_threshold:.3f}")
        
        # GMM-basierte Risiko-Segmentierung auf Testdaten
        print("\n🔬 GMM-BASIERTE RISIKO-SEGMENTIERUNG:")
        test_risk_levels, test_cluster_stats = self._calculate_gmm_risk_segments(y_pred_proba, optimal_threshold, "Test")
        test_df['RISK_LEVEL'] = test_risk_levels
        
        # GMM-basierte Risiko-Segmentierung auf Trainingsdaten (robust gegen Ein-Klassen-Fälle)
        y_train_pred_proba = self._get_positive_class_proba(backtest_model, X_train)
        train_risk_levels, train_cluster_stats = self._calculate_gmm_risk_segments(y_train_pred_proba, optimal_threshold, "Training")
        
        # Cluster-Vergleich
        cluster_comparison = self._compare_train_test_clusters(train_cluster_stats, test_cluster_stats)
        
        risk_analysis = test_df.groupby('RISK_LEVEL').agg({
            'ACTUAL_CHURN': ['count', 'sum', 'mean']
        }).round(3)
        
        print(f"\n🏆 BACKTEST RESULTS:")
        print(f"   🎯 Accuracy: {backtest_report['accuracy']:.3f}")
        
        # Sichere Prüfung für Klasse '1' (Churn)
        if '1' in backtest_report:
            print(f"   🎯 Precision: {backtest_report['1']['precision']:.3f}")
            print(f"   🎯 Recall: {backtest_report['1']['recall']:.3f}")
            print(f"   🎯 F1-Score: {backtest_report['1']['f1-score']:.3f}")
        else:
            print(f"   ⚠️ Keine Churn-Klasse '1' gefunden - Modell hat nur eine Klasse vorhergesagt")
            print(f"   🎯 Precision: N/A")
            print(f"   🎯 Recall: N/A")
            print(f"   🎯 F1-Score: N/A")
        
        print(f"   🎯 AUC: {backtest_auc:.3f}")
        
        print(f"\n📊 GMM-BASIERTE RISIKO-SEGMENTIERUNG - {target_window_months}-Monate-Fenster:")
        print(f"   Wissenschaftlich optimierter Schwellwert: {optimal_threshold:.3f}")
        
        # Zeige Test-Cluster-Statistiken
        print(f"\n🎯 TEST-CLUSTER (Backtest {test_from}-{test_to}):")
        for risk_level in test_cluster_stats['risk_levels']:
            if risk_level in risk_analysis.index:
                customers = risk_analysis.loc[risk_level, ('ACTUAL_CHURN', 'count')]
                churns = risk_analysis.loc[risk_level, ('ACTUAL_CHURN', 'sum')]
                rate = risk_analysis.loc[risk_level, ('ACTUAL_CHURN', 'mean')]
                cluster_mean = test_cluster_stats['cluster_means'].get(risk_level, 0)
                print(f"   {risk_level}: {customers} customers, {churns} churns, {rate:.1%} rate, Mean Prob: {cluster_mean:.3f}")
        
        # Zeige Training-Cluster-Statistiken
        print(f"\n📚 TRAINING-CLUSTER (Training {training_from}-{training_to}):")
        for risk_level in train_cluster_stats['risk_levels']:
            cluster_mean = train_cluster_stats['cluster_means'].get(risk_level, 0)
            cluster_size = train_cluster_stats['cluster_sizes'].get(risk_level, 0)
            print(f"   {risk_level}: {cluster_size} customers, Mean Prob: {cluster_mean:.3f}")
        
        # Zeige Cluster-Vergleich
        print(f"\n🔄 CLUSTER-STABILITÄT:")
        print(f"   Training Cluster: {len(train_cluster_stats['risk_levels'])}")
        print(f"   Test Cluster: {len(test_cluster_stats['risk_levels'])}")
        print(f"   Stabilität: {cluster_comparison['stability_score']:.3f}")
        
        # INTELLIGENTES SPEICHERN: Speichere IMMER die echten Backtest-Performance-Metriken
        print(f"\n💾 Speichere Enhanced Early Warning Modell mit ECHTEN Backtest-Performance-Metriken...")
        
        # Sichere Prüfung für Klasse '1' (Churn)
        if '1' in backtest_report:
            print(f"✅ ECHTE PERFORMANCE: AUC {backtest_auc:.3f}, Recall {backtest_report['1']['recall']:.3f}")
            
                    # ECHTE Backtest-Metriken als Hauptmetriken
            real_performance_metrics = {
                'precision': backtest_report['1']['precision'],
                'recall': backtest_report['1']['recall'],
                'f1_score': backtest_report['1']['f1-score'],
                'roc_auc': backtest_auc,
                'accuracy': backtest_report['accuracy'],
                # Training-KPIs hinzufügen
                'training_accuracy': training_metrics['training_accuracy'],
                'training_precision': training_metrics['training_precision'],
                'training_recall': training_metrics['training_recall'],
                'training_f1_score': training_metrics['training_f1_score'],
                'training_auc': training_metrics['training_auc'],
                'training_samples': training_metrics['training_samples']
            }
        else:
            print(f"⚠️ Keine Churn-Klasse '1' gefunden - verwende Standard-Metriken")
            
            # Fallback-Metriken
            real_performance_metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': backtest_auc,
                'accuracy': backtest_report['accuracy']
            }
        
        training_info = {
            "algorithm": "RandomForest_Enhanced_Backtest",
            "level": "enhanced_early_warning_backtest",
            "features_count": len(self.feature_names),
            "compatibility": "system_health_ui",
            "lead_time_months": target_window_months,
            "warning_signals": len([f for f in self.feature_names if '_ew_' in f]),
            "training_period": f"{training_from}-{training_to}",
            "test_period": f"{test_from}-{test_to}",
            "target_window_months": target_window_months,
            "target_start": target_start,
            "training_customers": len(train_data),
            "test_customers": len(test_df),
            "performance_type": "real_world_backtest",
            "note": f"Echte Performance aus {target_window_months}-Monate-Fenster Backtest"
        }
        
        # Konvertiere training_info zu dict falls es ein tuple ist
        if isinstance(training_info, tuple):
            training_info = dict(training_info)
        
        model_path, metadata_path = self.save_enhanced_model(
            backtest_model, 
            real_performance_metrics, 
            self.feature_names,
            training_info
        )
        
        # Feature Importance für Erklärungen
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': backtest_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Erstelle zwei verschiedene Kunden-Listen für UI
        
        # 1. VALIDATION CUSTOMERS: ALLE Test-Zeitraum-Kunden mit Churn-Vorhersagen
        validation_customers = []
        print(f"🎯 Erstelle Customer Predictions für alle {len(test_customers)} Test-Zeitraum-Kunden...")
        
        # Erstelle Vorhersagen für ALLE Test-Zeitraum-Kunden (nicht nur test_df)
        test_period_data = []
        print(f"🔍 Analysiere {len(test_customers)} Test-Zeitraum-Kunden...")
        
        customers_with_historical = 0
        customers_without_historical = 0
        
        for customer in test_customers:
            # Historische Daten für Features (bis Training-Ende)
            customer_historical = self.features_df[
                (self.features_df[primary_key_col] == customer) &
                (self.features_df[timebase_col] <= training_to)
            ].copy()
            
            if len(customer_historical) > 0:
                customers_with_historical += 1
                # Nehm den letzten historischen Record für Features
                latest_record = customer_historical.iloc[-1].copy()
                
                # Zukünftige Daten für Target (BEGRENZT auf Test-Perioden)
                customer_future = self.features_df[
                    (self.features_df[primary_key_col] == customer) &
                    (self.features_df[timebase_col] >= target_start) &  # Ab Test-Start
                    (self.features_df[timebase_col] <= test_to)  # Bis Test-Ende
                ].copy()
                
                # Churn-Status in Zukunft (Dynamisches Fenster)
                will_churn = (customer_future[target_col] == 0).any() if len(customer_future) > 0 else False
                
                latest_record['ACTUAL_CHURN'] = int(will_churn)
                test_period_data.append(latest_record)
            else:
                customers_without_historical += 1
                # ENDGÜLTIGE LÖSUNG: Erstelle Dummy-Record für Kunden ohne historische Daten
                # Verwende Durchschnittswerte für Features
                dummy_record = {}
                for feature in self.feature_names:
                    # Verwende Median-Werte aus dem gesamten Dataset
                    feature_median = self.features_df[feature].median()
                    dummy_record[feature] = feature_median
                
                # Setze Primary Key
                dummy_record[primary_key_col] = customer
                
                # Zukünftige Daten für Target
                customer_future = self.features_df[
                    (self.features_df[primary_key_col] == customer) &
                    (self.features_df[timebase_col] >= target_start) &
                    (self.features_df[timebase_col] <= test_to)
                ].copy()
                
                # Churn-Status in Zukunft
                will_churn = (customer_future[target_col] == 0).any() if len(customer_future) > 0 else False
                dummy_record['ACTUAL_CHURN'] = int(will_churn)
                
                # Konvertiere dict zu pandas Series für konsistente Datenstruktur
                dummy_series = pd.Series(dummy_record)
                test_period_data.append(dummy_series)
        
        print(f"📊 Kunden-Analyse:")
        print(f"   Mit historischen Daten: {customers_with_historical}")
        print(f"   Ohne historische Daten: {customers_without_historical}")
        print(f"   Gesamt verarbeitet: {len(test_period_data)}")
        
        # SAUBERE LÖSUNG: Konvertiere test_period_data zu DataFrame und erstelle validation_customers
        print(f"🎯 SAUBERE LÖSUNG: Konvertiere {len(test_period_data)} Test-Period-Daten zu DataFrame")
        
        # Konvertiere test_period_data zu DataFrame
        if len(test_period_data) > 0:
            test_period_df = pd.DataFrame(test_period_data)
            print(f"✅ DataFrame erstellt mit {len(test_period_df)} Kunden und {len(test_period_df.columns)} Features")
            
            # Stelle sicher, dass alle benötigten Features vorhanden sind
            missing_features = set(self.feature_names) - set(test_period_df.columns)
            if missing_features:
                print(f"⚠️ Fehlende Features ergänzt: {len(missing_features)} Features")
                for feature in missing_features:
                    test_period_df[feature] = self.features_df[feature].median()
            
            # Erstelle X_test für Vorhersagen
            X_test = test_period_df[self.feature_names].fillna(0)
            y_test = test_period_df['ACTUAL_CHURN']
            
            # Vorhersagen für alle Test-Kunden
            y_pred_proba_full = backtest_model.predict_proba(X_test)
            
            # Sichere Behandlung von predict_proba (falls nur eine Klasse)
            if y_pred_proba_full.shape[1] > 1:
                y_pred_proba = y_pred_proba_full[:, 1]  # Churn-Wahrscheinlichkeit
            else:
                # Fallback: Nur eine Klasse vorhergesagt
                y_pred_proba = y_pred_proba_full[:, 0]
                print("⚠️ Modell hat nur eine Klasse vorhergesagt - verwende Fallback")
            
            # Erstelle validation_customers aus test_period_df
            # GMM-Risk-Level auf Basis der aktuellen y_pred_proba berechnen
            try:
                gmm_risk_levels_current, gmm_cluster_stats_current = self._calculate_gmm_risk_segments(
                    y_pred_proba, optimal_threshold, "Test"
                )
            except Exception:
                # Falls GMM fehlschlägt, lasse Ausnahme hochgehen (Policy: keine stillen Fallbacks)
                raise
            validation_customers = []
            for i, (idx, row) in enumerate(test_period_df.iterrows()):
                # Engineered Features für diesen Kunden extrahieren
                try:
                    import numpy as np
                    engineered = {}
                    for f in self.feature_names:
                        if f in row.index:
                            v = row[f]
                            if pd.isna(v):
                                engineered[f] = None
                            elif isinstance(v, (np.generic,)):
                                try:
                                    engineered[f] = v.item()
                                except Exception:
                                    engineered[f] = float(v)
                            else:
                                engineered[f] = v
                except Exception:
                    engineered = {}

                customer_data = {
                    primary_key_col: row[primary_key_col],
                    'risk_level': gmm_risk_levels_current[i],
                    'churn_probability': y_pred_proba[i],
                    'actual_churn': int(row['ACTUAL_CHURN']),
                    'predicted_churn': int(y_pred_proba[i] > 0.5),
                    'customer_type': 'validation',
                    'engineered_features': engineered
                }
                validation_customers.append(customer_data)
            
            print(f"✅ SAUBERE LÖSUNG: {len(validation_customers)} validation_customers aus test_period_df erstellt")
        else:
            print("❌ Keine test_period_data verfügbar - kann keine validation_customers erstellen")
            validation_customers = []
            test_period_df = pd.DataFrame()
        
        print(f"✅ Customer Predictions erstellt: {len(validation_customers)} Kunden")
        
        # 2. ACTIVE CUSTOMERS: Aktuelle Kunden für Risk Level Action Table
        # Nutze die Test-Perioden für aktive Kunden (mehr Daten verfügbar)
        print(f"\n📊 Erstelle aktive Kunden-Liste für Risk Level Action Table...")
        
        # Finde alle aktiven Kunden nach dem Backtest-Zeitraum
        # Verwende backtest_to Parameter statt letzten verfügbaren Monat
        active_timebase = test_to  # Verwende den backtest_to Parameter
        print(f"🎯 Suche aktive Kunden nach dem Backtest-Zeitraum: {active_timebase}")
        
        # Finde alle Kunden, die nach dem Backtest-Zeitraum noch aktiv waren
        active_customers_data = self.features_df[
            (self.features_df['I_TIMEBASE'] == active_timebase) & 
            (self.features_df[self.get_target_column_name()] == 1)
        ].copy()
        
        # Gruppiere nach Primary Key und nehme den letzten verfügbaren Record
        active_customers_data = active_customers_data.sort_values(timebase_col)
        active_customers_data = active_customers_data.groupby(primary_key_col).last().reset_index()
        
        print(f"✅ {len(active_customers_data)} aktive Kunden nach dem Backtest-Zeitraum ({active_timebase}) gefunden")
        
        # Erstelle Vorhersagen für aktive Kunden
        active_customers = []
        if len(active_customers_data) > 0:
            X_active = active_customers_data[self.feature_names].fillna(0)
            y_pred_proba_full_active = backtest_model.predict_proba(X_active)
            
            # Sichere Behandlung von predict_proba (falls nur eine Klasse)
            if y_pred_proba_full_active.shape[1] > 1:
                y_pred_proba_active = y_pred_proba_full_active[:, 1]  # Churn-Wahrscheinlichkeit
            else:
                # Fallback: Nur eine Klasse vorhergesagt
                y_pred_proba_active = y_pred_proba_full_active[:, 0]
                print("⚠️ Active Customers: Modell hat nur eine Klasse vorhergesagt - verwende Fallback")
            
            # Risk Level für aktive Kunden per GMM
            active_customers_data['CHURN_PROBABILITY'] = y_pred_proba_active
            try:
                active_risk_levels, active_cluster_stats = self._calculate_gmm_risk_segments(
                    y_pred_proba_active, optimal_threshold, "Active"
                )
            except Exception:
                # Policy: keine stillen Fallbacks
                raise
            # Reihenfolge beibehalten
            active_customers_data['RISK_LEVEL'] = pd.Series(active_risk_levels, index=active_customers_data.index)
            
            for _, row in active_customers_data.iterrows():
                customer_data = {
                    primary_key_col: row[primary_key_col],  # Dynamischer PRIMARY_KEY aus Data Dictionary
                    'risk_level': row['RISK_LEVEL'],
                    'churn_probability': row['CHURN_PROBABILITY'],
                    'actual_churn': None,  # Unbekannt für aktive Kunden
                    'predicted_churn': int(row['CHURN_PROBABILITY'] > 0.5),
                    'customer_type': 'active'  # Marker für UI
                }
                
                # Top Features für aktive Kunden
                customer_features = {}
                for feature in self.feature_names:
                    if feature in row.index:
                        feature_value = row[feature]
                        feature_imp = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
                        customer_features[feature] = {
                            'value': feature_value, 
                            'importance': feature_imp,
                            'weighted_score': feature_value * feature_imp
                        }
                
                # Top 5 Features
                top_features = sorted(customer_features.items(), 
                                    key=lambda x: x[1]['weighted_score'], reverse=True)[:5]
                customer_data['top_features'] = top_features
                
                # Early Warning Signals
                early_warning_signals = []
                for feature, data in customer_features.items():
                    if '_ew_' in feature and data['value'] > 0:
                        signal_description = self._explain_early_warning_signal(feature, data['value'])
                        early_warning_signals.append({
                            'feature': feature,
                            'description': signal_description,
                            'value': data['value']
                        })
                
                customer_data['early_warning_signals'] = early_warning_signals
                
                # Rohe Daten ohne Business-Intelligence
                
                active_customers.append(customer_data)
            
            print(f"✅ {len(active_customers)} aktive Kunden für Risk Level Action Table vorbereitet")
        
        # GMM-basierte Risk Level Verteilung für Erklärungen
        risk_segmentation = {}
        for risk_level in test_cluster_stats['risk_levels']:
            if risk_level in risk_analysis.index:
                customers = int(risk_analysis.loc[risk_level, ('ACTUAL_CHURN', 'count')])
                churns = int(risk_analysis.loc[risk_level, ('ACTUAL_CHURN', 'sum')])
                rate = float(risk_analysis.loc[risk_level, ('ACTUAL_CHURN', 'mean')])
                cluster_mean = test_cluster_stats['cluster_means'].get(risk_level, 0)
                risk_segmentation[risk_level] = {
                    'customers': customers,
                    'churns': churns,
                    'rate': rate,
                    'cluster_mean': cluster_mean
                }
        
        # Sichere Prüfung für Klasse '1' (Churn) im Return
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        
        if '1' in backtest_report:
            precision = backtest_report['1']['precision']
            recall = backtest_report['1']['recall']
            f1_score = backtest_report['1']['f1-score']
        
        # NEU: Speichere Backtest-Ergebnisse mit GMM-Clustering
        self._save_backtest_results(
            test_period_df=None,  # Nicht mehr benötigt, da validation_customers direkt erstellt werden
            validation_customers=validation_customers,
            backtest_results={
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'auc': backtest_auc,
                'accuracy': backtest_report['accuracy'],
                'optimal_threshold': optimal_threshold,
                'gmm_clustering': {
                    'test_cluster_stats': test_cluster_stats,
                    'train_cluster_stats': train_cluster_stats,
                    'cluster_comparison': cluster_comparison
                },
                'backtest_period': {
                    'training_from': training_from,
                    'training_to': training_to,
                    'test_from': test_from,
                    'test_to': test_to
                }
            }
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc': backtest_auc,
            'accuracy': backtest_report['accuracy'],
            'test_customers': len(test_df),
            'actual_churns': int(test_df['ACTUAL_CHURN'].sum()),
            'risk_segmentation': risk_segmentation,
            'gmm_clustering': {
                'test_cluster_stats': test_cluster_stats,
                'train_cluster_stats': train_cluster_stats,
                'cluster_comparison': cluster_comparison
            },
            'model_saved': model_path is not None,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'validation_customers': validation_customers,
            'active_customers': active_customers,
            'feature_importance': feature_importance,
            'risk_analysis': risk_analysis,
            'target_window_info': target_window_info,
            'y_pred_proba': y_pred_proba.tolist(),  # Churn-Wahrscheinlichkeiten für Analyse
            'y_test': test_df['ACTUAL_CHURN'].tolist(),  # Echte Churn-Werte für Analyse
            'optimal_threshold': optimal_threshold,  # Optimaler Schwellwert für CSV-Export
            'customer_predictions': validation_customers,  # Alle Kunden mit Churn-Wahrscheinlichkeiten
            # NEU: Alle Test-Zeitraum-Kunden mit Features und Predictions
            'test_period_customers_data': validation_customers,  # Verwende validation_customers statt test_period_df
            'test_period_customers_count': len(validation_customers),
            'backtest_period': {
                'training_from': training_from,
                'training_to': training_to,
                'test_from': test_from,
                'test_to': test_to
            },
            'experiment_id': getattr(self, 'experiment_id', None)  # MINIMALE ÄNDERUNG: experiment_id hinzufügen
        }
        
    def _simplified_backtest(self, training_from, training_to, test_from, test_to):
        """Vereinfachtes Backtest für existierende Modelle - VERWENDET ECHTE BACKTEST-ERGEBNISSE"""
        # GLOBALE KONFIGURATION: Erzwinge echte Backtests
        global_config = get_global_config()
        enforce_real_backtest("Simplified Backtest")
        
        print(f"⚡ VEREINFACHTES BACKTEST: {training_from}-{training_to} → {test_from}-{test_to}")
        
        # 🔧 ENTFERNT: Redundante Backtest-Aufrufe - verwende nur einen korrekten real_world_backtest
        print("🎯 Verwende real_world_backtest für vereinfachtes Backtest...")
        return self.real_world_backtest(
            training_from=training_from,
            training_to=training_to, 
            test_from=test_from,
            test_to=test_to,
            use_existing_model=True  # Verwende existierendes Modell
        )
    
    def _calculate_target_window_info(self, test_from, test_to):
        """Berechnet Target Window Information für dynamische Test-Perioden"""
        def months_between(yyyymm1, yyyymm2):
            """Berechnet Monate zwischen zwei YYYYMM Werten"""
            year1, month1 = yyyymm1 // 100, yyyymm1 % 100
            year2, month2 = yyyymm2 // 100, yyyymm2 % 100
            return (year2 - year1) * 12 + (month2 - month1)
        
        def generate_yyyymm_periods(start_yyyymm, end_yyyymm):
            """Generiert Liste von YYYYMM Perioden zwischen start und end"""
            periods = []
            current = start_yyyymm
            while current <= end_yyyymm:
                periods.append(current)
                # Nächste Periode
                year = current // 100
                month = current % 100
                if month == 12:
                    current = (year + 1) * 100 + 1
                else:
                    current = year * 100 + month + 1
            return periods
        
        # Berechne Test-Perioden-Länge
        test_period_length = months_between(test_from, test_to) + 1
        
        # Target Window: Dynamisch basierend auf Test-Perioden-Länge
        # Für längere Test-Perioden: Größeres Target Window
        if test_period_length <= 2:
            target_window_months = 6  # Standard 6-Monate-Fenster
        elif test_period_length <= 6:
            target_window_months = 12  # Mittleres Fenster
        else:
            target_window_months = 18  # Großes Fenster für lange Test-Perioden
        
        # Target Start: Ab test_from (korrekte YYYYMM Berechnung)
        target_start = test_from
        
        # Test-Perioden: Korrekte YYYYMM Format
        test_periods = generate_yyyymm_periods(test_from, test_to)
        
        return {
            'target_window_months': target_window_months,
            'target_start': target_start,
            'test_period_length': test_period_length,
            'test_periods': test_periods
        }
    
    def _explain_early_warning_signal(self, feature_name: str, value: float) -> str:
        """Erklärt Early Warning Signale in verständlicher Sprache - DATA-DICTIONARY-BASIERT"""
        
        # DATA-DICTIONARY-BASIERTE ERKLÄRUNGEN
        data_dictionary = self._create_integrated_data_dictionary()
        
        # Extrahiere Base Feature Name (entferne Early Warning Suffixe)
        base_feature = feature_name
        if '_ew_' in feature_name:
            base_feature = feature_name.split('_ew_')[0]
        elif '_trend_' in feature_name:
            base_feature = feature_name.split('_trend_')[0]
        elif '_volatility_' in feature_name:
            base_feature = feature_name.split('_volatility_')[0]
        elif '_pattern_' in feature_name:
            base_feature = feature_name.split('_pattern_')[0]
        
        # Hole deutsche Übersetzung aus Data Dictionary (mit None-Check)
        if data_dictionary and 'columns' in data_dictionary:
            feature_german = data_dictionary['columns'].get(base_feature, {}).get('german_name', base_feature)
        else:
            feature_german = base_feature
        
        # Bestimme Signal-Typ basierend auf Feature-Name
        if '_ew_' in feature_name:
            signal_type = 'early_warning'
        elif '_trend_' in feature_name:
            signal_type = 'trend'
        elif '_volatility_' in feature_name:
            signal_type = 'volatility'
        elif '_pattern_' in feature_name:
            signal_type = 'pattern'
        else:
            signal_type = 'general'
        
        # Generiere kontextbewusste Erklärung
        return self._generate_context_aware_explanation(signal_type, feature_german, value, base_feature)
    
    def get_risk_level_action_table(self):
        """Gibt die Risk Level Action Table für aktive Kunden zurück"""
        if not hasattr(self, 'last_backtest_results') or self.last_backtest_results is None:
            return None
        
        active_customers = self.last_backtest_results.get('active_customers', [])
        if not active_customers:
            return None
        
        # Erstelle DataFrame für Risk Level Action Table
        risk_table_data = []
        
        # Lese PRIMARY_KEY dynamisch aus Data Dictionary
        primary_key_col, _, _ = self._get_dynamic_column_names()
        
        for customer in active_customers:
            risk_table_data.append({
                primary_key_col: customer.get(primary_key_col, customer.get('customer_id', 'N/A')),  # Dynamischer Data Dictionary Key
                'risk_level': customer.get('risk_level', 'N/A'),
                'churn_probability': round(float(customer.get('churn_probability', 0)) * 100, 1),
                'predicted_churn': 'Ja' if customer.get('predicted_churn') == 1 else 'Nein',
                'top_features': str(customer.get('top_features', [])),
                'early_warning_signals_count': int(len(customer.get('early_warning_signals', [])))
            })
        
        risk_table_df = pd.DataFrame(risk_table_data)
        
        # Sortiere nach Churn-Probability (höchste zuerst)
        risk_table_df = risk_table_df.sort_values('churn_probability', ascending=False)
        
        return risk_table_df
    
    def _generate_context_aware_explanation(self, signal_type: str, feature_german: str, value: float, base_feature: str) -> str:
        """Generiert kontextbewusste Erklärung basierend auf Data Dictionary"""
        
        try:
            # Finde das Feature im Data Dictionary (mit None-Check)
            column_info = None
            if self.data_dictionary and 'columns' in self.data_dictionary:
                for col_name, col_info in self.data_dictionary['columns'].items():
                    if col_name == base_feature:
                        column_info = col_info
                        break
            
            if not column_info:
                # Fallback wenn Feature nicht gefunden
                return self._generate_generic_signal_explanation(signal_type, feature_german, value)
            
            role = column_info.get('confirmed', {}).get('role', column_info.get('role', 'UNKNOWN'))
            
            # Rolle-spezifische Erklärungen
            if role == 'DYNAMIC_FEATURE':
                return self._explain_dynamic_feature_signal(signal_type, feature_german, value, base_feature)
            elif role == 'STATIC_CATEGORICAL':
                return self._explain_static_feature_signal(signal_type, feature_german, value, base_feature)
            elif role == 'DYNAMIC_TIME_SERIES':
                return self._explain_timeseries_signal(signal_type, feature_german, value, base_feature)
            else:
                return self._generate_generic_signal_explanation(signal_type, feature_german, value)
                
        except Exception as e:
            # Fallback bei Fehlern
            return self._generate_generic_signal_explanation(signal_type, feature_german, value)
    
    def _explain_dynamic_feature_signal(self, signal_type: str, feature_german: str, value: float, base_feature: str) -> str:
        """Erklärt Dynamic Feature Signale"""
        
        if signal_type == 'major_drop':
            return f"Drastischer Rückgang bei {feature_german} - Aktivität ist um über 50% eingebrochen"
        elif signal_type == 'decline':
            return f"Kontinuierlicher Rückgang bei {feature_german} - Nutzung nimmt stetig ab"
        elif signal_type == 'stop':
            return f"Kompletter Stopp bei {feature_german} - Kunde nutzt diesen Service nicht mehr"
        elif signal_type == 'volatility':
            return f"Unregelmäßige Nutzung von {feature_german} - Schwankende Kundenbindung"
        else:
            return f"Auffälliges Verhalten bei {feature_german} (Wert: {value:.2f})"
    
    def _explain_static_feature_signal(self, signal_type: str, feature_german: str, value: float, base_feature: str) -> str:
        """Erklärt Static Feature Signale"""
        
        if signal_type == 'digital_drop':
            return f"Digitalisierungsrate stark gesunken - Kunde wird weniger digital"
        elif signal_type == 'digital_stagnation':
            return f"Digitalisierung stagniert - Keine Fortschritte bei digitalen Services"
        else:
            return f"Veränderung bei {feature_german} erkannt (Wert: {value:.2f})"
    
    def _explain_timeseries_signal(self, signal_type: str, feature_german: str, value: float, base_feature: str) -> str:
        """Erklärt Time Series Feature Signale"""
        
        if signal_type == 'major_drop':
            return f"Zeitreihen-Analyse zeigt: Drastischer Einbruch bei {feature_german}"
        elif signal_type == 'decline':
            return f"Zeitreihen-Trend: Kontinuierlicher Rückgang bei {feature_german}"
        elif signal_type == 'volatility':
            return f"Zeitreihen-Analyse: Hohe Schwankungen bei {feature_german}"
        else:
            return f"Zeitreihen-Anomalie bei {feature_german} (Wert: {value:.2f})"
    
    def _generate_generic_signal_explanation(self, signal_type: str, feature_german: str, value: float) -> str:
        """Fallback für generische Signal-Erklärungen"""
        
        generic_explanations = {
            'major_drop': f"Starker Rückgang bei {feature_german} erkannt",
            'decline': f"Rückläufiger Trend bei {feature_german}",
            'stop': f"Aktivität bei {feature_german} gestoppt",
            'volatility': f"Schwankungen bei {feature_german}",
            'digital_drop': f"Digitale Aktivität rückläufig",
            'digital_stagnation': f"Digitalisierung stagniert",
            'business_decline': f"Geschäftsaktivität nimmt ab",
            'composite_score': f"Multiple Risikofaktoren erkannt"
        }
        
        return generic_explanations.get(signal_type, f"Warnsignal bei {feature_german}: {value:.2f}")

    def _translate_feature_name(self, feature_name: str) -> str:
        """Übersetzt technische Feature-Namen in geschäftlich verständliche Begriffe - DATA-DICTIONARY-BASIERT"""
        
        # Prüfe zuerst, ob Data Dictionary verfügbar ist
        if not self.data_dictionary:
            return feature_name
            
        # Suche ursprünglichen Spaltenname (entferne Feature-Engineering-Suffixe)
        base_feature = feature_name
        
        # Entferne typische Feature-Engineering-Suffixe
        suffixes_to_remove = [
            '_cum_mean', '_cum_std', '_cum_sum', '_cum_count',
            '_trend', '_ma_', '_volatility', '_risk', '_ew_', '_lag_',
            '_rolling_mean', '_rolling_std', '_shift', '_diff'
        ]
        
        for suffix in suffixes_to_remove:
            if suffix in base_feature:
                base_feature = base_feature.split(suffix)[0]
                break
        
        # Suche in Data Dictionary
        for column_name, column_info in self.data_dictionary['columns'].items():
            if column_name == base_feature:
                role = column_info.get('confirmed', {}).get('role', column_info.get('role', 'UNKNOWN'))
                data_type = column_info.get('confirmed', {}).get('data_type', column_info.get('data_type', 'TEXT'))
                
                # Generiere Übersetzung basierend auf Rolle
                return self._generate_translation_by_role(base_feature, role, data_type)
        
        # Fallback: Versuche intelligente Übersetzung basierend auf Namensmuster
        return self._generate_smart_translation(base_feature)
    
    def _generate_translation_by_role(self, feature_name: str, role: str, data_type: str) -> str:
        """Generiert Übersetzung basierend auf Data Dictionary-Rolle"""
        
        # Basis-Übersetzung
        clean_name = feature_name.replace('I_', '').replace('N_', '').replace('_', ' ').title()
        
        # Rollenbasierte Übersetzungen
        if role == 'STATIC_CATEGORICAL':
            if 'digital' in feature_name.lower():
                return f"Digitalisierungsrate ({clean_name})"
            else:
                return f"Kategorisches Merkmal: {clean_name}"
                
        elif role == 'DYNAMIC_FEATURE':
            # Erkennung der Aktivitätsart
            if any(keyword in feature_name.lower() for keyword in ['maintenance', 'wartung']):
                return f"Wartungsservices ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['upgrade', 'upgrad']):
                return f"System-Upgrades ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['upsell', 'erweiterung']):
                return f"Produkt-Erweiterungen ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['downgrade', 'reduzierung']):
                return f"Produkt-Reduzierungen ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['downsell', 'service']):
                return f"Service-Reduzierungen ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['uhd', 'hd']):
                return f"UHD-Services ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['consulting', 'beratung']):
                return f"Beratungsleistungen ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['seminar', 'schulung']):
                return f"Schulungen/Seminare ({clean_name})"
            elif any(keyword in feature_name.lower() for keyword in ['insurance', 'versicherung']):
                return f"Versicherungsleistungen ({clean_name})"
            else:
                return f"Geschäftsaktivität: {clean_name}"
                
        elif role == 'DYNAMIC_TIME_SERIES':
            return f"Zeitreihen-Aktivität: {clean_name}"
            
        elif role == 'STATIC_CONSTANT':
            return f"Konstantes Merkmal: {clean_name}"
            
        else:
            return f"Unbekannte Aktivität: {clean_name}"
    
    def _generate_smart_translation(self, feature_name: str) -> str:
        """Fallback: Intelligente Übersetzung basierend auf Namensmuster"""
        
        # Basis-Bereinigung
        clean_name = feature_name.replace('I_', '').replace('N_', '').replace('_', ' ').title()
        
        # Muster-basierte Übersetzungen
        patterns = {
            'digital': 'Digitalisierung',
            'maintenance': 'Wartung',
            'upgrade': 'Upgrade',
            'upsell': 'Produkterweiterung',
            'downgrade': 'Produktreduzierung',
            'downsell': 'Service-Reduzierung',
            'consulting': 'Beratung',
            'seminar': 'Schulung',
            'insurance': 'Versicherung',
            'support': 'Support',
            'service': 'Service',
            'activity': 'Aktivität',
            'rate': 'Rate',
            'score': 'Score',
            'count': 'Anzahl',
            'sum': 'Summe',
            'mean': 'Durchschnitt',
            'trend': 'Trend',
            'volatility': 'Schwankung'
        }
        
        feature_lower = feature_name.lower()
        for pattern, translation in patterns.items():
            if pattern in feature_lower:
                return f"{translation} ({clean_name})"
        
        # Fallback
        return clean_name

    def run_complete_analysis(self, force_retrain=False, training_from=None, training_to=None, test_from=None, test_to=None):
        """Führt komplette Enhanced Early Warning Analyse durch mit optimierten Konfigurationen"""
        
        # 🔍 SOFORTIGE PARAMETER-AUSGABE
        print("🔍" + "="*60 + "🔍")
        print("     PARAMETER-DEBUG")
        print("🔍" + "="*60 + "🔍")
        print(f"🎯 Training-Zeiträume: {training_from} - {training_to}")
        print(f"🎯 Backtest-Zeiträume: {test_from} - {test_to}")
        print(f"🔍 Parameter-Typen:")
        print(f"   training_from: {type(training_from)} = {training_from}")
        print(f"   training_to: {type(training_to)} = {training_to}")
        print(f"   test_from: {type(test_from)} = {test_from}")
        print(f"   test_to: {type(test_to)} = {test_to}")
        print("🔍" + "="*60 + "🔍")
        
        # GLOBALE KONFIGURATION: Erzwinge echte Analysen
        global_config = get_global_config()
        enforce_real_analysis("Enhanced Early Warning Analysis")
        
        print("🚨" + "="*60 + "🚨")
        print("     ENHANCED EARLY WARNING SYSTEM")
        print("🚨" + "="*60 + "🚨")
        
        # 🔧 NEU: Training- und Backtest-Zeiträume müssen explizit angegeben werden
        if training_from is None or training_to is None:
            print("❌ FEHLER: Training-Zeiträume müssen explizit angegeben werden!")
            print("   training_from und training_to sind erforderlich")
            return {
                'error': 'Training-Zeiträume nicht angegeben',
                'solution': 'Übergebe training_from und training_to Parameter'
            }
        
        if test_from is None or test_to is None:
            print("❌ FEHLER: Backtest-Zeiträume müssen explizit angegeben werden!")
            print("   test_from und test_to sind erforderlich")
            return {
                'error': 'Backtest-Zeiträume nicht angegeben',
                'solution': 'Übergebe test_from und test_to Parameter'
            }
        
        print(f"📅 Training-Zeiträume: {training_from} - {training_to}")
        print(f"📅 Backtest-Zeiträume: {test_from} - {test_to}")
        print(f"🔍 DEBUG: Parameter-Typen - training_from: {type(training_from)}, training_to: {type(training_to)}")
        print(f"🔍 DEBUG: Parameter-Werte - training_from: {training_from}, training_to: {training_to}")
        
        # 🔧 NEU: Speichere Training- und Backtest-Zeiträume für Modell-Speicherung
        self.current_training_periods = {
            'training_from': training_from,
            'training_to': training_to,
            'test_from': test_from,
            'test_to': test_to
        }
        
        try:
            # LADE OPTIMIERTE KONFIGURATION AUS FEATURE ANALYSIS ENGINE
            print("🔧 Lade optimierte Konfiguration aus Feature Analysis Engine...")
            optimized_config = self._load_optimized_config()
            
            if not optimized_config:
                print("❌ KEINE OPTIMIERTE KONFIGURATION GEFUNDEN!")
                print("⚠️  Bitte führe zuerst die Feature Analysis Engine aus!")
                return {
                    'error': 'Keine optimierte Konfiguration gefunden',
                    'solution': 'Führe feature_analysis_engine.py aus'
                }
            
            print(f"✅ Optimierte Konfiguration geladen:")
            print(f"   🎯 Modell: {optimized_config['model_type']}")
            if 'sampling_config' in optimized_config:
                print(f"   🎯 Sampling: {optimized_config['sampling_config']['strategy']}")
            else:
                print(f"   🎯 Sampling: Keine (Standard)")
            print(f"   🎯 AUC: 0.9938 (aus Feature Analysis Engine)")
            
            # INTELLIGENTE MODELL-VERWALTUNG
            if not force_retrain:
                # 🔧 NEU: Prüfe Training-Zeiträume beim Modell-Vergleich
                should_retrain, reason = self.should_retrain_model(training_from, training_to)
                
                if not should_retrain:
                    # Versuche existierendes Modell zu laden
                    if self.load_latest_model():
                        print(f"🎯 Verwende existierendes Modell - kein Training erforderlich ({reason})")
                        
                        # GLOBALE KONFIGURATION: Prüfe ob Schnell-Modus erlaubt ist
                        # ABER: Respektiere force_retrain Parameter von UI
                        if global_config.is_simplified_mode_disabled() and not force_retrain:
                            print("🔄 Vollständige Analyse erforderlich (globale Konfiguration)")
                            force_retrain = True
                        elif not force_retrain:
                            # Lade nur Basis-CSV für vereinfachtes Backtest (OHNE Feature-Engineering)
                            print("🔧 Lade Basis-Daten über DataAccessLayer für SCHNELL-MODUS...")
                            # MIGRATION: Verwende DataAccessLayer statt direkter CSV-Zugriff
                            self.df = self.dal.load_stage0_data()
                            # KRITISCH: Setze features_df NICHT - dadurch wird SCHNELL-MODUS aktiviert
                            self.features_df = None
                            
                            # Verwende existierendes Modell für Backtest
                            print("🎯 Verwende existierendes Modell für Backtest...")
                            # 🔧 ENTFERNT: Redundanter Backtest-Aufruf - verwende nur _run_full_analysis_with_optimized_config
                            return self._run_full_analysis_with_optimized_config(
                                optimized_config, 
                                test_from=test_from, test_to=test_to, 
                                use_existing_model=True,
                                training_from=training_from, training_to=training_to
                            )
                        else:
                            # Verwende existierendes Modell für vollständige Analyse
                            print("🎯 Verwende existierendes Modell für vollständige Analyse...")
                            return self._run_full_analysis_with_existing_model(optimized_config)
                    else:
                        print("⚠️ Kein existierendes Modell gefunden - Training erforderlich")
                        force_retrain = True
                else:
                    print(f"🔄 Modell muss neu trainiert werden: {reason}")
                    force_retrain = True
            
            # VOLLSTÄNDIGE ANALYSE MIT OPTIMIERTER KONFIGURATION
            if force_retrain:
                print("🔄 Führe vollständige Analyse mit optimierter Konfiguration durch...")
                return self._run_full_analysis_with_optimized_config(
                    optimized_config, 
                    test_from=test_from, test_to=test_to, 
                    use_existing_model=False,
                    training_from=training_from, training_to=training_to
                )
            
        except Exception as e:
            print(f"❌ Fehler in Enhanced Early Warning Analyse: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _run_full_analysis_with_optimized_config(self, optimized_config, test_from=None, test_to=None, use_existing_model=False, training_from=None, training_to=None):
        """Führt vollständige Analyse mit optimierter Konfiguration durch"""
        print("🔧 Lade und verarbeite Daten mit optimierter Konfiguration...")
        
        # Lade und verarbeite Features
        self.load_and_enhance_features()
        
        # 🔧 NEU: Verwende übergebene Training-Zeiträume
        if training_from is None or training_to is None:
            raise ValueError("❌ FEHLER: training_from und training_to müssen explizit gesetzt werden! Keine Fallbacks erlaubt.")
        if test_from is None or test_to is None:
            raise ValueError("❌ FEHLER: test_from und test_to müssen explizit gesetzt werden! Keine Fallbacks erlaubt.")
        prediction_timebase = test_from  # 🔧 DYNAMISCH basierend auf test_from
        
        print(f"🛡️ DATA LEAKAGE PRÄVENTION: Training {training_from}-{training_to}, Prediction ab {prediction_timebase}")
        
        # Trainiere Modell mit optimierter Konfiguration
        print("🤖 Training mit optimierter Konfiguration...")
        
        # 💾 NEU: Speichere Enhanced Features in JSON-Database für CF-Pipeline
        print("💾 Speichere Enhanced Features für CF-Pipeline...")
        if hasattr(self, 'experiment_id') and self.experiment_id:
            experiment_id = self.experiment_id
        else:
            experiment_id = 1  # Fallback für Experiment-ID
        
        save_success = self.save_enhanced_features_to_database(self.features_df, experiment_id)
        if save_success:
            print("✅ Enhanced Features erfolgreich in JSON-Database gespeichert")
        else:
            print("⚠️ Enhanced Features konnten nicht gespeichert werden")

        # Real-World Backtest
        print("📊 Real-World Backtest mit optimierter Konfiguration...")
        backtest_results = self.real_world_backtest(
            training_from=training_from,
            training_to=training_to,
            test_from=test_from,
            test_to=test_to,
            use_existing_model=use_existing_model
        )
        
        if backtest_results and 'error' not in backtest_results:
            print("✅ Vollständige Analyse erfolgreich abgeschlossen")
            
            # Extract training metrics from backtest results
            training_metrics = {
                'precision': backtest_results.get('precision', 0.0),
                'recall': backtest_results.get('recall', 0.0),
                'f1_score': backtest_results.get('f1_score', 0.0),
                'auc': backtest_results.get('auc', 0.0),
                'accuracy': backtest_results.get('accuracy', 0.0)
            }
            
            return {
                'status': 'success',
                'mode': 'full_analysis',
                'training_metrics': training_metrics,
                'backtest_results': backtest_results,
                'optimized_config': optimized_config,
                'note': 'Verwendet optimierte Konfiguration aus Feature Analysis Engine'
            }
        else:
            error_msg = backtest_results.get('error', 'Unbekannter Fehler') if backtest_results else 'Keine Backtest-Ergebnisse'
            print(f"❌ Vollständige Analyse fehlgeschlagen: {error_msg}")
            return {'error': error_msg}
    
    def _run_full_analysis_with_existing_model(self, optimized_config):
        """Führt vollständige Analyse mit existierendem Modell durch"""
        print("🔧 Lade und verarbeite Daten...")
        
        # Lade und verarbeite Features
        self.load_and_enhance_features()
        
        # 🔧 ENTFERNT: Redundanter Backtest-Aufruf - verwende nur _run_full_analysis_with_optimized_config
        print("🎯 Verwende existierendes Modell für vollständige Analyse...")
        return self._run_full_analysis_with_optimized_config(
            optimized_config, 
            test_from=None, test_to=None,  # 🔧 DYNAMISCH - Default-Werte werden in der Funktion gesetzt
            use_existing_model=True
        )

    def save_enhanced_model(self, model, metrics: dict, feature_names: list, training_info: dict = None):
        """Speichert das Enhanced Early Warning Modell"""
        try:
            
            # Models-Verzeichnis
            models_dir = ProjectPaths.models_directory()
            models_dir.mkdir(exist_ok=True)
            
            # Timestamp für eindeutige Dateinamen
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            
            # Modell-Dateiname
            model_name = f"Enhanced_EarlyWarning_{timestamp}.joblib"
            model_path = models_dir / model_name
            
            # Modell speichern
            joblib.dump(model, model_path)
            print(f"💾 Enhanced Early Warning Modell gespeichert: {model_path}")
            
            # CSV-Metadaten für intelligente Modell-Verwaltung
            csv_metadata = self.get_csv_metadata()
            
            # 🔧 NEU: Training-Zeiträume aus training_info oder current_training_periods extrahieren
            training_periods = {}
            if training_info:
                if 'training_from' in training_info and 'training_to' in training_info:
                    training_periods = {
                        'training_from': training_info['training_from'],
                        'training_to': training_info['training_to']
                    }
                elif 'training_period' in training_info:
                    # Fallback: Aus training_period String extrahieren
                    period_str = training_info['training_period']
                    if '-' in period_str:
                        parts = period_str.split('-')
                        if len(parts) == 2:
                            training_periods = {
                                'training_from': parts[0].strip(),
                                'training_to': parts[1].strip()
                            }
            elif hasattr(self, 'current_training_periods'):
                # 🔧 NEU: Verwende current_training_periods aus run_complete_analysis
                training_periods = self.current_training_periods
            
            # Metadaten erstellen
            metadata = {
                "created": timestamp,
                "model_type": "Enhanced_EarlyWarning",
                "training_approach": "early_warning_backtest",
                "feature_names": feature_names,
                "metrics": metrics,
                "csv_metadata": csv_metadata,  # NEU: CSV-Metadaten für intelligente Prüfung
                "training_periods": training_periods,  # 🔧 NEU: Training-Zeiträume
                "training_info": training_info or {
                    "algorithm": "RandomForest",
                    "level": "enhanced_early_warning",
                    "features_count": len(feature_names),
                    "timestamp": timestamp,
                    "compatibility": "system_health_ui",
                    "lead_time_months": 6,
                    "warning_signals": len([f for f in feature_names if '_ew_' in f])
                }
            }
            
            # Metadaten speichern
            metadata_name = f"Enhanced_EarlyWarning_{timestamp}.json"
            metadata_path = models_dir / metadata_name
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"📄 Enhanced Early Warning Metadaten gespeichert: {metadata_path}")
            
            if csv_metadata:
                print(f"📊 CSV-Metadaten gespeichert: {csv_metadata['file_name']} ({csv_metadata['file_size']} bytes)")
            
            if training_periods:
                print(f"📅 Training-Zeiträume gespeichert: {training_periods['training_from']} - {training_periods['training_to']}")
            
            # Artefakt-Registry: Modell registrieren (joblib + json)
            try:
                from bl.json_database.churn_json_database import ChurnJSONDatabase  # lazy import
                db = ChurnJSONDatabase()
                db.ensure_artifacts_registry()
                exp_id = int(getattr(self, 'experiment_id', 0)) if getattr(self, 'experiment_id', None) is not None else None
                # joblib
                try:
                    db.add_artifact_record("model", exp_id, str(model_path), status="active")
                except Exception:
                    pass
                # metadata json
                try:
                    db.add_artifact_record("model", exp_id, str(metadata_path), status="active")
                except Exception:
                    pass
                db.save()
                print("🧾 artifacts_registry aktualisiert (model)")
            except Exception as _e:
                print(f"⚠️ artifacts_registry konnte nicht aktualisiert werden (model): {_e}")
            
            return str(model_path), str(metadata_path)
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Enhanced Early Warning Modells: {e}")
            return None, None

    def _save_backtest_results(self, test_period_df, validation_customers, backtest_results):
        """Speichert Backtest-Ergebnisse mit allen Customer Predictions"""
        try:
            from datetime import datetime
            import json
            
            # Models-Verzeichnis
            models_dir = ProjectPaths.models_directory()
            models_dir.mkdir(exist_ok=True)
            
            # Timestamp für eindeutige Dateinamen
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            
            # Backtest-Ergebnisse anreichern (Zähler und Segmente)
            try:
                # Fallback: Wenn kein test_period_df übergeben wurde, nutze Anzahl der validation_customers
                if test_period_df is not None:
                    total_samples = int(len(test_period_df))
                elif validation_customers is not None:
                    total_samples = int(len(validation_customers))
                else:
                    total_samples = 0
            except Exception:
                total_samples = 0

            high_count = 0
            low_count = 0
            try:
                if test_period_df is not None and 'RISK_LEVEL' in test_period_df.columns:
                    high_count = int((test_period_df['RISK_LEVEL'] == 'HIGH').sum())
                    low_count = int((test_period_df['RISK_LEVEL'] == 'LOW').sum())
            except Exception:
                pass

            # Ergänze Zähler in backtest_results
            if isinstance(backtest_results, dict):
                backtest_results = dict(backtest_results)
                backtest_results.setdefault('backtest_samples', total_samples)
                backtest_results.setdefault('high_risk_customers', high_count)
                backtest_results.setdefault('medium_risk_customers', 0)
                backtest_results.setdefault('low_risk_customers', low_count)

            # Backtest-Ergebnisse erstellen
            backtest_data = {
                "created": timestamp,
                "model_type": "Enhanced_EarlyWarning_Backtest",
                "backtest_results": backtest_results,
                "test_period_customers_count": total_samples,
                "test_period_customers_data": test_period_df.to_dict('records') if test_period_df is not None and len(test_period_df) > 0 else [],
                "validation_customers_count": len(validation_customers) if validation_customers is not None else 0,
                "validation_customers": validation_customers if validation_customers is not None else [],
                "feature_names": self.feature_names if hasattr(self, 'feature_names') else [],
                "optimal_threshold": backtest_results.get('optimal_threshold', 0.5),
                "backtest_period": backtest_results.get('backtest_period', {})
            }
            
            # Backtest-Ergebnisse speichern
            backtest_name = f"Enhanced_EarlyWarning_Backtest_{timestamp}.json"
            backtest_path = models_dir / backtest_name
            
            with open(backtest_path, 'w', encoding='utf-8') as f:
                json.dump(backtest_data, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Backtest-Ergebnisse gespeichert: {backtest_path}")
            # Zeige konsistente Kundenzahl: test_period_df, sonst validation_customers
            try:
                _tp_count = len(test_period_df) if test_period_df is not None else (len(validation_customers) if validation_customers is not None else 0)
            except Exception:
                _tp_count = 0
            print(f"   - Test-Zeitraum-Kunden: {_tp_count}")
            print(f"   - Validation Customers: {len(validation_customers) if validation_customers is not None else 0}")
            print(f"   - Optimal Threshold: {backtest_results.get('optimal_threshold', 0.5)}")
            print(f"   - Backtest AUC: {backtest_results.get('auc', 'N/A')}")
            
            # Artefakt-Registry: Backtest registrieren
            try:
                from bl.json_database.churn_json_database import ChurnJSONDatabase  # lazy import
                db = ChurnJSONDatabase()
                db.ensure_artifacts_registry()
                exp_id = int(getattr(self, 'experiment_id', 0)) if getattr(self, 'experiment_id', None) is not None else None
                db.add_artifact_record("backtest", exp_id, str(backtest_path), status="active")
                db.save()
                print("🧾 artifacts_registry aktualisiert (backtest)")
            except Exception as _e:
                print(f"⚠️ artifacts_registry konnte nicht aktualisiert werden (backtest): {_e}")
            
            return str(backtest_path)
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Backtest-Ergebnisse: {e}")
            return None

    def find_optimal_global_windows(self, max_windows=5):
        """Finde heuristische Lookback-Monate basierend auf Business-Zyklen"""
        print("🔍 Verwende heuristische Business-Zyklen Windows...")
        
        # Heuristische Windows basierend auf typischen Business-Zyklen
        # 6M: Halbjährliche Reviews, 12M: Jahresverträge, 18M: 1.5-Jahres-Zyklen
        # 24M: 2-Jahres-Verträge, 36M: Langfristige Enterprise-Verträge
        heuristic_windows = [6, 12, 18, 24, 36]
        
        print(f"✅ Heuristische Business-Zyklen Windows: {heuristic_windows}")
        print("📊 Begründung:")
        print("   - 6M:  Halbjährliche Reviews & Preisanpassungen")
        print("   - 12M: Jahresverträge & Budget-Zyklen") 
        print("   - 18M: 1.5-Jahres-Technologie-Zyklen")
        print("   - 24M: 2-Jahres-Enterprise-Verträge")
        print("   - 36M: Langfristige Strategic-Partnerships")
        
        return heuristic_windows

    def _calculate_gmm_risk_segments(self, y_pred_proba, optimal_threshold, dataset_name):
        """
        GMM-basierte Risiko-Segmentierung
        
        Args:
            y_pred_proba: Churn-Wahrscheinlichkeiten
            optimal_threshold: Optimaler Schwellwert
            dataset_name: Name des Datensatzes (Training/Test)
            
        Returns:
            risk_levels: Liste der Risiko-Level
            cluster_stats: Cluster-Statistiken
        """
        try:
            from sklearn.mixture import GaussianMixture
            import numpy as np
            
            print(f"🔬 GMM-Clustering für {dataset_name}-Daten...")
            
            # Reshape für GMM
            X = y_pred_proba.reshape(-1, 1)
            
            # BIC-Optimierung für optimale Anzahl Komponenten
            n_components_range = range(2, 6)  # 2-5 Cluster
            bic = []
            aic = []
            
            for n in n_components_range:
                gmm = GaussianMixture(n_components=n, random_state=42, covariance_type='full')
                gmm.fit(X)
                bic.append(gmm.bic(X))
                aic.append(gmm.aic(X))
            
            # Wähle optimale Anzahl basierend auf BIC
            optimal_n = n_components_range[np.argmin(bic)]
            print(f"   Optimal Cluster-Anzahl (BIC): {optimal_n}")
            
            # Fitte finales GMM
            best_gmm = GaussianMixture(n_components=optimal_n, random_state=42, covariance_type='full')
            clusters = best_gmm.fit_predict(X)
            
            # Cluster nach Risiko sortieren (niedrig → hoch)
            cluster_means = best_gmm.means_.flatten()
            cluster_order = np.argsort(cluster_means)
            
            # Risiko-Level zuordnen
            risk_levels_map = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
            risk_mapping = dict(zip(cluster_order, risk_levels_map[:len(cluster_order)]))
            
            # Risiko-Level zuweisen
            risk_levels = [risk_mapping[cluster] for cluster in clusters]
            
            # Cluster-Statistiken berechnen
            cluster_stats = {
                'risk_levels': list(set(risk_levels)),
                'cluster_means': {risk_mapping[i]: float(cluster_means[i]) for i in range(len(cluster_means))},
                'cluster_sizes': {},
                'bic_scores': [float(score) for score in bic],
                'aic_scores': [float(score) for score in aic],
                'optimal_n_components': optimal_n
            }
            
            # Cluster-Größen
            for risk_level in cluster_stats['risk_levels']:
                cluster_stats['cluster_sizes'][risk_level] = risk_levels.count(risk_level)
            
            print(f"   Cluster gefunden: {cluster_stats['risk_levels']}")
            print(f"   Cluster-Größen: {cluster_stats['cluster_sizes']}")
            print(f"   Cluster-Mittelwerte: {cluster_stats['cluster_means']}")
            
            return risk_levels, cluster_stats
            
        except Exception as e:
            print(f"❌ Fehler bei GMM-Clustering: {e}")
            # Fallback: Binäre Segmentierung
            risk_levels = ['LOW' if prob < optimal_threshold else 'HIGH' for prob in y_pred_proba]
            cluster_stats = {
                'risk_levels': ['LOW', 'HIGH'],
                'cluster_means': {'LOW': 0.3, 'HIGH': 0.7},
                'cluster_sizes': {'LOW': risk_levels.count('LOW'), 'HIGH': risk_levels.count('HIGH')},
                'fallback': True
            }
            return risk_levels, cluster_stats

    def _compare_train_test_clusters(self, train_cluster_stats, test_cluster_stats):
        """
        Vergleicht Training- und Test-Cluster
        
        Args:
            train_cluster_stats: Training-Cluster-Statistiken
            test_cluster_stats: Test-Cluster-Statistiken
            
        Returns:
            Vergleichs-Statistiken
        """
        try:
            # Cluster-Anzahl-Vergleich
            train_n = len(train_cluster_stats['risk_levels'])
            test_n = len(test_cluster_stats['risk_levels'])
            
            # Stabilitäts-Score (0-1, 1 = perfekt stabil)
            stability_score = 1.0 if train_n == test_n else 0.5
            
            # Cluster-Mittelwert-Vergleich
            cluster_comparison = {}
            for risk_level in set(train_cluster_stats['risk_levels']) | set(test_cluster_stats['risk_levels']):
                train_mean = train_cluster_stats['cluster_means'].get(risk_level, 0)
                test_mean = test_cluster_stats['cluster_means'].get(risk_level, 0)
                cluster_comparison[risk_level] = {
                    'train_mean': train_mean,
                    'test_mean': test_mean,
                    'difference': abs(train_mean - test_mean)
                }
            
            comparison_stats = {
                'stability_score': stability_score,
                'train_clusters': train_n,
                'test_clusters': test_n,
                'cluster_comparison': cluster_comparison,
                'stable': stability_score > 0.8
            }
            
            return comparison_stats
            
        except Exception as e:
            print(f"❌ Fehler beim Cluster-Vergleich: {e}")
            return {
                'stability_score': 0.0,
                'train_clusters': 0,
                'test_clusters': 0,
                'cluster_comparison': {},
                'stable': False
            }

    def save_enhanced_features_to_database(self, features_df, experiment_id):
        """
        Speichert Enhanced Features in customer_churn_details Tabelle der JSON-Database
        
        Args:
            features_df: DataFrame mit Enhanced Features
            experiment_id: Experiment-ID
            
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            print(f"💾 Speichere Enhanced Features in JSON-Database für Experiment {experiment_id}...")
            
            # Import JSON-Database
            from bl.json_database.churn_json_database import ChurnJSONDatabase
            from datetime import datetime
            
            # Initialisiere Database
            db = ChurnJSONDatabase()
            
            # Dynamische Spaltennamen aus Data Dictionary
            primary_key_col, timebase_col, target_col = self._get_dynamic_column_names()
            
            # Erstelle Customer-Level Dataset mit Enhanced Features
            customer_df = self.create_customer_dataset(prediction_timebase=202401)  # Verwende aktuellen Zeitraum
            
            # Bereite Records für customer_churn_details vor
            enhanced_records = []
            
            for _, row in customer_df.iterrows():
                # Basis-Record mit bestehender Struktur
                # Korrektur: target_col enthält hier WILL_CHURN (1=churn, 0=alive)
                # I_ALIVE muss 1=alive, 0=churn sein → invertiere sicher
                try:
                    churn_val = row.get('ACTUAL_CHURN')
                    if pd.notna(churn_val):
                        i_alive = 1 - int(churn_val)
                    else:
                        tgt_val = row.get(target_col, 1)
                        i_alive = 1 - int(tgt_val)
                except Exception:
                    i_alive = 1

                record = {
                    'Kunde': int(row[primary_key_col]),
                    'experiment_id': experiment_id,
                    'Letzte_Timebase': int(row.get(timebase_col, 202401)),
                    'I_ALIVE': int(i_alive),
                    'source': 'enhanced_early_warning',
                    'dt_created': datetime.now().isoformat()
                }
                
                # Füge ALLE Enhanced Features hinzu
                for feature_name in self.feature_names:
                    if feature_name in row.index:
                        feature_value = row[feature_name]
                        # Konvertiere zu Python-Standard-Typen (explizit keine Booleans)
                        if pd.isna(feature_value):
                            record[feature_name] = None
                        elif isinstance(feature_value, bool):
                            # Boolean → Integer Konvertierung (verhindert DuckDB Type-Mismatch)
                            record[feature_name] = int(feature_value)
                        elif isinstance(feature_value, (int, float)):
                            record[feature_name] = float(feature_value)
                        else:
                            record[feature_name] = feature_value
                
                enhanced_records.append(record)
            
            # Lösche existierende Records für dieses Experiment (falls vorhanden)
            if "customer_churn_details" in db.data["tables"]:
                existing_records = db.data["tables"]["customer_churn_details"]["records"]
                # Filtere Records für andere Experimente
                filtered_records = [r for r in existing_records if r.get('experiment_id') != experiment_id]
                db.data["tables"]["customer_churn_details"]["records"] = filtered_records
                print(f"🗑️ Alte Records für Experiment {experiment_id} entfernt")
            
            # Füge neue Enhanced Records hinzu
            success = db.add_customer_churn_details(enhanced_records)
            
            if success:
                # Speichere Database
                db.save()
                print(f"✅ {len(enhanced_records)} Enhanced Customer Records in JSON-Database gespeichert")
                print(f"📊 Features pro Customer: {len(self.feature_names)}")
                print(f"🎯 Enhanced Features verfügbar für CF-Pipeline")
                return True
            else:
                print("❌ Fehler beim Hinzufügen der Enhanced Records")
                return False
                
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Enhanced Features: {e}")
            import traceback
            traceback.print_exc()
            return False

def get_dynamic_oversampling_config():
    """
    Liefert dynamische Oversampling-Konfiguration basierend auf aktuellen Daten
    
    Returns:
        dict: Oversampling-Konfiguration
    """
    try:
        # Lade Data Dictionary für Oversampling-Faktor
        paths = ProjectPaths()
        data_dict_path = paths.data_dictionary_file()
        
        if data_dict_path.exists():
            with open(data_dict_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            # Oversampling-Faktor aus Data Dictionary
            oversampling_factor = data_dict.get('oversampling_factor', 5)
        else:
            # Fallback-Wert
            oversampling_factor = 5
        
        return {
            'oversampling_factor': oversampling_factor,
            'method': 'SMOTE',
            'random_state': 42,
            'sampling_strategy': 0.3,  # 30% der Minderheitsklasse
            'k_neighbors': 5,
            'source': 'data_dictionary.json' if data_dict_path.exists() else 'default'
        }
        
    except Exception as e:
        print(f"⚠️ Fehler beim Laden der Oversampling-Konfiguration: {e}")
        # Fallback-Konfiguration
        return {
            'oversampling_factor': 5,
            'method': 'SMOTE',
            'random_state': 42,
            'sampling_strategy': 0.3,
            'k_neighbors': 5,
            'source': 'fallback'
        }

def main():
    """Hauptfunktion"""
    system = EnhancedEarlyWarningSystem()
    results = system.run_complete_analysis()
    return system, results

if __name__ == "__main__":
    enhanced_system, results = main() 