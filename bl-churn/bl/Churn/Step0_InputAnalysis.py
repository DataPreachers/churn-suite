"""
STUFE 0: CSV-ANALYSE & LERNFAHIGKEIT
====================================

Funktionen:
- CSV-Struktur-Analyse mit Hash-basierter Erkennung
- Automatische Data Dictionary Generierung
- Algorithmus-Performance-Analyse + Hyperparameter-Optimierung
- CSV-Hash-basierte Speicherung und Wiederverwendung
- Struktur-Ähnlichkeits-Erkennung für ähnliche CSVs
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Füge Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.paths_config import ProjectPaths


class CSVStructureAnalyzer:
    """CSV-Struktur-Analyse mit Hash-basierter Lernfähigkeit"""
    
    def __init__(self):
        self.cache_dir = os.path.join(ProjectPaths.dynamic_system_outputs_directory(), "stage0_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.similarity_threshold = 0.8
        
    def calculate_csv_hash(self, file_path: str) -> str:
        """Berechnet Hash basierend auf CSV-Struktur (Spalten, Datentypen, nicht Werte)"""
        try:
            # Lese nur die ersten 1000 Zeilen für Struktur-Analyse
            # Versuche zuerst mit Semikolon als Trennzeichen
            try:
                df_sample = pd.read_csv(file_path, nrows=1000, sep=';')
            except:
                # Fallback auf Standard-Trennzeichen
                df_sample = pd.read_csv(file_path, nrows=1000)
            
            # Erstelle Struktur-Signatur
            structure_info = {
                'columns': list(df_sample.columns),
                'dtypes': {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
                'null_counts': df_sample.isnull().sum().to_dict(),
                'unique_counts': {col: df_sample[col].nunique() for col in df_sample.columns},
                'file_size': os.path.getsize(file_path)
            }
            
            # Hash der Struktur
            structure_str = json.dumps(structure_info, sort_keys=True)
            return hashlib.md5(structure_str.encode()).hexdigest()
            
        except Exception as e:
            print(f"❌ Fehler beim Hash-Berechnung: {e}")
            return None
    
    def calculate_structure_similarity(self, hash1: str, hash2: str) -> float:
        """Berechnet Ähnlichkeit zwischen zwei CSV-Strukturen"""
        try:
            # Lade Struktur-Informationen
            cache_file1 = os.path.join(self.cache_dir, f"{hash1}.json")
            cache_file2 = os.path.join(self.cache_dir, f"{hash2}.json")
            
            if not (os.path.exists(cache_file1) and os.path.exists(cache_file2)):
                return 0.0
            
            with open(cache_file1, 'r') as f:
                data1 = json.load(f)
            with open(cache_file2, 'r') as f:
                data2 = json.load(f)
            
            # Vergleiche Spalten
            columns1 = set(data1['structure']['columns'])
            columns2 = set(data2['structure']['columns'])
            
            if len(columns1) == 0 or len(columns2) == 0:
                return 0.0
            
            # Jaccard-Ähnlichkeit
            intersection = len(columns1.intersection(columns2))
            union = len(columns1.union(columns2))
            
            similarity = intersection / union if union > 0 else 0.0
            return similarity
            
        except Exception as e:
            print(f"❌ Fehler bei Ähnlichkeits-Berechnung: {e}")
            return 0.0
    
    def analyze_csv_structure(self, file_path: str, force_reanalysis: bool = False) -> Dict[str, Any]:
        """Hauptfunktion: Analysiert CSV-Struktur mit Hash-basierter Lernfähigkeit"""
        print("🎯 STUFE 0 - CSV-ANALYSE")
        print(f"📁 Datei: {os.path.basename(file_path)}")
        
        # 1. CSV-Hash berechnen
        csv_hash = self.calculate_csv_hash(file_path)
        if not csv_hash:
            return {'error': 'Hash-Berechnung fehlgeschlagen'}
        
        print(f"🔍 CSV-Hash: {csv_hash[:12]}...")
        
        # 2. Cache prüfen
        cache_file = os.path.join(self.cache_dir, f"{csv_hash}.json")
        
        if not force_reanalysis and os.path.exists(cache_file):
            print("💾 Lade gespeicherte Ergebnisse...")
            try:
                with open(cache_file, 'r') as f:
                    cached_results = json.load(f)
                
                # Prüfe ob Cache-Datei vollständig ist
                if self._validate_cache(cached_results):
                    print("✅ Gespeicherte Ergebnisse geladen")
                    cached_results['loaded_from_cache'] = True
                    cached_results['analysis_time'] = 0.1  # Cache-Ladezeit
                    return cached_results
                else:
                    print("⚠️ Cache-Datei beschädigt, führe neue Analyse durch")
                    
            except Exception as e:
                print(f"⚠️ Cache-Laden fehlgeschlagen: {e}")
        
        # 3. Vollständige Analyse durchführen
        print("🔬 Führe vollständige Analyse durch...")
        start_time = datetime.now()
        
        try:
            # CSV laden (alle Zeilen für Stufe 0)
            # Versuche zuerst mit Semikolon als Trennzeichen
            try:
                df = pd.read_csv(file_path, sep=';')  # Alle Zeilen laden
            except:
                # Fallback auf Standard-Trennzeichen
                df = pd.read_csv(file_path)  # Alle Zeilen laden
            print(f"📊 Spalten: {len(df.columns)} gefunden (alle Zeilen: {len(df)} Zeilen)")
            
            # Struktur analysieren
            structure_analysis = self._analyze_structure(df)
            
            # Data Dictionary generieren
            data_dictionary = self._generate_data_dictionary(df)
            
            # Algorithmus-Performance testen
            algorithm_performance = self._test_algorithms(df)
            
            # Komplette Daten als JSON-Array hinzufügen
            print("💾 Konvertiere alle Daten zu JSON...")
            
            # Explizite Integer-Konvertierung für I_Alive (verhindert Boolean-Inferenz)
            if 'I_Alive' in df.columns:
                df['I_Alive'] = df['I_Alive'].astype(int)
                print("🔧 I_Alive explizit als Integer konvertiert")
            
            data_json = df.to_dict('records')  # Liste von Dictionaries
            
            # Ergebnisse zusammenstellen
            results = {
                'csv_hash': csv_hash,
                'file_path': file_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'structure': structure_analysis,
                'data_dictionary': data_dictionary,
                'algorithm_performance': algorithm_performance,
                'complete_data': data_json,  # Alle Zeilen als JSON-Array
                'loaded_from_cache': False,
                'analysis_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Ergebnisse speichern
            self._save_results(results, cache_file)
            
            print(f"✅ Analyse abgeschlossen in {results['analysis_time']:.1f} Sekunden")
            return results
            
        except Exception as e:
            error_msg = f"❌ Analyse fehlgeschlagen: {e}"
            print(error_msg)
            return {'error': error_msg}
    
    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert die Struktur des DataFrames"""
        structure = {
            'columns': list(df.columns),
            'rows': len(df),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_categories': {}
        }
        
        # Kategorisiere Spalten
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            # Erkenne deutsche Dezimalzahlen und konvertiere zu Float
            if dtype == 'object':
                # Teste ob es sich um deutsche Dezimalzahlen handelt
                try:
                    sample_values = df[col].dropna().head(10).astype(str)
                    has_german_decimals = any(',' in str(val) for val in sample_values if pd.notna(val))
                    
                    if has_german_decimals:
                        # Konvertiere zu Float für deutsche Dezimalzahlen
                        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                        structure['dtypes'][col] = 'float64'
                        structure['column_categories'][col] = 'NUMERICAL'
                        print(f"🔍 {col}: Deutsche Dezimalzahlen → Float konvertiert")
                    else:
                        # Behalte als String für kategorische Features
                        structure['column_categories'][col] = 'CATEGORICAL' if unique_count <= 50 else 'TEXT'
                except:
                    # Fallback: Behalte als String
                    structure['column_categories'][col] = 'CATEGORICAL' if unique_count <= 50 else 'TEXT'
            elif dtype in ['int64', 'float64']:
                if unique_count <= 20:
                    structure['column_categories'][col] = 'CATEGORICAL'
                else:
                    structure['column_categories'][col] = 'NUMERICAL'
            else:
                structure['column_categories'][col] = 'OTHER'
        
        return structure
    
    def _generate_data_dictionary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generiert automatisches Data Dictionary"""
        data_dict = {
            'features': {},
            'feature_groups': {
                'static_features': [],
                'dynamic_features': [],
                'categorical_features': [],
                'numerical_features': []
            }
        }
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            # Bestimme Feature-Typ
            if dtype in ['object', 'string']:
                if unique_count <= 50:
                    feature_type = 'CATEGORICAL_FEATURE'
                    data_dict['feature_groups']['categorical_features'].append(col)
                else:
                    feature_type = 'TEXT_FEATURE'
            elif dtype in ['int64', 'float64']:
                if unique_count <= 20:
                    feature_type = 'CATEGORICAL_FEATURE'
                    data_dict['feature_groups']['categorical_features'].append(col)
                else:
                    feature_type = 'NUMERICAL_FEATURE'
                    data_dict['feature_groups']['numerical_features'].append(col)
            else:
                feature_type = 'OTHER_FEATURE'
            
            # Feature-Definition
            data_dict['features'][col] = {
                'role': feature_type,
                'data_type': dtype.upper(),
                'description': f'Automatisch erkannt: {feature_type}',
                'dtype': dtype,
                'unique_count': unique_count,
                'null_count': null_count,
                'null_percentage': (null_count / len(df)) * 100
            }
        
        return data_dict
    
    def _test_algorithms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Testet verschiedene Algorithmen und optimiert Hyperparameter"""
        print("⚙️ Teste Algorithmen...")
        
        # Vereinfachte Version - keine komplexe Algorithmus-Tests in Step0
        # Die Feature Analysis Engine wird das später übernehmen
        
        algorithm_results = {
            'status': 'skipped',
            'reason': 'Algorithmus-Tests werden von Feature Analysis Engine übernommen',
            'recommendations': [
                'Verwende Feature Analysis Engine für detaillierte Algorithmus-Optimierung',
                'Step0 fokussiert sich auf Datenstruktur-Analyse'
            ]
        }
        
        return algorithm_results
    
    def _save_results(self, results: Dict[str, Any], cache_file: str) -> bool:
        """Speichert Ergebnisse in Cache"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Ergebnisse gespeichert: {os.path.basename(cache_file)}")
            return True
        except Exception as e:
            print(f"❌ Speicherung fehlgeschlagen: {e}")
            return False
    
    def _validate_cache(self, cached_data: Dict[str, Any]) -> bool:
        """Validiert Cache-Daten"""
        required_keys = ['csv_hash', 'structure', 'data_dictionary', 'algorithm_performance', 'complete_data']
        return all(key in cached_data for key in required_keys)
    
    def list_cached_analyses(self) -> List[Dict[str, Any]]:
        """Listet alle gespeicherten Analysen"""
        cached_analyses = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                cache_file = os.path.join(self.cache_dir, filename)
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    cached_analyses.append({
                        'hash': data.get('csv_hash', filename.replace('.json', '')),
                        'file_path': data.get('file_path', 'Unbekannt'),
                        'timestamp': data.get('analysis_timestamp', 'Unbekannt'),
                        'columns': len(data.get('structure', {}).get('columns', [])),
                        'analysis_time': data.get('analysis_time', 0)
                    })
                    
                except Exception as e:
                    print(f"⚠️ Cache-Datei {filename} beschädigt: {e}")
        
        return cached_analyses


def analyze_csv_input(file_path: str, force_reanalysis: bool = False) -> Dict[str, Any]:
    """Hauptfunktion für CSV-Analyse"""
    analyzer = CSVStructureAnalyzer()
    return analyzer.analyze_csv_structure(file_path, force_reanalysis)


if __name__ == "__main__":
    # Test mit aktueller CSV
    input_file = ProjectPaths.main_churn_data_file()
    
    if os.path.exists(input_file):
        print("🚀 Starte Stufe 0 Analyse...")
        results = analyze_csv_input(input_file)
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"❌ Datei nicht gefunden: {input_file}") 