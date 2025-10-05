#!/usr/bin/env python3
"""
Feature Analysis Engine - Intelligente Feature-Konfiguration & Algorithmus-Optimierung
====================================================================================

Ziel: Automatische Erkennung und Lösung von Feature-Engineering-Problemen
Basierend auf: docs/UI_DEVELOPMENT_CONTEXT.md

MIGRATION: Migriert zur DataSchema-Architektur für konsistente Dateninterpretation

Hauptprobleme die gelöst werden:
1. Digitalisierung-Problem: STATIC_FEATURE wird durch Customer-Aggregation konstant
2. Feature-Typ-Erkennung: Intelligente Klassifizierung basierend auf Datenanalyse
3. Algorithmus-Optimierung: Automatische Hyperparameter-Tuning
4. Meta-Konfiguration: Generierung von optimierten Konfigurationsdateien
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ML Imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# Gradient Boosting
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost nicht verfügbar")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM nicht verfügbar")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost nicht verfügbar")

# Projekt-Pfade - Verwende ProjectPaths für korrekte Pfad-Ermittlung
# Setze Python-Pfad für Import von config und bl Modulen
project_root = Path(__file__).parent.parent  # Ein Verzeichnis höher zum Projekt-Root
sys.path.insert(0, str(project_root))

from config.paths_config import ProjectPaths
from config.global_config import get_global_config

# MIGRATION: DataSchema Architecture imports
from config.data_access_layer import get_data_access
from config.data_schema import get_data_schema

class FeatureAnalysisEngine:
    """
    Intelligente Feature-Analyse und Konfiguration
    
    MIGRATION: Migriert zur DataSchema-Architektur
    """
    
    def __init__(self):
        self.paths = ProjectPaths()
        
        # MIGRATION: DataSchema Architecture Integration
        self.dal = get_data_access()
        self.schema = get_data_schema()
        
        self.df = None
        self.data_dictionary = None
        self.analysis_results = {}
        self.optimization_config = {}
        
        print("🔒 Feature Analysis Engine initialisiert mit DataSchema-Validierung")
        
    def load_data(self):
        """
        Lädt Daten und Data Dictionary mit DataAccessLayer
        
        MIGRATION: Verwendet DataAccessLayer statt direkter JSON/CSV-Zugriffe
        """
        print("📊 Lade Daten für Feature-Analyse über DataAccessLayer...")
        
        try:
            # MIGRATION: Verwende DataAccessLayer für automatisch validierte Daten
            raw_df = self.dal.load_stage0_data()
            print(f"✅ Daten über DataAccessLayer geladen: {len(raw_df)} Zeilen, {len(raw_df.columns)} Spalten")
            
            # MIGRATION: Data Dictionary über DataSchema
            self.data_dictionary = self.schema.data_dictionary
            print(f"✅ Data Dictionary über DataSchema geladen: {len(self.data_dictionary.get('columns', {}))} Features")
            
            # 🚀 ENHANCED EARLY WARNING SYSTEM für Customer-Level Dataset
            print("🚀 Führe Enhanced Early Warning System aus...")
            
            try:
                # Import Enhanced Early Warning System (bereits migriert)
                from bl.Churn.enhanced_early_warning import EnhancedEarlyWarningSystem
                
                # Erstelle Enhanced Early Warning System
                eews = EnhancedEarlyWarningSystem()
                eews.df = raw_df  # Verwende DataAccessLayer-Daten
                eews.data_dictionary = self.data_dictionary  # Verwende DataSchema Dictionary
                
                # Führe Feature Engineering aus
                print("🔧 Erstelle Features mit Enhanced Early Warning...")
                features_df, feature_names = eews._create_integrated_features()
                
                # Setze features_df im Enhanced Early Warning System
                eews.features_df = features_df
                eews.feature_names = feature_names
                
                # Erstelle Customer-Level Dataset
                print("👥 Erstelle Customer-Level Dataset...")
                customer_df = eews.create_customer_dataset(prediction_timebase=202401)
                
                # Verwende Customer-Level Dataset für Tests
                self.df = customer_df
                self.feature_names = feature_names
                
                print(f"✅ Customer-Level Dataset erstellt: {len(self.df)} Kunden, {len(self.feature_names)} Features")
                
            except Exception as e:
                print(f"⚠️ Enhanced Early Warning fehlgeschlagen: {e}")
                print("🔄 Verwende Rohdaten als Fallback...")
                self.df = raw_df
                self.feature_names = None
                print(f"✅ Fallback: Rohdaten verwendet: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")
                
        except Exception as e:
            print(f"❌ Fehler beim Laden über DataAccessLayer: {e}")
            self.df = None
            self.data_dictionary = {}
            
    def analyze_feature_characteristics(self) -> Dict[str, Any]:
        """
        Analysiert Feature-Charakteristiken und erkennt Probleme
        """
        print("\n🔍 Analysiere Feature-Charakteristiken...")
        
        analysis = {
            'feature_stats': {},
            'problems_detected': [],
            'recommendations': [],
            'feature_type_suggestions': {}
        }
        
        for column in self.df.columns:
            if column in self.data_dictionary.get('columns', {}):
                feature_info = self.data_dictionary['columns'][column]
                current_role = feature_info.get('role', 'UNKNOWN')
                
                # Berechne Feature-Statistiken
                stats = self._calculate_feature_statistics(column)
                analysis['feature_stats'][column] = stats
                
                # Erkenne Probleme
                problems = self._detect_feature_problems(column, stats, current_role)
                if problems:
                    analysis['problems_detected'].extend(problems)
                
                # Generiere Empfehlungen
                recommendations = self._generate_feature_recommendations(column, stats, current_role)
                if recommendations:
                    analysis['recommendations'].extend(recommendations)
                
                # Vorschlag für Feature-Typ
                suggested_type = self._suggest_feature_type(column, stats, current_role)
                analysis['feature_type_suggestions'][column] = suggested_type
                
        return analysis
    
    def _calculate_feature_statistics(self, column: str) -> Dict[str, Any]:
        """Berechnet detaillierte Feature-Statistiken"""
        stats = {
            'dtype': str(self.df[column].dtype),
            'unique_count': self.df[column].nunique(),
            'null_count': self.df[column].isnull().sum(),
            'null_percentage': (self.df[column].isnull().sum() / len(self.df)) * 100,
            'min_value': self.df[column].min() if self.df[column].dtype in ['int64', 'float64'] else None,
            'max_value': self.df[column].max() if self.df[column].dtype in ['int64', 'float64'] else None,
            'mean_value': self.df[column].mean() if self.df[column].dtype in ['int64', 'float64'] else None,
            'std_value': self.df[column].std() if self.df[column].dtype in ['int64', 'float64'] else None,
            'zero_count': (self.df[column] == 0).sum() if self.df[column].dtype in ['int64', 'float64'] else None,
            'zero_percentage': ((self.df[column] == 0).sum() / len(self.df)) * 100 if self.df[column].dtype in ['int64', 'float64'] else None,
        }
        
        # Zeitliche Analyse für numerische Features
        if self.df[column].dtype in ['int64', 'float64']:
            # Gruppiere nach Kunde und analysiere zeitliche Muster
            primary_key = self._get_primary_key_column()
            timebase_col = self._get_timebase_column()
            
            if primary_key and timebase_col:
                grouped = self.df.groupby(primary_key)[column]
                stats['temporal_analysis'] = {
                    'customer_variance': grouped.var().mean(),
                    'customer_std': grouped.std().mean(),
                    'customer_range': (grouped.max() - grouped.min()).mean(),
                    'temporal_correlation': self._calculate_temporal_correlation(column, primary_key, timebase_col)
                }
        
        return stats
    
    def _detect_feature_problems(self, column: str, stats: Dict[str, Any], current_role: str) -> List[Dict[str, Any]]:
        """Erkennt spezifische Feature-Probleme"""
        problems = []
        
        # Problem 1: Digitalisierung-Problem (STATIC_FEATURE mit wenigen Werten)
        if current_role == 'STATIC_FEATURE' and stats['unique_count'] <= 20:
            problems.append({
                'type': 'DIGITALIZATION_PROBLEM',
                'severity': 'HIGH',
                'description': f'Feature {column} ist als STATIC_FEATURE markiert, hat aber nur {stats["unique_count"]} verschiedene Werte. '
                              f'Customer-level Aggregation wird konstante Features erzeugen, die durch Quality Control entfernt werden.',
                'impact': 'Feature wird im ML-Modell nicht verwendet',
                'solution': 'Feature-Typ zu DYNAMIC_FEATURE ändern oder spezielle Behandlung implementieren'
            })
        
        # Problem 2: Konstante Features
        if stats['unique_count'] == 1:
            problems.append({
                'type': 'CONSTANT_FEATURE',
                'severity': 'HIGH',
                'description': f'Feature {column} ist konstant (nur 1 Wert)',
                'impact': 'Feature wird durch Quality Control entfernt',
                'solution': 'Feature als EXCLUDED_FEATURE markieren'
            })
        
        # Problem 3: Sehr wenige verschiedene Werte
        if stats['unique_count'] <= 5 and current_role in ['DYNAMIC_FEATURE', 'STATIC_FEATURE']:
            problems.append({
                'type': 'LOW_VARIANCE_FEATURE',
                'severity': 'MEDIUM',
                'description': f'Feature {column} hat nur {stats["unique_count"]} verschiedene Werte',
                'impact': 'Geringe prädiktive Kraft',
                'solution': 'Feature-Engineering erweitern oder als EXCLUDED_FEATURE markieren'
            })
        
        # Problem 4: Hohe Null-Werte
        if stats['null_percentage'] > 50:
            problems.append({
                'type': 'HIGH_NULL_VALUES',
                'severity': 'MEDIUM',
                'description': f'Feature {column} hat {stats["null_percentage"]:.1f}% Null-Werte',
                'impact': 'Viele fehlende Werte reduzieren Datenqualität',
                'solution': 'Imputation-Strategie implementieren'
            })
        
        # Problem 5: Sehr hohe Null-Werte
        if stats['null_percentage'] > 90:
            problems.append({
                'type': 'EXTREME_NULL_VALUES',
                'severity': 'HIGH',
                'description': f'Feature {column} hat {stats["null_percentage"]:.1f}% Null-Werte',
                'impact': 'Feature ist praktisch unbrauchbar',
                'solution': 'Als EXCLUDED_FEATURE markieren'
            })
        
        return problems
    
    def _generate_feature_recommendations(self, column: str, stats: Dict[str, Any], current_role: str) -> List[Dict[str, Any]]:
        """Generiert Empfehlungen für Feature-Optimierung"""
        recommendations = []
        
        # Empfehlung 1: Feature-Typ-Änderung für Low-Variance Features
        if current_role == 'STATIC_FEATURE' and stats['unique_count'] <= 20:
            recommendations.append({
                'type': 'FEATURE_TYPE_CHANGE',
                'priority': 'HIGH',
                'description': f'Ändere {column} von STATIC_FEATURE zu DYNAMIC_FEATURE',
                'reason': f'Low-Variance Feature ({stats["unique_count"]} Werte) führt zu konstanten Customer-Aggregationen',
                'implementation': 'Data Dictionary anpassen und Feature-Engineering-Logik erweitern',
                'category': 'LOW_VARIANCE_FEATURE'
            })
        
        # Empfehlung 2: Spezielle Behandlung für Low-Variance Features
        if stats['unique_count'] <= 10 and current_role in ['DYNAMIC_FEATURE', 'STATIC_FEATURE']:
            # Bestimme Feature-Typ
            data_type = self.data_dictionary.get('columns', {}).get(column, {}).get('data_type', 'NUMERIC')
            
            if data_type == 'CATEGORICAL':
                recommendations.append({
                    'type': 'SPECIAL_TREATMENT',
                    'priority': 'MEDIUM',
                    'description': f'Implementiere spezielle Behandlung für kategorisches Feature {column}',
                    'reason': f'Kategorisches Low-Variance Feature ({stats["unique_count"]} Kategorien) erfordert One-Hot-Encoding und Kategorie-Analyse',
                    'implementation': 'One-Hot-Encoding für Top-Kategorien, Kategorie-Änderungs-Detektion, Stabilitäts-Metriken',
                    'category': 'LOW_VARIANCE_CATEGORICAL_FEATURE'
                })
            else:
                recommendations.append({
                    'type': 'SPECIAL_TREATMENT',
                    'priority': 'MEDIUM',
                    'description': f'Implementiere spezielle Behandlung für numerisches Feature {column}',
                    'reason': f'Numerisches Low-Variance Feature ({stats["unique_count"]} Werte) erfordert angepasste Feature-Engineering-Strategie',
                    'implementation': 'Binarisierung, Kategorisierung, Änderungs-Detektion und Stabilitäts-Metriken',
                    'category': 'LOW_VARIANCE_NUMERIC_FEATURE'
                })
        
        # Empfehlung 3: Imputation für Null-Werte
        if stats['null_percentage'] > 20 and stats['null_percentage'] <= 50:
            recommendations.append({
                'type': 'IMPUTATION_STRATEGY',
                'priority': 'MEDIUM',
                'description': f'Implementiere Imputation für {column}',
                'reason': f'{stats["null_percentage"]:.1f}% Null-Werte reduzieren Datenqualität',
                'implementation': 'Median-Imputation für numerische, Mode-Imputation für kategorische Features'
            })
        
        return recommendations
    
    def _suggest_feature_type(self, column: str, stats: Dict[str, Any], current_role: str) -> Dict[str, Any]:
        """Schlägt optimalen Feature-Typ basierend auf Datenanalyse vor"""
        suggestion = {
            'current_role': current_role,
            'suggested_role': current_role,
            'confidence': 0.8,
            'reasoning': []
        }
        
        # Logik für Feature-Typ-Vorschlag
        if current_role == 'STATIC_FEATURE' and stats['unique_count'] <= 20:
            suggestion['suggested_role'] = 'DYNAMIC_FEATURE'
            suggestion['confidence'] = 0.9
            suggestion['reasoning'].append('Wenige verschiedene Werte - besser als DYNAMIC_FEATURE behandeln')
        
        elif stats['unique_count'] == 1:
            suggestion['suggested_role'] = 'EXCLUDED_FEATURE'
            suggestion['confidence'] = 1.0
            suggestion['reasoning'].append('Konstantes Feature - sollte ausgeschlossen werden')
        
        elif stats['null_percentage'] > 90:
            suggestion['suggested_role'] = 'EXCLUDED_FEATURE'
            suggestion['confidence'] = 0.95
            suggestion['reasoning'].append('Extrem viele Null-Werte - Feature unbrauchbar')
        
        return suggestion
    
    def _get_primary_key_column(self) -> Optional[str]:
        """Ermittelt Primary Key Spalte"""
        for col, info in self.data_dictionary.get('columns', {}).items():
            if info.get('role') == 'PRIMARY_KEY':
                return col
        return None
    
    def _get_timebase_column(self) -> Optional[str]:
        """Ermittelt Timebase Spalte"""
        for col, info in self.data_dictionary.get('columns', {}).items():
            if info.get('role') == 'TIMEBASE':
                return col
        return None
    
    def _calculate_temporal_correlation(self, column: str, primary_key: str, timebase_col: str) -> float:
        """Berechnet temporale Korrelation für Feature"""
        try:
            # Sortiere nach Kunde und Zeit
            sorted_df = self.df.sort_values([primary_key, timebase_col])
            
            # Berechne Korrelation zwischen Feature und Zeit für jeden Kunden
            correlations = []
            for customer in sorted_df[primary_key].unique():
                customer_data = sorted_df[sorted_df[primary_key] == customer]
                if len(customer_data) > 1:
                    correlation = customer_data[column].corr(customer_data[timebase_col])
                    if not pd.isna(correlation):
                        correlations.append(correlation)
            
            return np.mean(correlations) if correlations else 0.0
        except:
            return 0.0
    
    def analyze_algorithm_performance(self) -> Dict[str, Any]:
        """
        Analysiert Algorithmus-Performance und optimiert Hyperparameter
        """
        print("\n🤖 Analysiere Algorithmus-Performance...")
        
        analysis = {
            'current_config': {},
            'performance_metrics': {},
            'optimization_suggestions': [],
            'hyperparameter_recommendations': {}
        }
        
        # Lade aktuelle Konfiguration
        analysis['current_config'] = self._load_current_algorithm_config()
        
        # Analysiere Daten-Charakteristiken für Algorithmus-Optimierung
        data_characteristics = self._analyze_data_for_algorithm_optimization()
        
        # Generiere Algorithmus-Empfehlungen
        algorithm_recommendations = self._generate_algorithm_recommendations(data_characteristics)
        analysis['optimization_suggestions'] = algorithm_recommendations
        
        # Teste alle Model+Sampling-Kombinationen
        if hasattr(self, 'df') and self.df is not None:
            # Bereite Daten vor basierend auf Data Dictionary
            target_col = self._get_target_column()
            if target_col and target_col in self.df.columns:
                # Verwende Feature-Namen falls verfügbar, sonst Data Dictionary
                if hasattr(self, 'feature_names') and self.feature_names:
                    feature_cols = [col for col in self.feature_names if col in self.df.columns]
                    print(f"🔧 Verwende {len(feature_cols)} Feature-Namen aus Enhanced Early Warning")
                else:
                    # Fallback: Verwende Data Dictionary für Feature-Klassifikation
                    feature_cols = []
                    for col, info in self.data_dictionary.get('columns', {}).items():
                        if col in self.df.columns and info.get('role') not in ['PRIMARY_KEY', 'TIMEBASE', 'TARGET']:
                            feature_cols.append(col)
                    print(f"🔧 Verwende {len(feature_cols)} Features aus Data Dictionary")
                
                # Bereite Features vor - Customer-Level Dataset ist bereits verarbeitet
                X_df = self.df[feature_cols].copy()
                
                # Konvertiere kategorische Features zu numerischen Codes
                for col in X_df.columns:
                    col_info = self.data_dictionary.get('columns', {}).get(col, {})
                    feature_type = col_info.get('feature_type', 'UNKNOWN')
                    
                    if feature_type == 'CATEGORICAL_FEATURE' and X_df[col].dtype == 'object':
                        X_df[col] = pd.Categorical(X_df[col]).codes
                
                X = X_df.fillna(0).values
                y = self.df[target_col].astype(int).values
                feature_names = list(X_df.columns)
                
                print(f"📊 Customer-Level Dataset: {len(X)} Kunden, {X.shape[1]} Features")
                print(f"🎯 Target-Verteilung: {pd.Series(y).value_counts().to_dict()}")
                
                # 1. Temporär alle excluded features für dominante Feature-Erkennung aktivieren
                original_excluded_roles = {}
                excluded_features = []
                
                for col, info in self.data_dictionary.get('columns', {}).items():
                    if info.get('role') == 'EXCLUDED_FEATURE':
                        original_excluded_roles[col] = info['role']
                        self.data_dictionary['columns'][col]['role'] = 'DYNAMIC_FEATURE'
                        self.data_dictionary['columns'][col]['feature_group'] = 'dynamic_features'
                        excluded_features.append(col)
                
                if excluded_features:
                    print(f"🔧 Temporär {len(excluded_features)} excluded features für dominante Feature-Erkennung aktiviert")
                
                # 2. Dominante Features erkennen
                dominant_features = self._detect_dominant_features(X, y, feature_names)
                
                # 2. Dominante Features aus Testmenge entfernen
                if dominant_features:
                    # Indizes der dominanten Features finden
                    dominant_indices = [feature_names.index(feat) for feat in dominant_features]
                    
                    # Features aus X entfernen
                    X_filtered = np.delete(X, dominant_indices, axis=1)
                    feature_names_filtered = [feat for feat in feature_names if feat not in dominant_features]
                    
                    print(f"🔧 Entferne {len(dominant_features)} dominante Features aus Testmenge")
                    print(f"📊 Gefilterte Testmenge: {len(X_filtered)} Kunden, {X_filtered.shape[1]} Features")
                    
                    # Data Dictionary aktualisieren - dominante Features zu excluded verschieben
                    for feat in dominant_features:
                        if feat in self.data_dictionary.get('columns', {}):
                            self.data_dictionary['columns'][feat]['role'] = 'EXCLUDED_FEATURE'
                            self.data_dictionary['columns'][feat]['feature_group'] = 'excluded_features'
                            print(f"📝 {feat} zu EXCLUDED_FEATURE verschoben")
                    
                    # Alle ursprünglich excluded features wieder zu excluded verschieben, falls sie nicht dominant waren
                    for col in excluded_features:
                        if col not in dominant_features:
                            self.data_dictionary['columns'][col]['role'] = 'EXCLUDED_FEATURE'
                            self.data_dictionary['columns'][col]['feature_group'] = 'excluded_features'
                            print(f"📝 {col} wieder zu EXCLUDED_FEATURE verschoben (nicht dominant)")
                    
                    # Aktualisierte Feature-Gruppen
                    self._update_feature_groups(self.data_dictionary)
                    
                    X = X_filtered
                    feature_names = feature_names_filtered
                
                # 3. Führe echtes Modell-Training durch
                print("🤖 Starte echtes Modell-Training...")
                model_results = self._test_all_model_sampling_combinations(X, y)
                analysis['model_sampling_results'] = model_results
                
            else:
                analysis['model_sampling_results'] = {'error': 'Target-Spalte nicht gefunden'}
        else:
            analysis['model_sampling_results'] = {'error': 'Keine Daten verfügbar'}
        
        # Hyperparameter-Optimierung
        hyperparameter_recs = self._optimize_hyperparameters(data_characteristics)
        analysis['hyperparameter_recommendations'] = hyperparameter_recs
        
        return analysis
    
    def _load_current_algorithm_config(self) -> Dict[str, Any]:
        """Lädt aktuelle Algorithmus-Konfiguration"""
        config = {
            'model_type': 'RandomForestClassifier',
            'class_weight': 'balanced',
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Versuche aktuelle Konfiguration aus Dateien zu laden
        try:
            # Suche nach gespeicherten Modellen
            models_dir = self.paths.models_directory()
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith('.json') and 'Enhanced_EarlyWarning' in file:
                        with open(os.path.join(models_dir, file), 'r') as f:
                            model_info = json.load(f)
                            if 'model_config' in model_info:
                                config.update(model_info['model_config'])
                                break
        except Exception as e:
            print(f"⚠️ Konnte aktuelle Konfiguration nicht laden: {e}")
        
        return config
    
    def _analyze_data_for_algorithm_optimization(self) -> Dict[str, Any]:
        """Analysiert Daten-Charakteristiken für Algorithmus-Optimierung"""
        characteristics = {
            'sample_size': len(self.df),
            'feature_count': len([col for col in self.df.columns if col not in [self._get_primary_key_column(), self._get_timebase_column(), self._get_target_column()]]),
            'class_imbalance': {},
            'feature_correlation': {},
            'data_sparsity': 0.0,
            'sampling_analysis': {}
        }
        
        # Klassen-Ungleichgewicht analysieren
        target_col = self._get_target_column()
        if target_col:
            target_counts = self.df[target_col].value_counts()
            characteristics['class_imbalance'] = {
                'majority_class': target_counts.max(),
                'minority_class': target_counts.min(),
                'imbalance_ratio': target_counts.max() / target_counts.min(),
                'minority_percentage': (target_counts.min() / len(self.df)) * 100
            }
        
        # Feature-Korrelationen analysieren
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            correlation_matrix = self.df[numeric_features].corr()
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            characteristics['feature_correlation']['high_correlations'] = high_correlations
        
        # Daten-Sparsity berechnen
        zero_counts = (self.df == 0).sum().sum()
        total_elements = self.df.size
        characteristics['data_sparsity'] = (zero_counts / total_elements) * 100
        
        # Sampling-Analyse für Klassen-Ungleichgewicht
        if target_col:
            target_counts = self.df[target_col].value_counts()
            minority_class = target_counts.min()
            majority_class = target_counts.max()
            imbalance_ratio = majority_class / minority_class
            minority_percentage = (minority_class / len(self.df)) * 100
            
            characteristics['sampling_analysis'] = {
                'minority_class_count': minority_class,
                'majority_class_count': majority_class,
                'imbalance_ratio': imbalance_ratio,
                'minority_percentage': minority_percentage,
                'recommended_sampling': self._recommend_sampling_strategy(imbalance_ratio, minority_percentage, len(self.df)),
                'sampling_parameters': self._get_sampling_parameters(imbalance_ratio, minority_percentage)
            }
        
        return characteristics
    
    def _recommend_sampling_strategy(self, imbalance_ratio: float, minority_percentage: float, total_samples: int) -> Dict[str, Any]:
        """Empfiehlt optimale Sampling-Strategie basierend auf Klassen-Ungleichgewicht"""
        
        if imbalance_ratio <= 2:
            return {
                'strategy': 'none',
                'reason': 'Ausgewogene Klassen - kein Sampling erforderlich',
                'priority': 'LOW'
            }
        elif imbalance_ratio <= 5:
            return {
                'strategy': 'class_weight',
                'reason': 'Leichtes Ungleichgewicht - class_weight ausreichend',
                'priority': 'MEDIUM'
            }
        elif imbalance_ratio <= 10:
            return {
                'strategy': 'smote',
                'reason': 'Mittleres Ungleichgewicht - SMOTE empfohlen',
                'priority': 'HIGH'
            }
        elif imbalance_ratio <= 20:
            return {
                'strategy': 'smote_enhanced',
                'reason': 'Starkes Ungleichgewicht - Erweiterte SMOTE-Techniken',
                'priority': 'HIGH'
            }
        else:
            return {
                'strategy': 'combination',
                'reason': 'Extremes Ungleichgewicht - Kombination aus SMOTE + class_weight',
                'priority': 'CRITICAL'
            }
    
    def _get_sampling_parameters(self, imbalance_ratio: float, minority_percentage: float) -> Dict[str, Any]:
        """Generiert optimale Sampling-Parameter"""
        
        if imbalance_ratio <= 2:
            return {
                'sampling_ratio': 1.0,
                'k_neighbors': 5,
                'random_state': 42
            }
        elif imbalance_ratio <= 5:
            return {
                'sampling_ratio': 1.5,
                'k_neighbors': 5,
                'random_state': 42
            }
        elif imbalance_ratio <= 10:
            return {
                'sampling_ratio': 2.0,
                'k_neighbors': 5,
                'random_state': 42
            }
        elif imbalance_ratio <= 20:
            return {
                'sampling_ratio': 3.0,
                'k_neighbors': 3,
                'random_state': 42
            }
        else:
            return {
                'sampling_ratio': 5.0,
                'k_neighbors': 3,
                'random_state': 42
            }
    
    def _generate_algorithm_recommendations(self, data_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert Algorithmus-Empfehlungen basierend auf Daten-Charakteristiken"""
        recommendations = []
        
        # Empfehlung 1: Sampling-Strategie basierend auf Klassen-Ungleichgewicht
        sampling_analysis = data_characteristics.get('sampling_analysis', {})
        if sampling_analysis:
            recommended_sampling = sampling_analysis.get('recommended_sampling', {})
            sampling_params = sampling_analysis.get('sampling_parameters', {})
            
            recommendations.append({
                'type': 'SAMPLING_STRATEGY',
                'priority': recommended_sampling.get('priority', 'MEDIUM'),
                'description': f"Sampling-Strategie: {recommended_sampling.get('strategy', 'none')}",
                'reason': recommended_sampling.get('reason', 'Keine Empfehlung'),
                'current_imbalance_ratio': sampling_analysis.get('imbalance_ratio', 1),
                'recommendation': f"Verwende {recommended_sampling.get('strategy', 'none')} mit Ratio {sampling_params.get('sampling_ratio', 1.0)}",
                'implementation': f"Sampling-Parameter: {sampling_params}",
                'sampling_config': {
                    'strategy': recommended_sampling.get('strategy', 'none'),
                    'parameters': sampling_params
                }
            })
        
        # Empfehlung 2: Klassen-Ungleichgewicht (Legacy)
        imbalance_ratio = data_characteristics['class_imbalance'].get('imbalance_ratio', 1)
        if imbalance_ratio > 10:
            recommendations.append({
                'type': 'CLASS_IMBALANCE_HANDLING',
                'priority': 'HIGH',
                'description': 'Extremes Klassen-Ungleichgewicht erkannt',
                'current_ratio': imbalance_ratio,
                'recommendation': 'SMOTE oder andere Oversampling-Techniken verwenden',
                'implementation': 'Oversampling-Faktor auf 3-5 setzen'
            })
        
        # Empfehlung 2: Feature-Korrelationen
        high_correlations = data_characteristics['feature_correlation'].get('high_correlations', [])
        if len(high_correlations) > 5:
            recommendations.append({
                'type': 'FEATURE_CORRELATION',
                'priority': 'MEDIUM',
                'description': f'{len(high_correlations)} hoch korrelierte Feature-Paare erkannt',
                'recommendation': 'Feature-Selektion oder PCA implementieren',
                'implementation': 'Korrelation-basierte Feature-Auswahl'
            })
        
        # Empfehlung 3: Daten-Sparsity
        sparsity = data_characteristics['data_sparsity']
        if sparsity > 80:
            recommendations.append({
                'type': 'DATA_SPARSITY',
                'priority': 'MEDIUM',
                'description': f'Hohe Daten-Sparsity: {sparsity:.1f}%',
                'recommendation': 'Sparsity-freundliche Algorithmen verwenden',
                'implementation': 'L1-Regularisierung oder Sparse-Algorithmen'
            })
        
        return recommendations
    
    def _detect_dominant_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Erkennt dominante Features mit Logistic Regression"""
        print("🔍 Prüfe auf dominante Features mit Logistic Regression...")
        
        dominant_features = []
        
        # Logistic Regression für Feature-Importance - optimiert für große Datasets
        lr = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', n_jobs=1)
        
        # Verwende nur eine Stichprobe für schnellere Berechnung
        if len(X) > 10000:
            sample_indices = np.random.choice(len(X), 10000, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            print(f"📊 Verwende Stichprobe von 10.000 Zeilen für dominante Feature-Erkennung")
        else:
            X_sample = X
            y_sample = y
        
        lr.fit(X_sample, y_sample)
        
        # Feature-Importance berechnen
        feature_importance = np.abs(lr.coef_[0])
        total_importance = np.sum(feature_importance)
        
        # Dominante Features identifizieren (mehr als 50% der Gesamt-Importance)
        for i, importance in enumerate(feature_importance):
            importance_ratio = importance / total_importance
            if importance_ratio > 0.5:
                dominant_features.append(feature_names[i])
                print(f"⚠️ Dominantes Feature erkannt: {feature_names[i]} (Importance: {importance_ratio:.3f})")
        
        if dominant_features:
            print(f"🚨 {len(dominant_features)} dominante Features gefunden und werden ausgeschlossen")
        else:
            print("✅ Keine dominanten Features gefunden")
        
        return dominant_features

    def _test_all_model_sampling_combinations(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Testet alle Model+Sampling-Kombinationen und findet die beste"""
        print("🤖 Teste alle Model+Sampling-Kombinationen...")
        
        # Alle Modelle definieren
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Gradient Boosting Modelle hinzufügen falls verfügbar
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=False)
        
        # Alle Sampling-Methoden definieren (nur die funktionierenden)
        sampling_methods = {
            'none': None,
            'borderline_smote': BorderlineSMOTE(random_state=42),
            'random_under': RandomUnderSampler(random_state=42),
            'tomek_links': TomekLinks(),
            'smoteenn': SMOTEENN(random_state=42),
            'smotetomek': SMOTETomek(random_state=42)
        }
        
        # Kombinationen für Multithreading vorbereiten
        combinations = []
        for model_name, model in models.items():
            for sampling_name, sampler in sampling_methods.items():
                combinations.append((model_name, model, sampling_name, sampler))
        
        print(f"🔄 Teste {len(combinations)} Kombinationen mit Multithreading...")
        
        # Thread-Lock für Print-Ausgaben
        print_lock = threading.Lock()
        results = {}
        
        def test_combination(combo):
            model_name, model, sampling_name, sampler = combo
            combo_name = f"{model_name}_{sampling_name}"
            
            try:
                # Sampling anwenden
                if sampler:
                    X_resampled, y_resampled = sampler.fit_resample(X, y)
                else:
                    X_resampled, y_resampled = X, y
                
                # Cross-Validation
                cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='roc_auc')
                
                # Metriken berechnen
                model.fit(X_resampled, y_resampled)
                # Robust gegen Ein-Klassen-Fälle
                try:
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        y_pred_proba = proba
                    elif proba.shape[1] == 1:
                        cls = int(getattr(model, 'classes_', [0])[0]) if hasattr(model, 'classes_') else 0
                        val = 1.0 if cls == 1 else 0.0
                        y_pred_proba = np.full((proba.shape[0],), val)
                    else:
                        classes = list(getattr(model, 'classes_', [])) if hasattr(model, 'classes_') else []
                        if 1 in classes:
                            idx = classes.index(1)
                            y_pred_proba = proba[:, idx]
                        else:
                            y_pred_proba = proba[:, -1]
                except Exception:
                    y_pred_proba = np.zeros((len(X),))
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                result = {
                    'model': model_name,
                    'sampling': sampling_name,
                    'auc_mean': cv_scores.mean(),
                    'auc_std': cv_scores.std(),
                    'recall': recall_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'f1': f1_score(y, y_pred)
                }
                
                with print_lock:
                    print(f"✅ {combo_name}: AUC={result['auc_mean']:.4f}")
                
                return combo_name, result
                
            except Exception as e:
                with print_lock:
                    print(f"⚠️ Fehler bei {combo_name}: {e}")
                return combo_name, {
                    'model': model_name,
                    'sampling': sampling_name,
                    'error': str(e)
                }
        
        # Multithreading ausführen mit Error-Handling
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_combo = {executor.submit(test_combination, combo): combo for combo in combinations}
            
            completed = 0
            total = len(combinations)
            failed_combinations = []
            
            for future in as_completed(future_to_combo):
                try:
                    combo_name, result = future.result()
                    results[combo_name] = result
                    completed += 1
                    
                    print(f"📊 Fortschritt: {completed}/{total} Kombinationen abgeschlossen ({completed/total*100:.1f}%)")
                    
                except Exception as e:
                    # Error-Handling für fehlgeschlagene Kombinationen
                    combo = future_to_combo[future]
                    model_name, model, sampling_name, sampler = combo
                    combo_name = f"{model_name}_{sampling_name}"
                    
                    print(f"❌ Fehler bei {combo_name}: {str(e)}")
                    print(f"   🔄 Überspringe diese Kombination und fahre fort...")
                    
                    # Fehlerhafte Kombination als fehlgeschlagen markieren
                    results[combo_name] = {
                        'model': model_name,
                        'sampling': sampling_name,
                        'error': str(e),
                        'status': 'failed'
                    }
                    failed_combinations.append(combo_name)
                    completed += 1
        
        # Beste Kombination finden
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_combo = max(valid_results.items(), key=lambda x: x[1]['auc_mean'])
            best_model_name, best_result = best_combo
            
            print(f"\n🏆 BESTE KOMBINATION GEFUNDEN:")
            print(f"   Modell: {best_result['model']}")
            print(f"   Sampling: {best_result['sampling']}")
            print(f"   AUC: {best_result['auc_mean']:.4f} (±{best_result['auc_std']:.4f})")
            print(f"   Recall: {best_result['recall']:.4f}")
            print(f"   Precision: {best_result['precision']:.4f}")
            print(f"   F1-Score: {best_result['f1']:.4f}")
            
            # Top 5 Kombinationen anzeigen
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['auc_mean'], reverse=True)
            print(f"\n📊 TOP 5 KOMBINATIONEN:")
            for i, (combo_name, combo_result) in enumerate(sorted_results[:5], 1):
                print(f"   {i}. {combo_result['model']} + {combo_result['sampling']}: AUC={combo_result['auc_mean']:.4f}")
        
        return {
            'all_results': results,
            'best_combination': best_combo if valid_results else None,
            'summary': {
                'total_combinations': len(combinations),
                'successful_combinations': len(valid_results),
                'failed_combinations': len(results) - len(valid_results)
            }
        }
    
    def _optimize_hyperparameters(self, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiert Hyperparameter basierend auf Daten-Charakteristiken"""
        recommendations = {
            'RandomForestClassifier': {},
            'general_recommendations': []
        }
        
        # RandomForest-spezifische Optimierungen
        rf_recs = recommendations['RandomForestClassifier']
        
        # n_estimators basierend auf Datenmenge
        sample_size = data_characteristics['sample_size']
        if sample_size < 10000:
            rf_recs['n_estimators'] = {'recommended': 200, 'reason': 'Kleinere Datenmenge'}
        elif sample_size < 100000:
            rf_recs['n_estimators'] = {'recommended': 300, 'reason': 'Mittlere Datenmenge'}
        else:
            rf_recs['n_estimators'] = {'recommended': 500, 'reason': 'Große Datenmenge'}
        
        # max_depth basierend auf Feature-Anzahl
        feature_count = data_characteristics['feature_count']
        if feature_count < 20:
            rf_recs['max_depth'] = {'recommended': 10, 'reason': 'Wenige Features'}
        elif feature_count < 50:
            rf_recs['max_depth'] = {'recommended': 15, 'reason': 'Mittlere Feature-Anzahl'}
        else:
            rf_recs['max_depth'] = {'recommended': 20, 'reason': 'Viele Features'}
        
        # min_samples_split basierend auf Klassen-Ungleichgewicht
        imbalance_ratio = data_characteristics['class_imbalance'].get('imbalance_ratio', 1)
        if imbalance_ratio > 5:
            rf_recs['min_samples_split'] = {'recommended': 5, 'reason': 'Hohes Klassen-Ungleichgewicht'}
        else:
            rf_recs['min_samples_split'] = {'recommended': 2, 'reason': 'Ausgewogene Klassen'}
        
        # Allgemeine Empfehlungen
        if imbalance_ratio > 10:
            recommendations['general_recommendations'].append({
                'parameter': 'class_weight',
                'value': 'balanced_subsample',
                'reason': 'Extremes Klassen-Ungleichgewicht'
            })
        
        return recommendations
    
    def _create_basic_data_dictionary(self) -> Dict[str, Any]:
        """Erstellt ein grundlegendes Data Dictionary basierend auf den CSV-Daten"""
        data_dict = {
            "columns": {},
            "feature_groups": {
                "primary_keys": [],
                "timebase": [],
                "target_features": [],
                "categorical_features": [],
                "static_features": [],
                "dynamic_features": []
            }
        }
        
        # Interaktive Abfrage für kritische Spalten
        print("\n🔍 Bitte definieren Sie die kritischen Spalten:")
        print(f"Verfügbare Spalten: {list(self.df.columns)}")
        
        # Primary Key Abfrage
        primary_key_input = input("Primary Key Spalte: ").strip()
        if primary_key_input and primary_key_input in self.df.columns:
            data_dict['feature_groups']['primary_keys'].append(primary_key_input)
        elif primary_key_input:
            # Versuche case-insensitive Match
            for col in self.df.columns:
                if col.lower() == primary_key_input.lower():
                    data_dict['feature_groups']['primary_keys'].append(col)
                    print(f"✅ Primary Key gefunden: {col}")
                    break
        
        # Timebase Abfrage
        timebase_input = input("Timebase Spalte: ").strip()
        if timebase_input and timebase_input in self.df.columns:
            data_dict['feature_groups']['timebase'].append(timebase_input)
        elif timebase_input:
            # Versuche case-insensitive Match
            for col in self.df.columns:
                if col.lower() == timebase_input.lower():
                    data_dict['feature_groups']['timebase'].append(col)
                    print(f"✅ Timebase gefunden: {col}")
                    break
        
        # Target Abfrage
        target_input = input("Target Spalte: ").strip()
        if target_input and target_input in self.df.columns:
            data_dict['feature_groups']['target_features'].append(target_input)
        elif target_input:
            # Versuche case-insensitive Match
            for col in self.df.columns:
                if col.lower() == target_input.lower():
                    data_dict['feature_groups']['target_features'].append(col)
                    print(f"✅ Target gefunden: {col}")
                    break
        
        # Analysiere jede Spalte und erstelle grundlegende Einträge
        for column in self.df.columns:
            dtype = str(self.df[column].dtype)
            unique_count = self.df[column].nunique()
            null_count = self.df[column].isnull().sum()
            
            # Berücksichtige interaktiv definierte Spalten
            if column in data_dict['feature_groups']['primary_keys']:
                role = 'PRIMARY_KEY'
                feature_group = 'primary_keys'
            elif column in data_dict['feature_groups']['timebase']:
                role = 'TIMEBASE'
                feature_group = 'timebase'
            elif column in data_dict['feature_groups']['target_features']:
                role = 'TARGET'
                feature_group = 'target_features'
            else:
                # Dynamische Feature-Typ-Erkennung für restliche Spalten
                if unique_count <= 10:  # Sehr wenige verschiedene Werte
                    role = 'CATEGORICAL_FEATURE'
                    feature_group = 'categorical_features'
                elif unique_count <= 50:  # Wenige verschiedene Werte = wahrscheinlich statisch
                    role = 'STATIC_FEATURE'
                    feature_group = 'static_features'
                else:  # Viele verschiedene Werte = wahrscheinlich dynamisch
                    role = 'DYNAMIC_FEATURE'
                    feature_group = 'dynamic_features'
            
            data_dict['columns'][column] = {
                'role': role,
                'dtype': dtype,
                'unique_count': unique_count,
                'null_count': null_count,
                'null_percentage': (null_count / len(self.df)) * 100,
                'feature_group': feature_group
            }
            
            # Füge zur entsprechenden Feature-Gruppe hinzu (nur wenn nicht bereits vorhanden)
            if column not in data_dict['feature_groups'][feature_group]:
                data_dict['feature_groups'][feature_group].append(column)
        
        return data_dict

    def _get_target_column(self) -> Optional[str]:
        """Ermittelt Target-Spalte"""
        for col, info in self.data_dictionary.get('columns', {}).items():
            if info.get('role') == 'TARGET':
                return col
        return None
    
    def generate_meta_configuration(self) -> Dict[str, Any]:
        """
        Generiert Meta-Konfiguration für optimierte Feature-Verarbeitung
        """
        print("\n⚙️ Generiere Meta-Konfiguration...")
        
        # Führe alle Analysen durch
        feature_analysis = self.analyze_feature_characteristics()
        algorithm_analysis = self.analyze_algorithm_performance()
        
        # Erstelle Meta-Konfiguration
        meta_config = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'engine_version': '1.0',
                'analysis_summary': {
                    'total_features': len(feature_analysis['feature_stats']),
                    'problems_detected': len(feature_analysis['problems_detected']),
                    'recommendations_generated': len(feature_analysis['recommendations'])
                }
            },
            'feature_optimization': {
                'problems_detected': feature_analysis['problems_detected'],
                'recommendations': feature_analysis['recommendations'],
                'feature_type_suggestions': feature_analysis['feature_type_suggestions'],
                'optimized_data_dictionary': self._generate_optimized_data_dictionary(feature_analysis)
            },
            'algorithm_optimization': {
                'current_config': algorithm_analysis['current_config'],
                'optimization_suggestions': algorithm_analysis['optimization_suggestions'],
                'hyperparameter_recommendations': algorithm_analysis['hyperparameter_recommendations'],
                'optimized_config': self._generate_optimized_algorithm_config(algorithm_analysis)
            },
            'implementation_plan': self._generate_implementation_plan(feature_analysis, algorithm_analysis)
        }
        
        return meta_config
    
    def _generate_optimized_data_dictionary(self, feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generiert optimiertes Data Dictionary basierend auf Analyse"""
        optimized_dict = self.data_dictionary.copy()
        
        # Aktualisiere Feature-Rollen basierend auf Vorschlägen
        for column, suggestion in feature_analysis['feature_type_suggestions'].items():
            if suggestion['suggested_role'] != suggestion['current_role']:
                if column in optimized_dict['columns']:
                    optimized_dict['columns'][column]['role'] = suggestion['suggested_role']
                    print(f"🔄 {column}: {suggestion['current_role']} → {suggestion['suggested_role']}")
        
        # Aktualisiere Feature-Gruppen
        self._update_feature_groups(optimized_dict)
        
        return optimized_dict
    
    def _update_feature_groups(self, data_dict: Dict[str, Any]):
        """Aktualisiert Feature-Gruppen basierend auf neuen Rollen"""
        # Behalte ursprüngliche Feature-Gruppen bei, falls vorhanden
        if 'feature_groups' not in data_dict:
            feature_groups = {
                'primary_keys': [],
                'timebase': [],
                'target': [],
                'static_features': [],
                'dynamic_features': [],
                'excluded_features': []
            }
        else:
            feature_groups = data_dict['feature_groups'].copy()
        
        # Leere alle Gruppen außer den kritischen (primary_keys, timebase, target)
        feature_groups['static_features'] = []
        feature_groups['dynamic_features'] = []
        feature_groups['excluded_features'] = []
        
        for column, info in data_dict['columns'].items():
            role = info.get('role', 'UNKNOWN')
            if role == 'PRIMARY_KEY':
                if column not in feature_groups['primary_keys']:
                    feature_groups['primary_keys'].append(column)
            elif role == 'TIMEBASE':
                if column not in feature_groups['timebase']:
                    feature_groups['timebase'].append(column)
            elif role == 'TARGET':
                if column not in feature_groups['target']:
                    feature_groups['target'].append(column)
            elif role == 'STATIC_FEATURE':
                feature_groups['static_features'].append(column)
            elif role == 'DYNAMIC_FEATURE':
                feature_groups['dynamic_features'].append(column)
            elif role == 'EXCLUDED_FEATURE':
                feature_groups['excluded_features'].append(column)
        
        data_dict['feature_groups'] = feature_groups
    
    def _generate_optimized_algorithm_config(self, algorithm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generiert optimierte Algorithmus-Konfiguration"""
        optimized_config = algorithm_analysis['current_config'].copy()
        
        # Wende Hyperparameter-Empfehlungen an
        rf_recs = algorithm_analysis['hyperparameter_recommendations'].get('RandomForestClassifier', {})
        for param, rec in rf_recs.items():
            if isinstance(rec, dict) and 'recommended' in rec:
                optimized_config[param] = rec['recommended']
        
        # Wende allgemeine Empfehlungen an
        general_recs = algorithm_analysis['hyperparameter_recommendations'].get('general_recommendations', [])
        for rec in general_recs:
            optimized_config[rec['parameter']] = rec['value']
        
        # Füge Sampling-Konfiguration hinzu
        sampling_config = self._extract_sampling_config(algorithm_analysis)
        if not sampling_config or not sampling_config.get('strategy'):
            sampling_config = {'strategy': 'none'}
        optimized_config['sampling_config'] = sampling_config
        
        return optimized_config
    
    def _extract_sampling_config(self, algorithm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extrahiert Sampling-Konfiguration aus Algorithmus-Analyse"""
        for recommendation in algorithm_analysis.get('optimization_suggestions', []):
            if recommendation.get('type') == 'SAMPLING_STRATEGY':
                return recommendation.get('sampling_config', {})
        return {}
    
    def _generate_implementation_plan(self, feature_analysis: Dict[str, Any], algorithm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generiert Implementierungsplan für die Optimierungen"""
        plan = {
            'priority_1': [],
            'priority_2': [],
            'priority_3': [],
            'estimated_impact': {},
            'implementation_steps': []
        }
        
        # Priorisiere Probleme
        for problem in feature_analysis['problems_detected']:
            if problem['severity'] == 'HIGH':
                plan['priority_1'].append(problem)
            elif problem['severity'] == 'MEDIUM':
                plan['priority_2'].append(problem)
            else:
                plan['priority_3'].append(problem)
        
        # Schätze Impact
        plan['estimated_impact'] = {
            'recall_improvement': '5-10%' if any(p['type'] == 'DIGITALIZATION_PROBLEM' for p in plan['priority_1']) else '2-5%',
            'precision_improvement': '3-7%',
            'auc_improvement': '0.02-0.05',
            'training_time_optimization': '10-20%'
        }
        
        # Implementierungsschritte
        plan['implementation_steps'] = [
            {
                'step': 1,
                'action': 'Data Dictionary aktualisieren',
                'description': 'Feature-Rollen basierend auf Analyse anpassen',
                'files_to_modify': ['config/data_dictionary.json']
            },
            {
                'step': 2,
                'action': 'Enhanced Early Warning System erweitern',
                'description': 'Feature-Engineering-Logik für optimierte Features anpassen',
                'files_to_modify': ['bl/enhanced_early_warning.py']
            },
            {
                'step': 3,
                'action': 'Algorithmus-Konfiguration optimieren',
                'description': 'Hyperparameter basierend auf Empfehlungen anpassen',
                'files_to_modify': ['bl/enhanced_early_warning.py']
            },
            {
                'step': 4,
                'action': 'Tests durchführen',
                'description': 'Performance-Vergleich zwischen altem und neuem System',
                'files_to_modify': ['test_performance_comparison.py']
            }
        ]
        
        return plan
    
    def _convert_numpy_types(self, obj):
        """Konvertiert NumPy-Datentypen zu Standard-Python-Datentypen für JSON-Serialisierung"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_meta_configuration(self, meta_config: Dict[str, Any], filename: str = None):
        """Speichert Meta-Konfiguration"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meta_config_{timestamp}.json"
        
        # Konvertiere NumPy-Datentypen zu Standard-Python-Datentypen
        meta_config_serializable = self._convert_numpy_types(meta_config)
        
        output_path = self.paths.config_directory() / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(meta_config_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Meta-Konfiguration gespeichert: {output_path}")
        return output_path
    
    def generate_optimized_files(self, meta_config: Dict[str, Any]):
        """Generiert optimierte Konfigurationsdateien"""
        print("\n📝 Generiere optimierte Dateien...")
        
        # 1. Optimiertes Data Dictionary
        optimized_dict = meta_config['feature_optimization']['optimized_data_dictionary']
        optimized_dict_serializable = self._convert_numpy_types(optimized_dict)
        dict_path = self.paths.config_directory() / "data_dictionary_optimized.json"
        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_dict_serializable, f, indent=2, ensure_ascii=False)
        print(f"✅ Optimiertes Data Dictionary: {dict_path}")
        
        # 2. Algorithmus-Konfiguration
        algo_config = meta_config['algorithm_optimization']['optimized_config']
        algo_config_serializable = self._convert_numpy_types(algo_config)
        algo_path = self.paths.config_directory() / "algorithm_config_optimized.json"
        with open(algo_path, 'w', encoding='utf-8') as f:
            json.dump(algo_config_serializable, f, indent=2, ensure_ascii=False)
        print(f"✅ Optimierte Algorithmus-Konfiguration: {algo_path}")
        
        # 3. Implementierungsanleitung
        implementation_guide = self._generate_implementation_guide(meta_config)
        guide_path = self.paths.config_directory() / "implementation_guide.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(implementation_guide)
        print(f"✅ Implementierungsanleitung: {guide_path}")
    
    def _generate_implementation_guide(self, meta_config: Dict[str, Any]) -> str:
        """Generiert Implementierungsanleitung"""
        guide = f"""# Feature Analysis Engine - Implementierungsanleitung

## 📊 Analyse-Zusammenfassung

**Generiert am:** {meta_config['metadata']['generated_at']}
**Engine Version:** {meta_config['metadata']['engine_version']}

### Erkannte Probleme: {meta_config['metadata']['analysis_summary']['problems_detected']}
### Generierte Empfehlungen: {meta_config['metadata']['analysis_summary']['recommendations_generated']}

## 🚨 Kritische Probleme (Priority 1)

"""
        
        for problem in meta_config['implementation_plan']['priority_1']:
            guide += f"""
### {problem['type']}
- **Beschreibung:** {problem['description']}
- **Impact:** {problem['impact']}
- **Lösung:** {problem['solution']}

"""
        
        guide += f"""
## 📈 Erwartete Verbesserungen

- **Recall:** {meta_config['implementation_plan']['estimated_impact']['recall_improvement']}
- **Precision:** {meta_config['implementation_plan']['estimated_impact']['precision_improvement']}
- **AUC:** {meta_config['implementation_plan']['estimated_impact']['auc_improvement']}
- **Training-Zeit:** {meta_config['implementation_plan']['estimated_impact']['training_time_optimization']}

## 🔧 Implementierungsschritte

"""
        
        for step in meta_config['implementation_plan']['implementation_steps']:
            guide += f"""
### Schritt {step['step']}: {step['action']}
**Beschreibung:** {step['description']}
**Dateien:** {', '.join(step['files_to_modify'])}

"""
        
        guide += """
## ✅ Nächste Schritte

1. **Review der Meta-Konfiguration** - Prüfe alle Empfehlungen
2. **Backup erstellen** - Sichere aktuelle Konfiguration
3. **Schrittweise Implementierung** - Führe Änderungen einzeln durch
4. **Performance-Tests** - Vergleiche alte vs. neue Konfiguration
5. **Dokumentation aktualisieren** - Aktualisiere Entwicklungsdokumentation

## 📞 Support

Bei Fragen zur Implementierung siehe `docs/UI_DEVELOPMENT_CONTEXT.md`
"""
        
        return guide
    
    def run_complete_analysis(self):
        """Führt vollständige Analyse durch und speichert Ergebnisse"""
        print("🚀 Feature Analysis Engine - Starte vollständige Analyse...")
        
        # Lade Daten
        self.load_data()
        
        if self.df is None:
            print("❌ Keine Daten verfügbar - Analyse abgebrochen")
            return None
        
        # Führe Analysen durch
        feature_analysis = self.analyze_feature_characteristics()
        algorithm_analysis = self.analyze_algorithm_performance()
        
        # Erstelle Meta-Konfiguration
        meta_config = self.generate_meta_configuration()
        
        # Speichere Ergebnisse
        self.save_meta_configuration(meta_config)
        self.generate_optimized_files(meta_config)
        
        # Zeige Zusammenfassung
        self._print_analysis_summary(meta_config)
        
        print("\n✅ Feature Analysis Engine abgeschlossen!")
        print(f"📁 Ergebnisse gespeichert in: {self.paths.config_directory()}")
        print(f"✅ Meta-Konfiguration gespeichert: {self.paths.config_directory() / 'meta_config_latest.json'}")
        
        return meta_config
    
    def _print_analysis_summary(self, meta_config: Dict[str, Any]):
        """Gibt Analyse-Zusammenfassung aus"""
        print("\n" + "="*80)
        print("📊 FEATURE ANALYSIS ENGINE - ZUSAMMENFASSUNG")
        print("="*80)
        
        # Kritische Probleme
        priority_1 = meta_config['implementation_plan']['priority_1']
        if priority_1:
            print(f"\n🚨 KRITISCHE PROBLEME ({len(priority_1)}):")
            for problem in priority_1:
                print(f"   • {problem['type']}: {problem['description']}")
        
        # Empfehlungen
        recommendations = meta_config['feature_optimization']['recommendations']
        if recommendations:
            print(f"\n💡 EMPFEHLUNGEN ({len(recommendations)}):")
            for rec in recommendations[:5]:  # Top 5
                print(f"   • {rec['type']}: {rec['description']}")
        
        # Erwartete Verbesserungen
        impact = meta_config['implementation_plan']['estimated_impact']
        print(f"\n📈 ERWARTETE VERBESSERUNGEN:")
        print(f"   • Recall: {impact['recall_improvement']}")
        print(f"   • Precision: {impact['precision_improvement']}")
        print(f"   • AUC: {impact['auc_improvement']}")
        
        print("\n" + "="*80)


def main():
    """Hauptfunktion für Feature Analysis Engine"""
    engine = FeatureAnalysisEngine()
    meta_config = engine.run_complete_analysis()
    
    # Speichere auch als Standard-Meta-Config
    engine.save_meta_configuration(meta_config, "meta_config_latest.json")
    
    return meta_config


if __name__ == "__main__":
    main() 