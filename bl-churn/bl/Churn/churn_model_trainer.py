#!/usr/bin/env python3
"""
Churn Model Trainer - Random Forest Training & Hyperparameter Tuning
====================================================================

Zentrale Model Training Pipeline für die Churn Random Forest Pipeline.
Migriert und konsolidiert bewährte Funktionalitäten aus:
- step1_backtest_engine.py: Model Training, Backtesting
- feature_analysis_engine.py: Hyperparameter Optimization, Multi-Algorithm Support
- enhanced_early_warning.py: Optimierte RandomForest Konfiguration

Features:
- Optimiertes Random Forest Training (AUC > 0.99 Basis)
- Hyperparameter Tuning (GridSearch, RandomSearch)
- Cross-Validation mit StratifiedKFold
- Alternative Algorithmen (XGBoost, LightGBM, CatBoost)
- Model Persistence mit Metadaten

Autor: AI Assistant
Datum: 2025-01-27
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, validation_curve
)
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Optional Advanced Models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Project imports
from config.paths_config import ProjectPaths
from bl.Churn.churn_constants import *

class ChurnModelTrainer:
    """
    Zentrale Model Training Pipeline für Churn Prediction
    """
    
    def __init__(self, experiment_id: Optional[int] = None):
        """
        Initialisiert Model Trainer
        
        Args:
            experiment_id: Optional Experiment-ID für Tracking
        """
        self.paths = ProjectPaths()
        self.logger = self._setup_logging()
        self.experiment_id = experiment_id
        
        # Model State
        self.trained_model = None
        self.model_metadata = {}
        self.training_results = {}
        self.feature_importance = {}
        
        # Performance Tracking
        self.cv_results = {}
        self.validation_scores = {}
        
    def _setup_logging(self):
        """Setup für strukturiertes Logging"""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=LOG_FORMAT
        )
        return logging.getLogger(__name__)
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, 
                           y_val: Optional[np.ndarray] = None,
                           params: Optional[Dict] = None) -> RandomForestClassifier:
        """
        Trainiert optimiertes Random Forest Model
        
        Args:
            X_train: Training Features
            y_train: Training Target
            X_val: Optional Validation Features
            y_val: Optional Validation Target
            params: Optional Custom Parameters
            
        Returns:
            Trainiertes RandomForestClassifier
        """
        self.logger.info("🌲 Starte Random Forest Training...")
        
        # Parameter-Konfiguration
        if params is None:
            params = DEFAULT_RANDOM_FOREST_PARAMS.copy()
        
        # Adaptive Parameter-Anpassung basierend auf Daten-Charakteristiken
        adaptive_params = self._adapt_parameters_to_data(X_train, y_train, params)
        
        self.logger.info(f"   📊 Training Set: {X_train.shape[0]} Samples, {X_train.shape[1]} Features")
        self.logger.info(f"   ⚙️ RF Parameter: {adaptive_params}")
        
        # Initialisiere und trainiere Model
        rf_model = RandomForestClassifier(**adaptive_params)
        
        # Training mit Performance Monitoring
        start_time = datetime.now()
        rf_model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Validation (falls verfügbar)
        validation_metrics = {}
        if X_val is not None and y_val is not None:
            validation_metrics = self._evaluate_model_performance(rf_model, X_val, y_val)
            self.logger.info(f"   📈 Validation AUC: {validation_metrics.get('auc', 'N/A'):.4f}")
        
        # Feature Importance extrahieren
        if hasattr(rf_model, 'feature_importances_'):
            self.feature_importance = {
                f'feature_{i}': float(importance) 
                for i, importance in enumerate(rf_model.feature_importances_)
            }
        
        # Training Results speichern
        self.training_results = {
            'model_type': 'RandomForestClassifier',
            'parameters': adaptive_params,
            'training_time_seconds': training_time,
            'training_samples': int(X_train.shape[0]),
            'training_features': int(X_train.shape[1]),
            'validation_metrics': validation_metrics,
            'feature_importance_available': hasattr(rf_model, 'feature_importances_'),
            'trained_at': datetime.now().isoformat()
        }
        
        self.trained_model = rf_model
        
        self.logger.info(f"✅ Random Forest Training abgeschlossen in {training_time:.2f}s")
        
        return rf_model
    
    def _adapt_parameters_to_data(self, X: np.ndarray, y: np.ndarray, 
                                 base_params: Dict) -> Dict:
        """
        Passt Model-Parameter adaptiv an Daten-Charakteristiken an
        
        Args:
            X: Feature Matrix
            y: Target Vector
            base_params: Basis-Parameter
            
        Returns:
            Adaptierte Parameter
        """
        adapted_params = base_params.copy()
        
        # Analyse der Daten-Charakteristiken
        sample_size = len(X)
        feature_count = X.shape[1]
        
        # Class Distribution Analysis
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_imbalance_ratio = max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1.0
        
        # Adaptive n_estimators basierend auf Datenmenge
        if sample_size < 5000:
            adapted_params['n_estimators'] = min(200, adapted_params.get('n_estimators', 300))
            self.logger.info("   🔧 Reduzierte n_estimators für kleine Datenmenge")
        elif sample_size > 50000:
            adapted_params['n_estimators'] = max(500, adapted_params.get('n_estimators', 300))
            self.logger.info("   🔧 Erhöhte n_estimators für große Datenmenge")
        
        # Adaptive max_depth basierend auf Feature-Anzahl
        if feature_count < 20:
            adapted_params['max_depth'] = min(10, adapted_params.get('max_depth', 12))
        elif feature_count > 100:
            adapted_params['max_depth'] = max(15, adapted_params.get('max_depth', 12))
        
        # Adaptive Class Weight für Imbalanced Data
        if class_imbalance_ratio > 5:
            adapted_params['class_weight'] = 'balanced_subsample'
            self.logger.info(f"   ⚖️ Class Weight angepasst für Imbalance Ratio: {class_imbalance_ratio:.2f}")
        
        # Adaptive min_samples_split für Overfitting-Kontrolle
        if sample_size < 10000 and feature_count > 50:
            adapted_params['min_samples_split'] = max(10, adapted_params.get('min_samples_split', 5))
            self.logger.info("   🔧 Erhöhte min_samples_split für Overfitting-Kontrolle")
        
        return adapted_params
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             method: str = 'grid', 
                             cv_folds: int = None) -> Dict[str, Any]:
        """
        Führt Hyperparameter Tuning durch
        
        Args:
            X: Feature Matrix
            y: Target Vector
            method: Tuning-Methode ('grid', 'random', 'adaptive')
            cv_folds: Anzahl CV-Folds
            
        Returns:
            Dictionary mit besten Parametern und Scores
        """
        if cv_folds is None:
            cv_folds = CV_FOLDS
        
        self.logger.info(f"🔍 Starte Hyperparameter Tuning: {method}")
        
        # Parameter Grid definieren
        if method == 'grid':
            return self._grid_search_tuning(X, y, cv_folds)
        elif method == 'random':
            return self._random_search_tuning(X, y, cv_folds)
        elif method == 'adaptive':
            return self._adaptive_tuning(X, y, cv_folds)
        else:
            self.logger.warning(f"⚠️ Unbekannte Tuning-Methode: {method} - verwende 'grid'")
            return self._grid_search_tuning(X, y, cv_folds)
    
    def _grid_search_tuning(self, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int) -> Dict[str, Any]:
        """Grid Search Hyperparameter Tuning"""
        
        # Kompakter Parameter Grid für Performance
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 12, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Reduziere Grid für große Datasets (Performance)
        if len(X) > 20000:
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [10, 12],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            }
            self.logger.info("   🔧 Reduzierter Parameter Grid für große Datenmenge")
        
        # GridSearchCV
        rf = RandomForestClassifier(
            random_state=CV_RANDOM_STATE,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Führe Grid Search durch
        start_time = datetime.now()
        grid_search.fit(X, y)
        tuning_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'method': 'grid_search',
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'tuning_time_seconds': tuning_time,
            'total_fits': len(grid_search.cv_results_['params']),
            'cv_folds': cv_folds
        }
        
        self.logger.info(f"✅ Grid Search abgeschlossen: AUC = {results['best_score']:.4f}")
        self.logger.info(f"   🏆 Beste Parameter: {results['best_params']}")
        
        return results
    
    def _random_search_tuning(self, X: np.ndarray, y: np.ndarray, 
                             cv_folds: int) -> Dict[str, Any]:
        """Random Search Hyperparameter Tuning"""
        
        # Parameter Distributions für Random Search
        param_distributions = {
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Anzahl Iterationen basierend auf Datenmenge
        n_iter = 20 if len(X) < 10000 else 10
        
        rf = RandomForestClassifier(
            random_state=CV_RANDOM_STATE,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        # Führe Random Search durch
        start_time = datetime.now()
        random_search.fit(X, y)
        tuning_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'method': 'random_search',
            'best_params': random_search.best_params_,
            'best_score': float(random_search.best_score_),
            'tuning_time_seconds': tuning_time,
            'total_fits': n_iter * cv_folds,
            'cv_folds': cv_folds
        }
        
        self.logger.info(f"✅ Random Search abgeschlossen: AUC = {results['best_score']:.4f}")
        
        return results
    
    def _adaptive_tuning(self, X: np.ndarray, y: np.ndarray, 
                        cv_folds: int) -> Dict[str, Any]:
        """Adaptive Tuning basierend auf Daten-Charakteristiken"""
        
        # Basis-Parameter aus Daten-Analyse
        base_params = self._adapt_parameters_to_data(X, y, DEFAULT_RANDOM_FOREST_PARAMS)
        
        # Fokussierte Parameter-Suche um adaptive Basis
        param_grid = {
            'n_estimators': [
                max(100, base_params['n_estimators'] - 100),
                base_params['n_estimators'],
                base_params['n_estimators'] + 100
            ],
            'max_depth': [
                max(5, base_params['max_depth'] - 3),
                base_params['max_depth'],
                base_params['max_depth'] + 3
            ],
            'min_samples_split': [2, base_params['min_samples_split'], base_params['min_samples_split'] * 2]
        }
        
        rf = RandomForestClassifier(
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
            class_weight=base_params['class_weight'],
            min_samples_leaf=base_params['min_samples_leaf']
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        start_time = datetime.now()
        grid_search.fit(X, y)
        tuning_time = (datetime.now() - start_time).total_seconds()
        
        # Kombiniere adaptive und tuned Parameter
        final_params = base_params.copy()
        final_params.update(grid_search.best_params_)
        
        results = {
            'method': 'adaptive_tuning',
            'best_params': final_params,
            'best_score': float(grid_search.best_score_),
            'adaptive_base_params': base_params,
            'tuning_time_seconds': tuning_time,
            'cv_folds': cv_folds
        }
        
        self.logger.info(f"✅ Adaptive Tuning abgeschlossen: AUC = {results['best_score']:.4f}")
        
        return results
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                            scoring: str = 'roc_auc', cv_folds: int = None) -> Dict[str, float]:
        """
        Führt Cross-Validation für Model durch
        
        Args:
            model: Zu validierendes Model
            X: Feature Matrix
            y: Target Vector
            scoring: Scoring-Metrik
            cv_folds: Anzahl CV-Folds
            
        Returns:
            CV-Results Dictionary
        """
        if cv_folds is None:
            cv_folds = CV_FOLDS
        
        self.logger.info(f"🔄 Führe {cv_folds}-Fold Cross-Validation durch...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
        
        # Multiple Scoring Metrics
        scoring_metrics = {
            'auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        cv_results = {}
        
        for metric_name, metric_scorer in scoring_metrics.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric_scorer, n_jobs=-1)
                cv_results[metric_name] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'scores': scores.tolist()
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Fehler bei CV-Metric {metric_name}: {e}")
                cv_results[metric_name] = {'mean': 0.0, 'std': 0.0}
        
        # Performance Summary
        primary_score = cv_results.get('auc', {}).get('mean', 0.0)
        self.logger.info(f"✅ Cross-Validation abgeschlossen: {scoring} = {primary_score:.4f} ± {cv_results.get('auc', {}).get('std', 0.0):.4f}")
        
        self.cv_results = cv_results
        
        return cv_results
    
    def _evaluate_model_performance(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluiert Model Performance auf Test-Set"""
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Handle single class predictions
        if y_pred_proba.shape[1] > 1:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba[:, 0]
            self.logger.warning("⚠️ Model hat nur eine Klasse vorhergesagt")
        
        # Calculate Metrics
        metrics = {}
        
        try:
            metrics['auc'] = float(roc_auc_score(y, y_scores))
        except Exception:
            metrics['auc'] = 0.0
        
        try:
            metrics['precision'] = float(precision_score(y, y_pred))
            metrics['recall'] = float(recall_score(y, y_pred))
            metrics['f1_score'] = float(f1_score(y, y_pred))
        except Exception as e:
            self.logger.warning(f"⚠️ Fehler bei Metric-Berechnung: {e}")
            metrics.update({'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0})
        
        # Confusion Matrix
        try:
            cm = confusion_matrix(y, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn)
                })
        except Exception:
            pass
        
        return metrics
    
    def train_alternative_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Trainiert alternative Models (XGBoost, LightGBM, CatBoost) für Vergleich
        
        Args:
            X: Feature Matrix
            y: Target Vector
            
        Returns:
            Dictionary mit trainierten Models und Performance
        """
        self.logger.info("🔀 Trainiere alternative Models für Vergleich...")
        
        models_results = {}
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = XGBClassifier(**XGBOOST_PARAMS)
                xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                models_results['XGBoost'] = {
                    'model': xgb_model,
                    'cv_auc_mean': float(xgb_scores.mean()),
                    'cv_auc_std': float(xgb_scores.std())
                }
                self.logger.info(f"   📊 XGBoost CV AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
            except Exception as e:
                self.logger.warning(f"⚠️ XGBoost Training fehlgeschlagen: {e}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = LGBMClassifier(**LIGHTGBM_PARAMS)
                lgb_scores = cross_val_score(lgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                models_results['LightGBM'] = {
                    'model': lgb_model,
                    'cv_auc_mean': float(lgb_scores.mean()),
                    'cv_auc_std': float(lgb_scores.std())
                }
                self.logger.info(f"   📊 LightGBM CV AUC: {lgb_scores.mean():.4f} ± {lgb_scores.std():.4f}")
            except Exception as e:
                self.logger.warning(f"⚠️ LightGBM Training fehlgeschlagen: {e}")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            try:
                cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
                cat_scores = cross_val_score(cat_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                models_results['CatBoost'] = {
                    'model': cat_model,
                    'cv_auc_mean': float(cat_scores.mean()),
                    'cv_auc_std': float(cat_scores.std())
                }
                self.logger.info(f"   📊 CatBoost CV AUC: {cat_scores.mean():.4f} ± {cat_scores.std():.4f}")
            except Exception as e:
                self.logger.warning(f"⚠️ CatBoost Training fehlgeschlagen: {e}")
        
        # Beste Model identifizieren
        if models_results:
            best_model_name = max(models_results.keys(), 
                                key=lambda k: models_results[k]['cv_auc_mean'])
            best_score = models_results[best_model_name]['cv_auc_mean']
            
            self.logger.info(f"🏆 Bestes alternatives Model: {best_model_name} (AUC: {best_score:.4f})")
            
            models_results['best_alternative'] = {
                'model_name': best_model_name,
                'score': best_score
            }
        
        return models_results
    
    def save_model(self, model, feature_names: List[str], 
                   model_metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Speichert trainiertes Model mit Metadaten
        
        Args:
            model: Trainiertes Model
            feature_names: Liste der Feature-Namen
            model_metadata: Optional zusätzliche Metadaten
            
        Returns:
            Pfad zur gespeicherten Model-Datei oder None bei Fehler
        """
        try:
            # Output Directory
            output_dir = self.paths.dynamic_system_outputs_directory() / DYNAMIC_OUTPUTS_SUBDIR
            output_dir.mkdir(exist_ok=True)
            
            # Filename mit Experiment-ID und Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.experiment_id:
                model_filename = f"churn_model_{self.experiment_id}_{timestamp}.joblib"
                metadata_filename = f"churn_model_{self.experiment_id}_{timestamp}_metadata.json"
            else:
                model_filename = f"churn_model_{timestamp}.joblib"
                metadata_filename = f"churn_model_{timestamp}_metadata.json"
            
            model_path = output_dir / model_filename
            metadata_path = output_dir / metadata_filename
            
            # Speichere Model
            joblib.dump(model, model_path, compress=MODEL_COMPRESSION_LEVEL)
            
            # Erstelle umfassende Metadaten
            metadata = {
                'model_info': {
                    'type': type(model).__name__,
                    'version': MODULE_VERSIONS.get('churn_model_trainer', '1.0.0'),
                    'sklearn_version': getattr(model, '_sklearn_version', 'unknown')
                },
                'training_info': self.training_results,
                'features': {
                    'feature_names': feature_names,
                    'feature_count': len(feature_names),
                    'feature_importance': self.feature_importance
                },
                'performance': {
                    'cv_results': self.cv_results,
                    'validation_scores': self.validation_scores
                },
                'experiment': {
                    'experiment_id': self.experiment_id,
                    'created_at': datetime.now().isoformat(),
                    'model_file': model_filename
                }
            }
            
            # Merge mit zusätzlichen Metadaten
            if model_metadata:
                metadata.update(model_metadata)
            
            # Speichere Metadaten
            with open(metadata_path, 'w', encoding=DEFAULT_ENCODING) as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Model gespeichert: {model_path}")
            self.logger.info(f"✅ Metadaten gespeichert: {metadata_path}")
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Speichern des Models: {e}")
            return None
    
    def load_model(self, model_path: str) -> Optional[Tuple[Any, Dict]]:
        """
        Lädt gespeichertes Model mit Metadaten
        
        Args:
            model_path: Pfad zur Model-Datei
            
        Returns:
            Tuple von (model, metadata) oder None bei Fehler
        """
        try:
            # Lade Model
            model = joblib.load(model_path)
            
            # Versuche Metadaten zu laden
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            metadata = {}
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding=DEFAULT_ENCODING) as f:
                    metadata = json.load(f)
            
            self.logger.info(f"✅ Model geladen: {model_path}")
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden des Models: {e}")
            return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Erstellt umfassende Zusammenfassung des Trainings"""
        
        summary = {
            'experiment_id': self.experiment_id,
            'model_trained': self.trained_model is not None,
            'training_results': self.training_results,
            'cv_results': self.cv_results,
            'feature_importance_available': len(self.feature_importance) > 0,
            'performance_targets': {
                'auc_target': TARGET_METRICS['auc'],
                'auc_achieved': self.cv_results.get('auc', {}).get('mean', 0.0),
                'target_met': self.cv_results.get('auc', {}).get('mean', 0.0) >= TARGET_METRICS['auc']
            }
        }
        
        return summary


if __name__ == "__main__":
    # Test der ChurnModelTrainer Funktionalität
    from bl.Churn.churn_data_loader import ChurnDataLoader
    from bl.Churn.churn_feature_engine import ChurnFeatureEngine
    
    print("🧪 Teste ChurnModelTrainer...")
    
    # 1. Lade und verarbeite Test-Daten
    data_loader = ChurnDataLoader()
    # Lade Daten strikt über DAL (keine direkten CSV-Zugriffe)
    df = data_loader.load_stage0_data()
    
    if df is None:
        print("❌ Konnte Test-Daten nicht laden")
        exit(1)
    
    # 2. Feature Engineering
    data_dict = getattr(data_loader, "data_dictionary", None)
    feature_engine = ChurnFeatureEngine(data_dictionary=data_dict)
    
    # Teste mit kleinem Subset für Performance
    test_df = df.head(2000).copy()
    df_with_features = feature_engine.create_rolling_features(test_df)
    df_with_enhanced = feature_engine.create_enhanced_features(df_with_features)
    
    # 3. Training Data Preparation
    X, y, feature_names = feature_engine.prepare_training_data(
        df_with_enhanced, 
        prediction_timebase="202401"
    )
    print(f"   Training Data: {X.shape[0]} Samples, {X.shape[1]} Features")
    
    # 4. Initialisiere Model Trainer
    trainer = ChurnModelTrainer(experiment_id=999)
    
    # 5. Train-Validation Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train, X_val, y_val)
    print(f"✅ Random Forest trainiert")
    
    # 7. Cross-Validation
    cv_results = trainer.cross_validate_model(rf_model, X_train, y_train)
    print(f"✅ Cross-Validation: AUC = {cv_results.get('auc', {}).get('mean', 0.0):.4f}")
    
    # 8. Hyperparameter Tuning (Quick Test)
    try:
        tuning_results = trainer.hyperparameter_tuning(X_train, y_train, method='grid')
        print(f"✅ Hyperparameter Tuning: Best AUC = {tuning_results.get('best_score', 0.0):.4f}")
    except Exception as e:
        print(f"⚠️ Hyperparameter Tuning übersprungen: {e}")
    
    # 9. Model Speichern
    model_path = trainer.save_model(rf_model, feature_names)
    if model_path:
        print(f"✅ Model gespeichert: {model_path}")
    
    # 10. Training Summary
    summary = trainer.get_training_summary()
    print(f"✅ Training Summary: Target AUC {'erreicht' if summary['performance_targets']['target_met'] else 'nicht erreicht'}")
    
    print("\n🎯 ChurnModelTrainer Test erfolgreich abgeschlossen!")
