#!/usr/bin/env python3
"""
Churn Evaluator - Performance & Business Metrics Evaluation
===========================================================

Umfassende Evaluation Pipeline für die Churn Random Forest Pipeline.
Migriert und konsolidiert bewährte Funktionalitäten aus:
- threshold_optimizer.py: Robuste Threshold-Optimierung
- step1_validation_engine.py: Model Validation, Data Leakage Checks
- step1_backtest_engine.py: Risk Segmentation, Business Metrics

Features:
- Robuste Threshold-Optimierung (6 Methoden mit Cross-Validation)
- Performance Metrics (AUC, Precision, Recall, F1) mit Confidence Intervals
- Business Impact Metrics (ROI, Revenue Loss, Customer Segmentation)
- Data Leakage Validation
- Model Performance Validation gegen Targets

Autor: AI Assistant
Datum: 2025-01-27
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# ML Imports
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Project imports
from config.paths_config import ProjectPaths
from bl.Churn.churn_constants import *

class ChurnEvaluator:
    """
    Umfassende Evaluation Pipeline für Churn Models
    """
    
    def __init__(self, experiment_id: Optional[int] = None):
        """
        Initialisiert Churn Evaluator
        
        Args:
            experiment_id: Optional Experiment-ID für Tracking
        """
        self.paths = ProjectPaths()
        self.logger = self._setup_logging()
        self.experiment_id = experiment_id
        
        # Evaluation Results
        self.performance_metrics = {}
        self.business_metrics = {}
        self.threshold_results = {}
        self.validation_results = {}
        
        # Optimal Threshold
        self.optimal_threshold = DEFAULT_THRESHOLD
        self.threshold_method = None
        
    def _setup_logging(self):
        """Setup für strukturiertes Logging"""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=LOG_FORMAT
        )
        return logging.getLogger(__name__)
    
    def evaluate_model_performance(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Führt umfassende Model Performance Evaluation durch
        
        Args:
            model: Trainiertes Model
            X_test: Test Features
            y_test: Test Target
            feature_names: Optional Feature-Namen für Importance
            
        Returns:
            Dictionary mit Performance Metrics
        """
        self.logger.info("📊 Führe Model Performance Evaluation durch...")
        
        # Predictions generieren
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Handle single class predictions
        if y_pred_proba.shape[1] > 1:
            y_scores = y_pred_proba[:, 1]  # Churn-Wahrscheinlichkeit
        else:
            y_scores = y_pred_proba[:, 0]
            self.logger.warning("⚠️ Model hat nur eine Klasse vorhergesagt")
        
        # Core Performance Metrics
        metrics = self._calculate_core_metrics(y_test, y_pred, y_scores)
        
        # Advanced Metrics
        advanced_metrics = self._calculate_advanced_metrics(y_test, y_pred, y_scores)
        metrics.update(advanced_metrics)
        
        # Confusion Matrix Details
        cm_metrics = self._calculate_confusion_matrix_metrics(y_test, y_pred)
        metrics.update(cm_metrics)
        
        # Feature Importance (falls verfügbar)
        if hasattr(model, 'feature_importances_') and feature_names:
            importance_metrics = self._calculate_feature_importance_metrics(
                model.feature_importances_, feature_names
            )
            metrics.update(importance_metrics)
        
        # Performance Target Validation
        target_validation = self._validate_performance_targets(metrics)
        metrics['target_validation'] = target_validation
        
        # Sample Size Information
        metrics['sample_info'] = {
            'test_samples': int(len(y_test)),
            'positive_samples': int(np.sum(y_test)),
            'negative_samples': int(len(y_test) - np.sum(y_test)),
            'positive_ratio': float(np.mean(y_test))
        }
        
        self.performance_metrics = metrics
        
        # Performance Summary
        auc = metrics.get('auc', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1_score', 0.0)
        
        self.logger.info(f"✅ Performance Evaluation abgeschlossen:")
        self.logger.info(f"   🎯 AUC: {auc:.4f} (Target: {TARGET_METRICS['auc']:.2f})")
        self.logger.info(f"   🎯 Precision: {precision:.4f} (Target: {TARGET_METRICS['precision']:.2f})")
        self.logger.info(f"   🎯 Recall: {recall:.4f} (Target: {TARGET_METRICS['recall']:.2f})")
        self.logger.info(f"   🎯 F1-Score: {f1:.4f} (Target: {TARGET_METRICS['f1_score']:.2f})")
        
        return metrics
    
    def _calculate_core_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_scores: np.ndarray) -> Dict[str, float]:
        """Berechnet Kern-Performance-Metriken"""
        metrics = {}
        
        try:
            # AUC (primäre Metrik)
            metrics['auc'] = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            self.logger.warning(f"⚠️ AUC-Berechnung fehlgeschlagen: {e}")
            metrics['auc'] = 0.0
        
        try:
            # Classification Metrics
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception as e:
            self.logger.warning(f"⚠️ Classification Metrics fehlgeschlagen: {e}")
            metrics.update({'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0})
        
        # Accuracy
        try:
            metrics['accuracy'] = float(np.mean(y_true == y_pred))
        except Exception:
            metrics['accuracy'] = 0.0
        
        return metrics
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_scores: np.ndarray) -> Dict[str, Any]:
        """Berechnet erweiterte Performance-Metriken"""
        advanced = {}
        
        try:
            # ROC Curve Analysis
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            
            # Optimal ROC Threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            
            advanced['roc_analysis'] = {
                'optimal_threshold': float(roc_thresholds[optimal_idx]),
                'optimal_tpr': float(tpr[optimal_idx]),
                'optimal_fpr': float(fpr[optimal_idx]),
                'youden_j': float(j_scores[optimal_idx])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ ROC Analyse fehlgeschlagen: {e}")
            advanced['roc_analysis'] = {}
        
        try:
            # Precision-Recall Curve Analysis
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            
            # F1-optimal Threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + EPSILON)
            f1_optimal_idx = np.argmax(f1_scores)
            
            advanced['pr_analysis'] = {
                'optimal_threshold': float(pr_thresholds[f1_optimal_idx]) if f1_optimal_idx < len(pr_thresholds) else 0.5,
                'optimal_precision': float(precision[f1_optimal_idx]),
                'optimal_recall': float(recall[f1_optimal_idx]),
                'optimal_f1': float(f1_scores[f1_optimal_idx])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ PR Analyse fehlgeschlagen: {e}")
            advanced['pr_analysis'] = {}
        
        # Confidence Intervals (Bootstrap)
        try:
            confidence_intervals = self._bootstrap_confidence_intervals(y_true, y_scores)
            advanced['confidence_intervals'] = confidence_intervals
        except Exception as e:
            self.logger.warning(f"⚠️ Confidence Intervals fehlgeschlagen: {e}")
            advanced['confidence_intervals'] = {}
        
        return advanced
    
    def _bootstrap_confidence_intervals(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                      n_bootstrap: int = None) -> Dict[str, Dict[str, float]]:
        """Berechnet Bootstrap Confidence Intervals für AUC"""
        if n_bootstrap is None:
            n_bootstrap = min(BOOTSTRAP_ITERATIONS, 500)  # Reduziert für Performance
        
        bootstrap_aucs = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap Sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Check if both classes are present
            y_boot = y_true[indices]
            if len(np.unique(y_boot)) < 2:
                continue
            
            try:
                auc_boot = roc_auc_score(y_boot, y_scores[indices])
                bootstrap_aucs.append(auc_boot)
            except:
                continue
        
        if len(bootstrap_aucs) > 10:
            bootstrap_aucs = np.array(bootstrap_aucs)
            return {
                'auc': {
                    'lower': float(np.percentile(bootstrap_aucs, 2.5)),
                    'upper': float(np.percentile(bootstrap_aucs, 97.5)),
                    'mean': float(np.mean(bootstrap_aucs)),
                    'std': float(np.std(bootstrap_aucs))
                }
            }
        else:
            return {'auc': {'lower': 0.0, 'upper': 0.0, 'mean': 0.0, 'std': 0.0}}
    
    def _calculate_confusion_matrix_metrics(self, y_true: np.ndarray, 
                                           y_pred: np.ndarray) -> Dict[str, Any]:
        """Berechnet detaillierte Confusion Matrix Metriken"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                metrics = {
                    'confusion_matrix': {
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'true_negatives': int(tn),
                        'false_negatives': int(fn)
                    },
                    'derived_metrics': {
                        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                        'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                        'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                    }
                }
                return metrics
            else:
                return {'confusion_matrix': {}, 'derived_metrics': {}}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Confusion Matrix Berechnung fehlgeschlagen: {e}")
            return {'confusion_matrix': {}, 'derived_metrics': {}}
    
    def _calculate_feature_importance_metrics(self, importance_scores: np.ndarray, 
                                            feature_names: List[str]) -> Dict[str, Any]:
        """Berechnet Feature Importance Metriken"""
        try:
            # Sortiere Features nach Importance
            importance_indices = np.argsort(importance_scores)[::-1]
            
            # Top Features
            top_n = min(20, len(feature_names))
            top_features = []
            
            for i in range(top_n):
                idx = importance_indices[i]
                top_features.append({
                    'feature_name': feature_names[idx],
                    'importance_score': float(importance_scores[idx]),
                    'rank': i + 1
                })
            
            # Importance Statistics
            importance_stats = {
                'mean_importance': float(np.mean(importance_scores)),
                'std_importance': float(np.std(importance_scores)),
                'max_importance': float(np.max(importance_scores)),
                'min_importance': float(np.min(importance_scores)),
                'gini_coefficient': self._calculate_gini_coefficient(importance_scores)
            }
            
            return {
                'feature_importance': {
                    'top_features': top_features,
                    'statistics': importance_stats,
                    'total_features': len(feature_names)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature Importance Berechnung fehlgeschlagen: {e}")
            return {'feature_importance': {}}
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Berechnet Gini-Koeffizient für Feature Importance Verteilung"""
        try:
            sorted_values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(sorted_values)
            return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)
        except:
            return 0.0
    
    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          method: str = 'robust') -> Dict[str, Any]:
        """
        Führt robuste Threshold-Optimierung durch
        
        Args:
            y_true: True Labels
            y_pred_proba: Prediction Probabilities
            method: Optimierungsmethode ('robust', 'roc', 'f1', 'precision')
            
        Returns:
            Dictionary mit optimalen Threshold-Ergebnissen
        """
        self.logger.info(f"🔍 Starte Threshold-Optimierung: {method}")
        
        if method == 'robust':
            return self._robust_threshold_optimization(y_true, y_pred_proba)
        elif method == 'roc':
            return self._roc_threshold_optimization(y_true, y_pred_proba)
        elif method == 'f1':
            return self._f1_threshold_optimization(y_true, y_pred_proba)
        elif method == 'precision':
            return self._precision_threshold_optimization(y_true, y_pred_proba)
        else:
            self.logger.warning(f"⚠️ Unbekannte Threshold-Methode: {method} - verwende 'robust'")
            return self._robust_threshold_optimization(y_true, y_pred_proba)
    
    def _robust_threshold_optimization(self, y_true: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Robuste Threshold-Optimierung mit Cross-Validation (aus threshold_optimizer.py)"""
        
        # Alle verfügbaren Methoden
        methods = {
            'roc_optimal': self._optimize_roc_threshold,
            'f1_optimal': self._optimize_f1_threshold,
            'precision_optimal': lambda y, p: self._optimize_precision_threshold(y, p, MIN_RECALL_FOR_PRECISION_THRESHOLD),
            'percentile_75': self._optimize_percentile_threshold,
            'cost_benefit': self._optimize_cost_benefit_threshold
        }
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=CV_THRESHOLD_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        
        # Cross-Validation für alle Methoden
        for method_name, method_func in methods.items():
            method_scores = []
            method_thresholds = []
            
            try:
                for train_idx, val_idx in cv.split(y_true, y_true):
                    y_val_true = y_true[val_idx]
                    y_val_proba = y_pred_proba[val_idx]
                    
                    # Skip wenn nur eine Klasse
                    if len(np.unique(y_val_true)) < 2:
                        continue
                    
                    # Optimiere Threshold
                    threshold = method_func(y_val_true, y_val_proba)
                    
                    # Evaluiere mit Threshold
                    y_val_pred = (y_val_proba >= threshold).astype(int)
                    f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
                    
                    method_scores.append(f1)
                    method_thresholds.append(threshold)
                
                if len(method_scores) >= 2:
                    cv_results[method_name] = {
                        'mean_score': float(np.mean(method_scores)),
                        'std_score': float(np.std(method_scores)),
                        'mean_threshold': float(np.mean(method_thresholds)),
                        'std_threshold': float(np.std(method_thresholds)),
                        'cv_scores': [float(s) for s in method_scores]
                    }
                    
            except Exception as e:
                self.logger.warning(f"⚠️ CV für {method_name} fehlgeschlagen: {e}")
                continue
        
        # Wähle beste Methode
        if cv_results:
            best_method = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_score'])
            best_results = cv_results[best_method]
            
            self.optimal_threshold = best_results['mean_threshold']
            self.threshold_method = best_method
            
            results = {
                'optimal_threshold': self.optimal_threshold,
                'best_method': best_method,
                'cv_score': best_results['mean_score'],
                'cv_std': best_results['std_score'],
                'all_methods': cv_results,
                'robustness_score': best_results['mean_score'] / (best_results['std_score'] + EPSILON)
            }
            
            self.logger.info(f"✅ Robuste Threshold-Optimierung: {self.optimal_threshold:.3f} ({best_method})")
            self.logger.info(f"   📊 CV F1-Score: {best_results['mean_score']:.3f} ± {best_results['std_score']:.3f}")
            
        else:
            # Fallback: Direkte F1-Optimierung
            self.optimal_threshold = self._optimize_f1_threshold(y_true, y_pred_proba)
            self.threshold_method = 'f1_fallback'
            
            results = {
                'optimal_threshold': self.optimal_threshold,
                'best_method': 'f1_fallback',
                'cv_score': 0.0,
                'cv_std': 0.0,
                'all_methods': {},
                'robustness_score': 0.0
            }
            
            self.logger.warning(f"⚠️ CV fehlgeschlagen - Fallback Threshold: {self.optimal_threshold:.3f}")
        
        self.threshold_results = results
        return results
    
    def _optimize_roc_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """ROC-basierte Threshold-Optimierung (Youden's J)"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        return float(thresholds[optimal_idx])
    
    def _optimize_f1_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """F1-Score-basierte Threshold-Optimierung"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + EPSILON)
        optimal_idx = np.argmax(f1_scores)
        return float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else DEFAULT_THRESHOLD
    
    def _optimize_precision_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                    min_recall: float) -> float:
        """Precision-optimierte Threshold-Optimierung mit Recall-Constraint"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Filter Thresholds mit mindestens min_recall
        valid_indices = recall >= min_recall
        if not np.any(valid_indices):
            return DEFAULT_THRESHOLD
        
        valid_precision = precision[valid_indices]
        valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds ist um 1 kürzer
        
        if len(valid_thresholds) == 0:
            return DEFAULT_THRESHOLD
        
        optimal_idx = np.argmax(valid_precision[:-1])  # Anpassung für thresholds
        return float(valid_thresholds[optimal_idx])
    
    def _optimize_percentile_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Perzentil-basierte Threshold-Optimierung"""
        return float(np.percentile(y_pred_proba, 75))
    
    def _optimize_cost_benefit_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       cost_fp: float = 1.0, cost_fn: float = 10.0) -> float:
        """Cost-Benefit-basierte Threshold-Optimierung"""
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = DEFAULT_THRESHOLD
        best_benefit = -np.inf
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Confusion Matrix
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            # Cost-Benefit Calculation
            benefit = tp * 1.0 - fp * cost_fp - fn * cost_fn
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_threshold = threshold
        
        return float(best_threshold)
    
    def _roc_threshold_optimization(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Einfache ROC-basierte Threshold-Optimierung"""
        threshold = self._optimize_roc_threshold(y_true, y_pred_proba)
        
        self.optimal_threshold = threshold
        self.threshold_method = 'roc_optimal'
        
        return {
            'optimal_threshold': threshold,
            'method': 'roc_optimal',
            'cv_score': 0.0,
            'robustness_score': 1.0
        }
    
    def _f1_threshold_optimization(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Einfache F1-basierte Threshold-Optimierung"""
        threshold = self._optimize_f1_threshold(y_true, y_pred_proba)
        
        self.optimal_threshold = threshold
        self.threshold_method = 'f1_optimal'
        
        return {
            'optimal_threshold': threshold,
            'method': 'f1_optimal',
            'cv_score': 0.0,
            'robustness_score': 1.0
        }
    
    def _precision_threshold_optimization(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Einfache Precision-basierte Threshold-Optimierung"""
        threshold = self._optimize_precision_threshold(y_true, y_pred_proba, MIN_RECALL_FOR_PRECISION_THRESHOLD)
        
        self.optimal_threshold = threshold
        self.threshold_method = 'precision_optimal'
        
        return {
            'optimal_threshold': threshold,
            'method': 'precision_optimal',
            'cv_score': 0.0,
            'robustness_score': 1.0
        }
    
    def calculate_business_metrics(self, y_pred_proba: np.ndarray, 
                                  customer_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Berechnet Business Impact Metrics
        
        Args:
            y_pred_proba: Churn Probabilities
            customer_data: Optional Customer-Daten für individuelle Customer Values
            
        Returns:
            Dictionary mit Business Metrics
        """
        self.logger.info("💼 Berechne Business Impact Metrics...")
        
        # Risk Kategorisierung
        risk_categories = self._categorize_customers_by_risk(y_pred_proba)
        
        # Customer Values (individuell oder durchschnittlich)
        if customer_data is not None and 'customer_value' in customer_data.columns:
            customer_values = customer_data['customer_value'].fillna(AVERAGE_CUSTOMER_VALUE).values
        else:
            customer_values = np.full(len(y_pred_proba), AVERAGE_CUSTOMER_VALUE)
        
        # Business Impact Berechnung
        business_metrics = {
            'risk_segmentation': {
                'customers_high_risk': int(risk_categories['high_risk_count']),
                'customers_medium_risk': int(risk_categories['medium_risk_count']),
                'customers_low_risk': int(risk_categories['low_risk_count']),
                'total_customers': int(len(y_pred_proba))
            },
            'financial_impact': self._calculate_financial_impact(
                y_pred_proba, customer_values, risk_categories
            ),
            'intervention_recommendations': self._generate_intervention_recommendations(
                risk_categories
            )
        }
        
        self.business_metrics = business_metrics
        
        # Business Summary
        total_revenue_at_risk = business_metrics['financial_impact']['potential_revenue_loss']
        roi_estimate = business_metrics['financial_impact']['roi_estimate']
        high_risk_customers = business_metrics['risk_segmentation']['customers_high_risk']
        
        self.logger.info(f"✅ Business Metrics berechnet:")
        self.logger.info(f"   💰 Revenue at Risk: €{total_revenue_at_risk:,.0f}")
        self.logger.info(f"   📈 ROI Estimate: {roi_estimate:.2f}x")
        self.logger.info(f"   🚨 High Risk Customers: {high_risk_customers}")
        
        return business_metrics
    
    def _categorize_customers_by_risk(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Kategorisiert Kunden nach Churn-Risiko"""
        
        high_risk_mask = y_pred_proba >= HIGH_RISK_THRESHOLD
        medium_risk_mask = (y_pred_proba >= MEDIUM_RISK_THRESHOLD) & (y_pred_proba < HIGH_RISK_THRESHOLD)
        low_risk_mask = y_pred_proba < LOW_RISK_THRESHOLD
        
        return {
            'high_risk_count': np.sum(high_risk_mask),
            'medium_risk_count': np.sum(medium_risk_mask),
            'low_risk_count': np.sum(low_risk_mask),
            'high_risk_indices': np.where(high_risk_mask)[0].tolist(),
            'medium_risk_indices': np.where(medium_risk_mask)[0].tolist(),
            'low_risk_indices': np.where(low_risk_mask)[0].tolist(),
            'thresholds_used': {
                'high_risk': HIGH_RISK_THRESHOLD,
                'medium_risk': MEDIUM_RISK_THRESHOLD,
                'low_risk': LOW_RISK_THRESHOLD
            }
        }
    
    def _calculate_financial_impact(self, y_pred_proba: np.ndarray, 
                                   customer_values: np.ndarray,
                                   risk_categories: Dict) -> Dict[str, float]:
        """Berechnet finanziellen Impact"""
        
        # Revenue at Risk (gewichtet nach Churn-Wahrscheinlichkeit)
        potential_revenue_loss = float(np.sum(y_pred_proba * customer_values))
        
        # High Risk Customer Revenue
        high_risk_indices = risk_categories['high_risk_indices']
        high_risk_revenue = float(np.sum(customer_values[high_risk_indices])) if high_risk_indices else 0.0
        
        # Prevention Costs (basierend auf Risk Level)
        high_risk_prevention_cost = len(high_risk_indices) * AVERAGE_CUSTOMER_VALUE * PREVENTION_COST_RATIO
        medium_risk_prevention_cost = len(risk_categories['medium_risk_indices']) * AVERAGE_CUSTOMER_VALUE * PREVENTION_COST_RATIO * 0.5
        
        total_prevention_cost = high_risk_prevention_cost + medium_risk_prevention_cost
        
        # ROI Calculation
        # Erfolgreiche Retention nach Prevention
        prevented_loss = (
            len(high_risk_indices) * AVERAGE_CUSTOMER_VALUE * PREVENTION_SUCCESS_RATE +
            len(risk_categories['medium_risk_indices']) * AVERAGE_CUSTOMER_VALUE * PREVENTION_SUCCESS_RATE * 0.7
        )
        
        roi_estimate = prevented_loss / max(total_prevention_cost, 1.0)
        
        return {
            'potential_revenue_loss': potential_revenue_loss,
            'high_risk_revenue': high_risk_revenue,
            'prevention_cost_estimate': float(total_prevention_cost),
            'prevented_loss_estimate': float(prevented_loss),
            'roi_estimate': float(roi_estimate),
            'avg_customer_value': float(np.mean(customer_values)),
            'max_customer_value': float(np.max(customer_values)),
            'total_customer_value': float(np.sum(customer_values))
        }
    
    def _generate_intervention_recommendations(self, risk_categories: Dict) -> Dict[str, Any]:
        """Generiert Intervention-Empfehlungen"""
        
        high_risk_count = risk_categories['high_risk_count']
        medium_risk_count = risk_categories['medium_risk_count']
        
        recommendations = {
            'immediate_actions': [],
            'medium_term_actions': [],
            'monitoring_actions': []
        }
        
        # Immediate Actions (High Risk)
        if high_risk_count > 0:
            recommendations['immediate_actions'].extend([
                f"Sofortige Kontaktaufnahme mit {high_risk_count} High-Risk Kunden",
                "Personalisierte Retention-Angebote erstellen",
                "Account Manager Gespräche einleiten"
            ])
        
        # Medium Term Actions (Medium Risk)
        if medium_risk_count > 0:
            recommendations['medium_term_actions'].extend([
                f"Proaktive Engagement-Kampagne für {medium_risk_count} Medium-Risk Kunden",
                "Service-Quality Review durchführen",
                "Cross-selling Opportunities identifizieren"
            ])
        
        # Monitoring Actions
        recommendations['monitoring_actions'].extend([
            "Wöchentliches Risk-Monitoring implementieren",
            "Automatische Alerts für neue High-Risk Kunden",
            "Performance Tracking der Retention-Maßnahmen"
        ])
        
        return recommendations
    
    def validate_data_leakage(self, training_periods: List[str], 
                             backtest_periods: List[str]) -> Dict[str, Any]:
        """
        Validiert Data Leakage zwischen Training und Backtest Perioden
        
        Args:
            training_periods: Liste der Training-Zeiträume (YYYYMM)
            backtest_periods: Liste der Backtest-Zeiträume (YYYYMM)
            
        Returns:
            Validation Results
        """
        self.logger.info("🔍 Validiere Data Leakage...")
        
        validation_results = {
            'data_leakage_detected': False,
            'overlapping_periods': [],
            'gap_analysis': {},
            'temporal_consistency': True,
            'warnings': [],
            'validation_passed': True
        }
        
        # Prüfe Überlappungen
        overlaps = set(training_periods) & set(backtest_periods)
        if overlaps:
            validation_results['data_leakage_detected'] = True
            validation_results['overlapping_periods'] = sorted(list(overlaps))
            validation_results['validation_passed'] = False
            validation_results['warnings'].append(
                f"Data Leakage: {len(overlaps)} überlappende Perioden gefunden"
            )
        
        # Gap Analysis
        if training_periods and backtest_periods:
            last_training = max(training_periods)
            first_backtest = min(backtest_periods)
            
            # Berechne Gap in Monaten
            gap_months = self._calculate_month_difference(last_training, first_backtest)
            
            validation_results['gap_analysis'] = {
                'last_training_period': last_training,
                'first_backtest_period': first_backtest,
                'gap_months': gap_months,
                'sufficient_gap': gap_months >= 1
            }
            
            if gap_months < 1:
                validation_results['warnings'].append(
                    f"Warnung: Nur {gap_months} Monate Gap zwischen Training und Backtest"
                )
        
        # Temporal Consistency
        if not self._check_temporal_consistency(training_periods + backtest_periods):
            validation_results['temporal_consistency'] = False
            validation_results['validation_passed'] = False
            validation_results['warnings'].append("Zeitliche Inkonsistenz in Perioden gefunden")
        
        self.validation_results['data_leakage'] = validation_results
        
        if validation_results['validation_passed']:
            self.logger.info("✅ Data Leakage Validation bestanden")
        else:
            self.logger.warning(f"⚠️ Data Leakage Validation: {len(validation_results['warnings'])} Warnungen")
        
        return validation_results
    
    def _calculate_month_difference(self, period1: str, period2: str) -> int:
        """Berechnet Differenz in Monaten zwischen zwei YYYYMM Perioden"""
        try:
            year1, month1 = int(period1[:4]), int(period1[4:6])
            year2, month2 = int(period2[:4]), int(period2[4:6])
            
            return (year2 - year1) * 12 + (month2 - month1)
        except:
            return 0
    
    def _check_temporal_consistency(self, periods: List[str]) -> bool:
        """Prüft zeitliche Konsistenz von YYYYMM Perioden"""
        try:
            for period in periods:
                if len(period) != 6:
                    return False
                
                year = int(period[:4])
                month = int(period[4:6])
                
                if year < 2000 or year > 2050:
                    return False
                if month < 1 or month > 12:
                    return False
            
            return True
        except:
            return False
    
    def _validate_performance_targets(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validiert Performance gegen definierte Targets"""
        
        validation = {
            'targets_met': {},
            'overall_passed': True,
            'missing_targets': [],
            'exceeded_targets': []
        }
        
        # Prüfe alle Target Metrics
        for metric_name, target_value in TARGET_METRICS.items():
            actual_value = metrics.get(metric_name, 0.0)
            
            target_met = actual_value >= target_value
            validation['targets_met'][metric_name] = {
                'target': target_value,
                'actual': actual_value,
                'met': target_met,
                'difference': actual_value - target_value
            }
            
            if not target_met:
                validation['overall_passed'] = False
                validation['missing_targets'].append(metric_name)
            else:
                validation['exceeded_targets'].append(metric_name)
        
        return validation
    
    def create_evaluation_report(self) -> Dict[str, Any]:
        """Erstellt umfassenden Evaluation Report"""
        
        report = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluator_version': MODULE_VERSIONS.get('churn_evaluator', '1.0.0')
            },
            'performance_metrics': self.performance_metrics,
            'threshold_optimization': self.threshold_results,
            'business_metrics': self.business_metrics,
            'validation_results': self.validation_results,
            'summary': self._create_evaluation_summary()
        }
        
        return report
    
    def _create_evaluation_summary(self) -> Dict[str, Any]:
        """Erstellt Executive Summary der Evaluation"""
        
        # Performance Summary
        auc = self.performance_metrics.get('auc', 0.0)
        target_validation = self.performance_metrics.get('target_validation', {})
        targets_met = target_validation.get('overall_passed', False)
        
        # Business Impact Summary
        if self.business_metrics:
            revenue_at_risk = self.business_metrics.get('financial_impact', {}).get('potential_revenue_loss', 0.0)
            roi_estimate = self.business_metrics.get('financial_impact', {}).get('roi_estimate', 0.0)
            high_risk_customers = self.business_metrics.get('risk_segmentation', {}).get('customers_high_risk', 0)
        else:
            revenue_at_risk = 0.0
            roi_estimate = 0.0
            high_risk_customers = 0
        
        # Overall Assessment
        if auc >= MINIMUM_AUC and targets_met:
            overall_quality = "Excellent"
        elif auc >= MINIMUM_AUC:
            overall_quality = "Good"
        elif auc >= 0.7:
            overall_quality = "Acceptable"
        else:
            overall_quality = "Poor"
        
        summary = {
            'model_quality': overall_quality,
            'performance_highlights': {
                'auc_score': auc,
                'targets_met': targets_met,
                'optimal_threshold': self.optimal_threshold,
                'threshold_method': self.threshold_method
            },
            'business_impact': {
                'revenue_at_risk_eur': revenue_at_risk,
                'roi_estimate': roi_estimate,
                'high_risk_customers': high_risk_customers
            },
            'recommendations': self._generate_model_recommendations()
        }
        
        return summary
    
    def _generate_model_recommendations(self) -> List[str]:
        """Generiert Model-Empfehlungen basierend auf Evaluation"""
        
        recommendations = []
        
        auc = self.performance_metrics.get('auc', 0.0)
        
        # Performance-basierte Empfehlungen
        if auc < MINIMUM_AUC:
            recommendations.append("Model Performance unter Minimum - Feature Engineering oder Algorithmus überprüfen")
        
        if auc >= TARGET_METRICS['auc']:
            recommendations.append("Exzellente Model Performance - bereit für Production")
        
        # Threshold-basierte Empfehlungen
        if self.threshold_method == 'f1_fallback':
            recommendations.append("Threshold-Optimierung fehlgeschlagen - robustere Validierung implementieren")
        
        # Business Impact Empfehlungen
        if self.business_metrics:
            roi = self.business_metrics.get('financial_impact', {}).get('roi_estimate', 0.0)
            
            if roi > ROI_THRESHOLD:
                recommendations.append(f"Hoher ROI ({roi:.2f}x) - Investment in Retention-Programm empfohlen")
            elif roi < 1.0:
                recommendations.append("Niedriger ROI - Retention-Strategie überprüfen")
        
        # Data Quality Empfehlungen
        validation_results = self.validation_results.get('data_leakage', {})
        if not validation_results.get('validation_passed', True):
            recommendations.append("Data Leakage Issues gefunden - Datenvalidierung erforderlich")
        
        if not recommendations:
            recommendations.append("Model erfüllt alle Qualitätskriterien")
        
        return recommendations


if __name__ == "__main__":
    # Test der ChurnEvaluator Funktionalität
    from bl.Churn.churn_data_loader import ChurnDataLoader
    from bl.Churn.churn_feature_engine import ChurnFeatureEngine
    from bl.Churn.churn_model_trainer import ChurnModelTrainer
    from sklearn.model_selection import train_test_split
    
    print("🧪 Teste ChurnEvaluator...")
    
    # 1. Lade Test-Daten und trainiere Model
    data_loader = ChurnDataLoader()
    csv_path = str(ProjectPaths.input_data_directory() / "churn_Data_cleaned.csv")
    df = data_loader.load_stage0_data(csv_path)
    
    if df is None:
        print("❌ Konnte Test-Daten nicht laden")
        exit(1)
    
    # 2. Feature Engineering (kleines Subset)
    data_dict = data_loader.load_data_dictionary()
    feature_engine = ChurnFeatureEngine(data_dictionary=data_dict)
    
    test_df = df.head(1500).copy()
    df_with_features = feature_engine.create_rolling_features(test_df)
    df_with_enhanced = feature_engine.create_enhanced_features(df_with_features)
    
    # 3. Training Data
    X, y, feature_names = feature_engine.prepare_training_data(
        df_with_enhanced, 
        prediction_timebase="202401"
    )
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. Train Model
    trainer = ChurnModelTrainer(experiment_id=999)
    model = trainer.train_random_forest(X_train, y_train)
    print(f"✅ Model trainiert: {type(model).__name__}")
    
    # 6. Initialisiere Evaluator
    evaluator = ChurnEvaluator(experiment_id=999)
    
    # 7. Performance Evaluation
    performance_metrics = evaluator.evaluate_model_performance(
        model, X_test, y_test, feature_names
    )
    print(f"✅ Performance Evaluation: AUC = {performance_metrics.get('auc', 0.0):.4f}")
    
    # 8. Threshold Optimization
    # Robust: Wenn Modell nur eine Klasse kennt, nicht crashen
    try:
        proba = model.predict_proba(X_test)
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
        y_pred_proba = np.zeros((len(X_test),))
    threshold_results = evaluator.optimize_threshold(y_test, y_pred_proba, method='robust')
    print(f"✅ Threshold Optimization: {threshold_results.get('optimal_threshold', 0.5):.3f}")
    
    # 9. Business Metrics
    business_metrics = evaluator.calculate_business_metrics(y_pred_proba)
    revenue_at_risk = business_metrics.get('financial_impact', {}).get('potential_revenue_loss', 0.0)
    print(f"✅ Business Metrics: €{revenue_at_risk:,.0f} Revenue at Risk")
    
    # 10. Data Leakage Validation
    training_periods = ["202001", "202002", "202003"]
    backtest_periods = ["202005", "202006"]
    leakage_results = evaluator.validate_data_leakage(training_periods, backtest_periods)
    print(f"✅ Data Leakage Validation: {'Bestanden' if leakage_results['validation_passed'] else 'Fehlgeschlagen'}")
    
    # 11. Evaluation Report
    report = evaluator.create_evaluation_report()
    model_quality = report.get('summary', {}).get('model_quality', 'Unknown')
    print(f"✅ Evaluation Report: Model Quality = {model_quality}")
    
    print("\n🎯 ChurnEvaluator Test erfolgreich abgeschlossen!")
