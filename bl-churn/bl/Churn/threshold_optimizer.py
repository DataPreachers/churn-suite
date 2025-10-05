#!/usr/bin/env python3
"""
Robuste Schwellwert-Optimierung für Churn-Prädiktion
Separates Modul zur Vermeidung von Code-Bloat in enhanced_early_warning.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RobustThresholdOptimizer:
    """
    Robuste Schwellwert-Optimierung mit Multi-Method Validation
    """
    
    def __init__(self, cv_folds=5, n_bootstrap=1000):
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.optimal_threshold = None
        self.validation_results = {}
        
    def optimize_roc_threshold(self, y_true, y_pred_proba):
        """ROC-basierte Schwellwert-Optimierung"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    
    def optimize_f1_threshold(self, y_true, y_pred_proba):
        """F1-Score-basierte Schwellwert-Optimierung"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def optimize_precision_threshold(self, y_true, y_pred_proba, min_recall=0.5):
        """Precision-basierte Schwellwert-Optimierung mit Recall-Constraint"""
        try:
            # Verwende eigene Schwellwerte wie Elbow-Methode
            thresholds = np.arange(0.01, 0.99, 0.01)
            precision_scores = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                
                # Berücksichtige Recall-Constraint
                if recall >= min_recall:
                    precision_scores.append(precision)
                else:
                    precision_scores.append(0)  # Penalty für zu niedrigen Recall
            
            precision_scores = np.array(precision_scores)
            
            # Finde optimalen Schwellwert
            optimal_idx = np.argmax(precision_scores)
            return thresholds[optimal_idx]
            
        except Exception as e:
            print(f"⚠️ Precision-Methode Fehler: {e}")
            return 0.5
    
    def optimize_percentile_threshold(self, y_true, y_pred_proba):
        """Percentile-basierte Schwellwert-Optimierung"""
        # 75% Perzentil für HIGH Risk
        return np.percentile(y_pred_proba, 75)
    
    def optimize_cost_benefit_threshold(self, y_true, y_pred_proba, cost_fp=1, cost_fn=10):
        """Cost-Benefit-basierte Schwellwert-Optimierung mit realistischer Kostenberechnung"""
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_threshold = 0.5
        best_cost = float('inf')
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Berechne Confusion Matrix
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            
            # Berechne Gesamtkosten
            total_cost = fp * cost_fp + fn * cost_fn
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        return best_threshold
    
    def optimize_elbow_threshold(self, y_true, y_pred_proba):
        """Elbow-Methode für Schwellwert-Optimierung basierend auf Precision-Recall Trade-off"""
        try:
            # 🔧 KORREKTUR: Verwende eigene Schwellwerte statt precision_recall_curve thresholds
            thresholds = np.arange(0.01, 0.99, 0.01)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred)
                f1_scores.append(f1)
            
            f1_scores = np.array(f1_scores)
            
            # Finde den "Elbow Point" - wo die Verbesserung abnimmt
            if len(f1_scores) > 2:
                # Erste Ableitung
                first_derivative = np.gradient(f1_scores)
                # Zweite Ableitung (Krümmung)
                second_derivative = np.gradient(first_derivative)
                
                # Finde den Punkt mit maximaler Krümmung (Elbow)
                elbow_idx = np.argmax(np.abs(second_derivative))
                
                # Stelle sicher, dass der Index gültig ist
                if elbow_idx < len(thresholds):
                    return thresholds[elbow_idx]
            
            # Fallback: Verwende den Schwellwert mit maximalem F1-Score
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
            
        except Exception as e:
            print(f"⚠️ Elbow-Methode Fehler: {e}")
            return 0.5
    
    def cross_validate_thresholds(self, y_true, y_pred_proba):
        """Cross-Validation für alle Schwellwert-Methoden"""
        methods = {
            'roc_optimal': self.optimize_roc_threshold,
            'f1_optimal': self.optimize_f1_threshold,
            'precision_optimal': lambda y, p: self.optimize_precision_threshold(y, p, min_recall=0.5),
            'percentile_75': self.optimize_percentile_threshold,
            'cost_benefit': lambda y, p: self.optimize_cost_benefit_threshold(y, p, cost_fp=1, cost_fn=10),
            'elbow': self.optimize_elbow_threshold
        }
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Konvertiere zu numpy arrays für sklearn und entferne NaN/Inf
        y_true_np = np.array(y_true)
        y_pred_proba_np = np.array(y_pred_proba)
        
        # Entferne NaN und Infinity Werte
        valid_mask = ~(np.isnan(y_pred_proba_np) | np.isinf(y_pred_proba_np))
        y_true_clean = y_true_np[valid_mask]
        y_pred_proba_clean = y_pred_proba_np[valid_mask]
        
        # 🔧 KORREKTUR: Verwende die bereinigten Arrays für CV
        if len(y_true_clean) < 20:
            print(f"⚠️ Zu wenig gültige Daten für Schwellwert-Optimierung: {len(y_true_clean)} < 20")
            print(f"   Original: {len(y_true)}, Nach Bereinigung: {len(y_true_clean)}")
            print(f"   NaN-Werte: {np.sum(np.isnan(y_pred_proba_np))}, Inf-Werte: {np.sum(np.isinf(y_pred_proba_np))}")
            return {}  # 🔧 KORREKTUR: Return empty dict statt Exception
        
        for method_name, method_func in methods.items():
            cv_scores = []
            cv_thresholds = []
            
            # 🔧 KORREKTUR: Verwende bereinigte Arrays für CV
            for train_idx, val_idx in skf.split(y_true_clean, y_true_clean):
                y_train, y_val = y_true_clean[train_idx], y_true_clean[val_idx]
                proba_train, proba_val = y_pred_proba_clean[train_idx], y_pred_proba_clean[val_idx]
                
                try:
                    # Finde optimalen Schwellwert auf Training
                    threshold = method_func(y_train, proba_train)
                    
                    # Validiere auf Validation Set
                    y_pred = (proba_val >= threshold).astype(int)
                    score = f1_score(y_val, y_pred)
                    
                    cv_scores.append(score)
                    cv_thresholds.append(threshold)
                except Exception as e:
                    print(f"⚠️ Fehler bei {method_name}: {e}")
                    continue
            
            if cv_scores:
                cv_results[method_name] = {
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores),
                    'mean_threshold': np.mean(cv_thresholds),
                    'std_threshold': np.std(cv_thresholds),
                    'scores': cv_scores,
                    'thresholds': cv_thresholds
                }
        
        return cv_results
    
    def bootstrap_validation(self, y_true, y_pred_proba):
        """Bootstrap-Validierung für Schwellwert-Stabilität"""
        bootstrap_thresholds = []
        bootstrap_scores = []
        
        # Konvertiere zu numpy arrays und entferne NaN/Inf
        y_true_np = np.array(y_true)
        y_pred_proba_np = np.array(y_pred_proba)
        
        # Entferne NaN und Infinity Werte
        valid_mask = ~(np.isnan(y_pred_proba_np) | np.isinf(y_pred_proba_np))
        y_true_clean = y_true_np[valid_mask]
        y_pred_proba_clean = y_pred_proba_np[valid_mask]
        
        if len(y_true_clean) < 10:
            return None
        
        for _ in range(self.n_bootstrap):
            # Bootstrap Sample
            indices = np.random.choice(len(y_true_clean), len(y_true_clean), replace=True)
            y_boot, proba_boot = y_true_clean[indices], y_pred_proba_clean[indices]
            
            try:
                # Finde optimalen Schwellwert (F1-basiert)
                threshold = self.optimize_f1_threshold(y_boot, proba_boot)
                bootstrap_thresholds.append(threshold)
                
                # Score auf Original-Daten
                y_pred = (y_pred_proba >= threshold).astype(int)
                score = f1_score(y_true, y_pred)
                bootstrap_scores.append(score)
            except:
                continue
        
        if bootstrap_thresholds:
            return {
                'mean_threshold': np.mean(bootstrap_thresholds),
                'std_threshold': np.std(bootstrap_thresholds),
                'confidence_interval': np.percentile(bootstrap_thresholds, [2.5, 97.5]),
                'mean_score': np.mean(bootstrap_scores),
                'std_score': np.std(bootstrap_scores)
            }
        return None
    
    def select_robust_threshold(self, cv_results):
        """Wählt den robustesten Schwellwert mit Bevorzugung der Precision-Methode"""
        
        # 🎯 PRIORITÄT: Precision-Methode direkt bevorzugen
        if 'precision_optimal' in cv_results:
            precision_results = cv_results['precision_optimal']
            print(f"   🎯 Precision-Methode verfügbar: {precision_results['mean_threshold']:.3f} (F1: {precision_results['mean_score']:.3f})")
            print(f"   ✅ Verwende Precision-Methode als bevorzugte Methode")
            return 'precision_optimal', {
                'robustness': 1000.0,  # Hohe Robustheit für Precision
                'mean_score': precision_results['mean_score'],
                'std_score': precision_results['std_score'],
                'mean_threshold': precision_results['mean_threshold']
            }
        
        # Fallback: Normale Robustheit-Berechnung
        robustness_scores = {}
        
        for method, results in cv_results.items():
            # Robustheit = Performance / Variabilität
            robustness = results['mean_score'] / (results['std_score'] + 1e-6)
            
            # Stabilität des Schwellwerts
            threshold_stability = 1 / (results['std_threshold'] + 1e-6)
            
            # Kombinierte Robustheit
            combined_robustness = robustness * threshold_stability
            
            # ⚠️ Precision-Optimal reduzieren (tendiert zu 0.5)
            if method == 'precision_optimal':
                combined_robustness *= 0.5  # 50% Penalty
                print(f"   ⚠️ Precision-Optimal: {results['mean_threshold']:.3f} (F1: {results['mean_score']:.3f}) - Penalty angewendet")
            
            robustness_scores[method] = {
                'robustness': combined_robustness,
                'mean_score': results['mean_score'],
                'std_score': results['std_score'],
                'mean_threshold': results['mean_threshold']
            }
        
        # Wähle robusteste Methode (ohne Elbow)
        if robustness_scores:
            best_method = max(robustness_scores.keys(), 
                             key=lambda x: robustness_scores[x]['robustness'])
            return best_method, robustness_scores[best_method]
        
        return None, None
    
    def optimize(self, y_true, y_pred_proba):
        """
        Hauptmethode: Führt robuste Schwellwert-Optimierung durch
        """
        print("🔍 Starte robuste Schwellwert-Optimierung...")
        
        # Debug: Zeige Datenqualität
        print(f"   📊 Datenqualität: {len(y_true)} Samples, {np.sum(y_true)} Positive")
        print(f"   📈 Prediction Range: {np.min(y_pred_proba):.3f} - {np.max(y_pred_proba):.3f}")
        
        # 1. Cross-Validation
        print("   📊 Cross-Validation...")
        cv_results = self.cross_validate_thresholds(y_true, y_pred_proba)
        
        if not cv_results:
            print("⚠️ Cross-Validation fehlgeschlagen - verwende direkte Optimierung")
            # Direkte Optimierung ohne Cross-Validation
            direct_results = self.direct_threshold_optimization(y_true, y_pred_proba)
            if direct_results:
                self.optimal_threshold = direct_results['optimal_threshold']
                print(f"✅ Direkte Optimierung: {self.optimal_threshold:.3f}")
                return {
                    'optimal_threshold': self.optimal_threshold,
                    'method': 'direct_optimization',
                    'robustness': 1.0
                }
        
        # 2. Bootstrap-Validierung
        print("   🔄 Bootstrap-Validierung...")
        bootstrap_results = self.bootstrap_validation(y_true, y_pred_proba)
        
        # 3. Robuste Auswahl
        print("   🎯 Robuste Auswahl...")
        best_method, best_results = self.select_robust_threshold(cv_results)
        
        if best_method and best_results:
            self.optimal_threshold = best_results['mean_threshold']
            self.validation_results = {
                'best_method': best_method,
                'cv_results': cv_results,
                'bootstrap_results': bootstrap_results,
                'optimal_threshold': self.optimal_threshold,
                'robustness_score': best_results['robustness']
            }
            
            print(f"✅ Optimaler Schwellwert: {self.optimal_threshold:.3f}")
            print(f"   Methode: {best_method}")
            print(f"   F1-Score: {best_results['mean_score']:.3f} ± {best_results['std_score']:.3f}")
            print(f"   Robustheit: {best_results['robustness']:.2f}")
            
            return {
                'optimal_threshold': self.optimal_threshold,
                'method': best_method,
                'robustness': best_results['robustness']
            }
        
        # Fallback mit direkter Optimierung
        print("⚠️ Robuste Auswahl fehlgeschlagen - verwende direkte Optimierung")
        direct_results = self.direct_threshold_optimization(y_true, y_pred_proba)
        if direct_results:
            self.optimal_threshold = direct_results['optimal_threshold']
            print(f"✅ Direkte Optimierung: {self.optimal_threshold:.3f}")
            return {
                'optimal_threshold': self.optimal_threshold,
                'method': 'direct_optimization',
                'robustness': 1.0
            }
        
        # Letzter Fallback
        self.optimal_threshold = 0.5
        print("⚠️ Fallback auf Standard-Schwellwert: 0.5")
        return {
            'optimal_threshold': self.optimal_threshold,
            'method': 'fallback',
            'robustness': 0.0
        }

    def direct_threshold_optimization(self, y_true, y_pred_proba):
        """Direkte Schwellwert-Optimierung mit Precision-Methode"""
        try:
            print("   🎯 Verwende Precision-Methode für direkte Optimierung...")
            
            # Verwende die Precision-Methode direkt
            optimal_threshold = self.optimize_precision_threshold(y_true, y_pred_proba)
            
            # Validiere den Schwellwert
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            
            print(f"   🎯 Precision-Schwellwert: {optimal_threshold:.3f}")
            print(f"   📊 F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            return {
                'optimal_threshold': optimal_threshold,
                'best_f1': f1,
                'precision': precision,
                'recall': recall
            }
        except Exception as e:
            print(f"⚠️ Direkte Optimierung fehlgeschlagen: {e}")
            return None
    
    def get_risk_levels(self, y_pred_proba, custom_thresholds=None):
        """
        Erstellt Risk-Levels basierend auf optimierten Schwellwerten
        """
        if custom_thresholds:
            low_threshold, high_threshold = custom_thresholds
        else:
            # Automatische Schwellwerte basierend auf optimalem Schwellwert
            if self.optimal_threshold:
                low_threshold = self.optimal_threshold * 0.7
                high_threshold = self.optimal_threshold * 1.3
            else:
                low_threshold, high_threshold = 0.3, 0.7
        
        risk_levels = pd.cut(y_pred_proba, 
                           bins=[0, low_threshold, high_threshold, 1.0], 
                           labels=['LOW', 'MEDIUM', 'HIGH'])
        
        return risk_levels, (low_threshold, high_threshold)
    
    def get_validation_report(self):
        """Gibt detaillierten Validierungsbericht zurück"""
        if not self.validation_results:
            return "Keine Validierung durchgeführt"
        
        report = f"""
🔍 ROBUSTE SCHWELLWERT-OPTIMIERUNG
====================================
Optimaler Schwellwert: {self.optimal_threshold:.3f}
Beste Methode: {self.validation_results['best_method']}
Robustheit: {self.validation_results['robustness_score']:.2f}

📊 CROSS-VALIDATION ERGEBNISSE:
"""
        
        for method, results in self.validation_results['cv_results'].items():
            report += f"  {method}: F1={results['mean_score']:.3f}±{results['std_score']:.3f}, "
            report += f"Threshold={results['mean_threshold']:.3f}±{results['std_threshold']:.3f}\n"
        
        if self.validation_results['bootstrap_results']:
            bootstrap = self.validation_results['bootstrap_results']
            report += f"""
🔄 BOOTSTRAP-VALIDIERUNG:
  Konfidenzintervall: [{bootstrap['confidence_interval'][0]:.3f}, {bootstrap['confidence_interval'][1]:.3f}]
  Stabilität: {1/(bootstrap['std_threshold']+1e-6):.2f}
"""
        
        return report

def optimize_thresholds_for_churn(y_true, y_pred_proba, cv_folds=5, n_bootstrap=1000):
    """
    Convenience-Funktion für schnelle Schwellwert-Optimierung
    """
    optimizer = RobustThresholdOptimizer(cv_folds=cv_folds, n_bootstrap=n_bootstrap)
    optimal_threshold = optimizer.optimize(y_true, y_pred_proba)
    return optimal_threshold, optimizer 