# bl-churn - Churn-Prediction Pipeline

**Last reviewed: 2025-09-29**

## 🎯 **Zweck**

Business-Logic für Churn-Prediction mit Enhanced Early Warning Model.

## 🏗️ **Architektur**

- **Feature-Engineering**: Rolling/Activity/Trend/Enhanced Features
- **Model-Training**: Enhanced Early Warning mit Kalibrierung
- **Evaluation**: Backtest mit Metriken (AUC, Precision, Recall)
- **Artefakte**: Modelle, Backtest-Results, Customer Details

## 🚀 **Quick Start**

### **Pipeline starten:**
```bash
# Über UI
http://localhost:8080/ → Experiment auswählen → "Churn" starten

# Über API
curl -X POST http://localhost:5050/run/churn -d '{"experiment_id":1}'
```

### **Ergebnisse ansehen:**
- **Management Studio**: http://localhost:5051/sql/
- **Tabellen**: `backtest_results`, `customer_churn_details`, `experiment_kpis`

## 📊 **Output-Tabellen**

- `backtest_results`: Churn-Wahrscheinlichkeiten pro Kunde
- `customer_churn_details`: Engineered Features + Metadaten
- `experiment_kpis`: Modell-Performance (AUC, Precision, Recall)
- `churn_model_metrics`: Detaillierte Modell-Metriken
- `churn_threshold_metrics`: Threshold-Optimierung

## 🔧 **Konfiguration**

- **Data Dictionary**: `config/shared/config/data_dictionary_optimized.json`
- **Feature-Mapping**: `config/shared/config/feature_mapping.json`
- **Algorithmus-Config**: `config/shared/config/algorithm_config.json`

## 📚 **Dokumentation**

**Zentrale Dokumentation:** [NEXT_STEPS.md](../NEXT_STEPS.md)

**Detaillierte Anleitungen:**
- [bl-churn/RUNBOOK.md](RUNBOOK.md) - Betriebsabläufe
- [bl-churn/nextSteps.md](nextSteps.md) - Entwicklungshinweise