# json-database - Zentrale Datenhaltung

**Last reviewed: 2025-09-29**

## 🎯 **Zweck**

Zentrale JSON-Database mit DuckDB-Interface für alle ML-Pipeline-Ergebnisse.

## 🏗️ **Architektur**

- **ChurnJSONDatabase**: Singleton-Pattern für zentrale Datenhaltung
- **SQLQueryInterface**: DuckDB-basierte SQL-Abfragen
- **LeakageGuard**: Datenqualität und Konsistenz
- **Persistenz**: JSON-Datei mit Schema-Validierung

## 🚀 **Quick Start**

### **SQL-Interface:**
```bash
# Management Studio
http://localhost:5051/sql/

# Direkte Abfragen
from bl.json_database.sql_query_interface import SQLQueryInterface
qi = SQLQueryInterface()
results = qi.execute_query("SELECT * FROM customer_churn_details")
```

### **Datenbank-Pfad:**
- **Hauptdatei**: `bl-churn/dynamic_system_outputs/churn_database.json`
- **Backup**: Automatische Sicherung bei Änderungen

## 📊 **Haupttabellen**

### **Churn-Pipeline:**
- `rawdata`: Rohdaten aus CSV-Import
- `backtest_results`: Churn-Wahrscheinlichkeiten
- `customer_churn_details`: Engineered Features
- `experiment_kpis`: Modell-Performance

### **Cox-Analyse:**
- `cox_survival`: Survival-Wahrscheinlichkeiten
- `cox_prioritization_results`: ROI-Priorisierung

### **SHAP-Erklärbarkeit:**
- `shap_global`: Globale Feature-Importance
- `shap_local_topk`: Lokale Top-K Erklärungen
- `shap_global_by_digitalization`: Segmentierte SHAP-Analyse

### **Counterfactuals:**
- `cf_individual`: Kunden-spezifische CF-Empfehlungen
- `cf_aggregate`: Feature-Impact-Analyse
- `cf_business_metrics`: ROI und Kosten-Nutzen-Bewertung
- `cf_*_by_digitalization`: Segmentierte CF-Ergebnisse

## 🔧 **Features**

- **Singleton-Pattern**: Zentrale Dateninstanz
- **Schema-Validierung**: Automatische Typ-Überprüfung
- **SQL-Interface**: DuckDB-basierte Abfragen
- **Persistenz**: JSON-Datei mit Backup

## 📚 **Dokumentation**

**Zentrale Dokumentation:** [NEXT_STEPS.md](../NEXT_STEPS.md)

**Detaillierte Anleitungen:**
- [json-database/RUNBOOK.md](RUNBOOK.md) - Betriebsabläufe