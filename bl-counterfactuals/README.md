# bl-counterfactuals - Counterfactuals-Analyse

**Last reviewed: 2025-09-29**

## 🎯 **Zweck**

Business-Logic für Counterfactuals-Analyse mit SHAP-Integration und Digitalization-Segmentierung.

## 🏗️ **Architektur**

- **SHAP-Integration**: Feature-Selektion basierend auf SHAP-Ergebnissen
- **CF-Suche**: Greedy coordinate descent mit unit weights
- **Business-Metriken**: ROI-Analyse und Kosten-Nutzen-Bewertung
- **Segmentierung**: Digitalization-basierte Cluster-Analyse

## 🚀 **Quick Start**

### **Pipeline starten:**
```bash
# Über UI
http://localhost:8080/ → Experiment auswählen → "CF" starten

# Über API
curl -X POST http://localhost:5050/run/cf -d '{"experiment_id":1}'
```

### **Ergebnisse ansehen:**
- **Management Studio**: http://localhost:5051/sql/
- **Tabellen**: `cf_individual`, `cf_aggregate`, `cf_business_metrics`

## 📊 **Output-Tabellen**

- `cf_individual`: Kunden-spezifische CF-Empfehlungen
- `cf_aggregate`: Feature-Impact-Analyse
- `cf_business_metrics`: ROI und Kosten-Nutzen-Bewertung
- `cf_individual_by_digitalization`: Segmentierte CF-Ergebnisse
- `cf_aggregate_by_digitalization`: Segmentierte Feature-Impacts
- `cf_business_metrics_by_digitalization`: Segmentierte ROI-Analyse

## 🔧 **Features**

- **SHAP-basierte Feature-Selektion**: Nur relevante Features werden modifiziert
- **Unit-Cost-Distanzfunktion**: Alle Features mit Gewicht 1.0
- **Vollständige Analyse**: 100% Sample (alle Kunden im Zielbereich)
- **Digitalization-Segmentierung**: Cluster-spezifische CF-Analyse

## 📚 **Dokumentation**

**Zentrale Dokumentation:** [NEXT_STEPS.md](../NEXT_STEPS.md)

**Detaillierte Anleitungen:**
- [bl-counterfactuals/RUNBOOK.md](RUNBOOK.md) - Betriebsabläufe
- [bl-counterfactuals/nextSteps.md](nextSteps.md) - Entwicklungshinweise