# bl-cox - Cox-Survival-Analyse

**Last reviewed: 2025-09-29**

## 🎯 **Zweck**

Business-Logic für Cox-Survival-Analyse mit Customer Risk Profiling und Prioritization.

## 🏗️ **Architektur**

- **Survival-Analyse**: Cox-Proportional-Hazards-Model
- **Risk-Profiling**: Customer-spezifische Survival-Wahrscheinlichkeiten
- **Prioritization**: ROI-basierte Kunden-Priorisierung
- **Segmentierung**: Digitalization-basierte Cluster-Analyse

## 🚀 **Quick Start**

### **Pipeline starten:**
```bash
# Über UI
http://localhost:5051/crud/index.html → Experiment auswählen → "Cox" starten

# Über API
curl -X POST http://localhost:5050/run/cox -d '{"experiment_id":1, "cutoff_exclusive":"202501"}'

# Coverage-Diagnose (lokal)
make cox
```

### **Ergebnisse ansehen:**
- **Management Studio**: http://localhost:5051/sql/
- **Tabellen**: `cox_survival`, `cox_prioritization_results`

## 📊 **Output-Tabellen**

- `cox_survival`: Survival-Wahrscheinlichkeiten (6/12/18/24 Monate)
- `cox_prioritization_results`: ROI-basierte Kunden-Priorisierung
- `customer_cox_details_{experiment_id}`: Customer Risk Profiles
- `churn_cox_fusion`: Fusion-View (Churn + Cox)

## 🔧 **Konfiguration**

- **Survival-Horizonte**: 6, 12, 18, 24 Monate
- **Prioritization**: ROI-basierte Scoring
- **Segmentierung**: Digitalization-Cluster

## 📚 **Dokumentation**

**Zentrale Dokumentation:** [NEXT_STEPS.md](../NEXT_STEPS.md)

**Detaillierte Anleitungen:**
- [bl-cox/RUNBOOK.md](RUNBOOK.md) - Betriebsabläufe

## ✅ **Aktuelle Coverage-Diagnose (Stand: 2025-10-06)**

- Report & Plots: `reports/cox_coverage_and_rolling.md` sowie `reports/plots/`
- Coverage nach Merge (Survival-Panel → Features):
  - 2018: 0 % (keine Feature-Historie vor Event)
  - 2019: 97,7 %
  - 2020: 98,1 %
  - 2021: 99,1 %
  - 2022: 98,4 %
  - 2023: 96,8 %
  - 2024: 97,1 %
  - 2025: 100 %
- Trainingsdaten nach Merge: 5 877 Kunden, 743 Events.
- Optionaler C-Index (Train): Fixed-Rollings ≈ 0,718 · Expanding ≈ 0,714 (jeweils ohne Leakage via `shift(1)`).
- Historie: 2019+ ausreichend (≥ 19 Monate Median); 2018 weiterhin nur 7 Monate → ggf. expanding Fenster nutzen oder Jahr auslassen.