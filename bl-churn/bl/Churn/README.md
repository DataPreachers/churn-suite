# Churn Prediction Pipeline - Business Logic Module

```yaml
module_info:
  name: "Churn Random Forest Pipeline"
  purpose: "Binary churn classification with enhanced early warning"
  status: "PRODUCTION"
  integration_level: "CORE_COMPONENT"
  performance_target: "AUC > 0.99"
  last_updated: "2025-09-18"
  ai_agent_optimized: true
```

## 🎯 **MODULE OVERVIEW**

### **Primary Functions:**
- **Binary Churn Classification** - RandomForest-based customer churn prediction
- **Enhanced Early Warning System** - 110+ engineered features with rolling windows
- **Threshold Optimization** - Scientific threshold calculation for business metrics
- **JSON-Database Integration** - Persistent storage with experiment tracking
- **Performance Evaluation** - Comprehensive metrics and business impact analysis

### **Business Impact:**
- **Customer Retention ROI** - Identifies high-risk customers for proactive intervention
- **Revenue Protection** - Prevents customer churn through early identification
- **Campaign Optimization** - Precision targeting reduces marketing costs
- **Business Intelligence** - Risk segmentation and customer lifetime value optimization

## 🏗️ **ARCHITECTURE COMPONENTS**

### **Core Classes:**
```python
# Primary Pipeline Components
ChurnAutoProcessor()          # Orchestrates experiment processing
ChurnDataLoader()            # Stage0 data loading and preprocessing  
ChurnFeatureEngine()         # 110+ feature engineering pipeline
ChurnModelTrainer()          # RandomForest training and optimization
ChurnEvaluator()             # Performance metrics and threshold optimization
ChurnPersistenceService()    # JSON-Database persistence layer

# Analysis Components  
EnhancedEarlyWarning()       # Legacy comprehensive pipeline
FeatureAnalysisEngine()     # Feature selection and analysis
ThresholdOptimizer()        # Scientific threshold calculation
```

### **Data Flow:**
```yaml
input: 
  - "Stage0 cache data (dynamic_system_outputs/stage0_cache/)"
  - "Data dictionary (config/data_dictionary_optimized.json)"
  - "Algorithm config (config/algorithm_config_optimized.json)"
  
process:
  1. "Data loading and validation"
  2. "110+ feature engineering with rolling windows"
  3. "SMOTE-based class balancing"
  4. "RandomForest training with hyperparameter tuning"
  5. "Scientific threshold optimization"
  6. "Performance evaluation and business metrics"
  
output:
  - "Trained models (models/Enhanced_EarlyWarning_*.joblib)"
  - "JSON-Database tables (5 churn-specific tables)"
  - "Performance reports and customer risk scores"
```

## 🚀 **QUICK START FOR AI-AGENTS**

### **Basic Usage:**
```bash
# Environment setup
source churn_prediction_env/bin/activate
cd /Users/klaus.reiners/Projekte/Cursor\ ChurnPrediction\ -\ Reengineering

# Process single experiment
python bl/Churn/churn_working_main.py --experiment-id 200 --training-from 202001 --training-to 202312 --prediction-timebase 202401

# Auto-process all pending experiments  
python bl/Churn/churn_auto_processor.py

# Check processing status
python bl/Churn/churn_auto_processor.py --status
```

### **Programmatic API:**
```python
from bl.Churn.churn_auto_processor import ChurnAutoProcessor
from bl.json_database.sql_query_interface import SQLQueryInterface

# Process experiments
processor = ChurnAutoProcessor()
results = processor.run_all_pending_experiments()

# Query results
qi = SQLQueryInterface()
metrics = qi.execute_query("SELECT * FROM churn_model_metrics WHERE experiment_id = 200")
```

## 📊 **CONFIGURATION & CONSTANTS**

### **Key Configuration Files:**
```yaml
config_files:
  algorithm_config: "config/algorithm_config_optimized.json"
  data_dictionary: "config/data_dictionary_optimized.json" 
  paths: "config/paths_config.py"
  constants: "bl/Churn/churn_constants.py"
```

### **Performance Targets:**
```yaml
target_metrics:
  auc: 0.99
  precision: 0.90
  recall: 0.95
  f1_score: 0.92
  processing_time: "< 30 seconds per experiment"
  
business_metrics:
  average_customer_value: 2500.00
  prevention_cost_ratio: 0.10
  roi_threshold: 3.0
```

## 🔗 **SYSTEM INTEGRATION**

### **Database Schema:**
```yaml
json_database_tables:
  churn_training_data: "Training/validation datasets with features"
  churn_predictions: "Customer risk scores and classifications"  
  churn_feature_importance: "Feature importance tracking"
  churn_model_metrics: "Performance metrics per experiment"
  churn_business_metrics: "Business impact and ROI calculations"
  
foreign_keys:
  - "All tables: id_experiments → experiments.experiment_id"
  
status_management:
  - "experiments.status: 'created' → 'processed'"
```

### **Dependencies:**
```yaml
internal_dependencies:
  - "bl/json_database/churn_json_database.py"
  - "bl/json_database/sql_query_interface.py"
  - "config/paths_config.py"
  
external_dependencies:
  - "scikit-learn >= 1.0"
  - "pandas >= 1.3"
  - "numpy >= 1.21"
  - "imbalanced-learn (SMOTE)"
```

## 📈 **PERFORMANCE & MONITORING**

### **Current Performance (Production):**
```yaml
model_performance:
  auc_roc: 0.878
  precision: 0.52  # at 2% churn rate
  recall: 0.696
  f1_score: 0.595
  customer_coverage: 4742
  features_engineered: "110+"
  
system_performance:
  avg_processing_time: "25 seconds"
  memory_usage: "~150MB"
  success_rate: "100%"
  
business_impact:
  customers_at_risk: 1247
  high_risk_customers: 342
  potential_revenue_protection: "€2.8M annually"
```

## 🔧 **TROUBLESHOOTING FOR AI-AGENTS**

### **Common Issues:**
```yaml
data_loading_errors:
  symptom: "Stage0 cache not found"
  solution: "Verify dynamic_system_outputs/stage0_cache/ exists and contains valid JSON files"
  
feature_engineering_errors:
  symptom: "Missing columns in feature engineering"
  solution: "Check data_dictionary_optimized.json for feature mappings"
  
model_training_errors:
  symptom: "SMOTE fails with insufficient samples"
  solution: "Increase training data timerange or adjust class balancing strategy"
  
json_database_errors:
  symptom: "Database write failures"  
  solution: "Check churn_json_database.py file permissions and disk space"
```

### **Performance Optimization:**
```yaml
optimization_tips:
  - "Use sampling for large datasets: --sample-rate 0.1"
  - "Parallel processing: Set n_jobs=-1 in RandomForest config"
  - "Memory optimization: Process experiments in batches"
  - "Feature selection: Use importance-based filtering"
```

## 📋 **AI-AGENT MAINTENANCE CHECKLIST**

### **After Code Changes:**
```yaml
validation_steps:
  - "Run: python bl/Churn/churn_auto_processor.py --validate"
  - "Check: All 5 JSON-Database tables populated correctly"
  - "Verify: Performance metrics within expected ranges"
  - "Test: End-to-end experiment processing"
  
update_requirements:
  - "Performance changes → Update metrics in this README"
  - "API changes → Update programmatic examples"
  - "New features → Update configuration section"
  - "Dependencies → Update requirements section"
```

---

**📅 Last Updated:** 2025-09-18  
**🤖 Optimized for:** AI-Agent maintenance and usage  
**🎯 Status:** Production-ready core component  
**🔗 Related:** docs/CHURN_ARCHITECTURE_SPECIFICATION.md

**📦 Outbox/Importer-Workflow (Sandbox)**

- Outbox-Pfade (via `config/paths_config.py`):
  - `ProjectPaths.outbox_churn_experiment_directory(experiment_id)`
- Churn-Auto-Processor exportiert nach erfolgreichem Lauf Backtest-JSON und KPIs in die Outbox.
- JSON-DB kann die Artefakte importieren:

```bash
python - <<'PY'
from bl.json_database.churn_json_database import ChurnJSONDatabase

db = ChurnJSONDatabase()
ok = db.import_from_outbox_churn(2)  # experiment_id
if ok:
    db.save()
    print('OK')
else:
    print('FAILED')
PY
```
