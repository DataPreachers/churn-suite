# Counterfactuals Analysis System - Business Logic Module

```yaml
module_info:
  name: "Counterfactual Analysis & Feature Recommendations"
  purpose: "AI-driven actionable insights for customer churn prevention"
  status: "PRODUCTION"
  integration_level: "ADVANCED_ANALYTICS"
  performance_target: "ROI > 3.0 recommendations"
  last_updated: "2025-09-18"  
  ai_agent_optimized: true
```

## 🎯 **MODULE OVERVIEW**

### **Primary Functions:**
- **Counterfactual Search** - Identifies minimal feature changes to reduce churn risk
- **Cost-Benefit Analysis** - ROI calculations for intervention recommendations
- **Feature Impact Mapping** - Raw-to-engineered feature relationship analysis
- **Business Recommendations** - Actionable customer-specific interventions
- **Policy-Based Constraints** - Realistic feature modification boundaries

### **Business Impact:**
- **Intervention Optimization** - Precise recommendations for high-value customers
- **ROI-Driven Decisions** - Cost-effective customer retention strategies  
- **Resource Allocation** - Priority ranking based on impact and feasibility
- **Actionable Intelligence** - Specific steps to reduce customer churn risk
- **Strategic Planning** - Feature importance for business development

## 🏗️ **ARCHITECTURE COMPONENTS**

### **Core Classes:**
```python
# Primary Analysis Components
Policy()                     # Feature modification constraints and costs
counterfactuals_cli.py      # Main CLI interface for CF analysis
cf_raw_reporter.py          # Feature name mapping (engineered ↔ raw)

# Core Functions
find_counterfactual()       # Greedy coordinate descent CF search
build_model()               # RandomForest classifier with calibration
build_surrogate_prob_model() # RF probability approximation
project_value()             # Policy-constrained feature projections
weighted_l2()               # Cost-weighted distance calculation
```

### **Data Flow:**
```yaml
input:
  - "customer_details table from JSON-Database"
  - "cf_cost_policy.json (feature constraints and costs)"
  - "feature_mapping.json (raw to engineered mapping)"
  - "Experiment ID for customer selection"
  
process:
  1. "Load customers with churn probability 0.4-0.9"
  2. "Apply policy-based feature selection"
  3. "Train surrogate probability model"
  4. "Execute greedy counterfactual search"
  5. "Calculate business metrics (ROI, costs)"
  6. "Generate feature recommendations"
  
output:
  - "cf_individual: Customer-specific recommendations"
  - "cf_aggregate: Feature impact analysis"  
  - "cf_business_metrics: ROI and cost analysis"
  - "cf_feature_recommendations: Actionable insights"
  - "cf_cost_analysis: Policy configuration analysis"
```

## 🚀 **QUICK START FOR AI-AGENTS**

### **Basic Usage:**
```bash
# Environment setup
source churn_prediction_env/bin/activate
cd /Users/klaus.reiners/Projekte/Cursor\ ChurnPrediction\ -\ Reengineering

# Analyze counterfactuals for experiment (20% sample)
python bl/Counterfactuals/counterfactuals_cli.py --experiment-id 1 --sample 0.2

# Full analysis for all customers
python bl/Counterfactuals/counterfactuals_cli.py --experiment-id 1 --sample 1.0 

# Limited analysis (first 100 customers)  
python bl/Counterfactuals/counterfactuals_cli.py --experiment-id 1 --limit 100
```

### **Programmatic API:**
```python
from bl.Counterfactuals.counterfactuals_cli import run
from bl.json_database.sql_query_interface import SQLQueryInterface

# Run counterfactual analysis
success = run(experiment_id=1, sample=0.2)

# Query results
qi = SQLQueryInterface()
individual = qi.execute_query("SELECT * FROM cf_individual WHERE id_experiments = 1")
business_metrics = qi.execute_query("SELECT * FROM cf_business_metrics WHERE experiment_id = 1")
recommendations = qi.execute_query("SELECT * FROM cf_feature_recommendations WHERE experiment_id = 1")
```

## 📊 **CONFIGURATION & POLICY**

### **Key Configuration Files:**
```yaml
config_files:
  cost_policy: "config/cf_cost_policy.json"
  feature_mapping: "config/feature_mapping.json"
  data_dictionary: "config/data_dictionary_optimized.json"
```

### **Policy Configuration Example:**
```yaml
cf_cost_policy_structure:
  default_step: 0.1
  features:
    I_CONSULTING:
      type: "continuous"
      min: 0.0
      max: 10.0
      step: 0.5
      weight: 2.0  # Implementation cost multiplier
    N_DIGITALIZATIONRATE:
      type: "continuous"  
      min: 0.0
      max: 1.0
      step: 0.05
      weight: 1.5
    I_UHD:
      type: "integer"
      min: 0
      max: 50
      step: 1
      weight: 1.0
```

### **Business Metrics Configuration:**
```yaml
business_constants:
  avg_customer_value: 1000.0  # EUR - Customer Lifetime Value
  churn_cost_factor: 0.2      # 20% of CLV as churn cost
  roi_thresholds:
    highly_recommended: 3.0   # ROI > 300%
    recommended: 1.0          # ROI > 100%
    consider: 0.0             # ROI > 0%
```

## 🔗 **SYSTEM INTEGRATION**

### **Database Schema:**
```yaml
json_database_tables:
  cf_individual:
    description: "Customer-specific counterfactual recommendations"
    key_fields: ["Kunde", "p_old", "p_new", "relative_reduction", "l2_weighted", "top_changes"]
    
  cf_aggregate:
    description: "Aggregated feature impact analysis"
    key_fields: ["feature", "count", "median_abs_delta"]
    
  cf_business_metrics:
    description: "ROI and cost-benefit analysis per customer"
    key_fields: ["customer_id", "roi", "net_benefit", "recommendation"]
    
  cf_feature_recommendations:
    description: "Actionable feature-level recommendations"
    key_fields: ["customer_id", "feature_name", "delta", "feature_cost", "cost_per_reduction"]
    
  cf_cost_analysis:
    description: "Policy configuration and cost structure analysis"
    key_fields: ["raw_feature", "weight", "feature_type", "engineered_count"]
```

### **Dependencies:**
```yaml
internal_dependencies:
  - "bl/json_database/churn_json_database.py"
  - "bl/json_database/sql_query_interface.py"
  - "bl/json_database/leakage_guard.py (load_cf_cost_policy)"
  - "config/paths_config.py"
  
external_dependencies:
  - "scikit-learn >= 1.0 (RandomForest, CalibratedClassifierCV)"
  - "pandas >= 1.3"
  - "numpy >= 1.21"
```

## 📈 **PERFORMANCE & MONITORING**

### **Algorithm Performance:**
```yaml
counterfactual_search:
  max_iterations: 200
  convergence_target: "20% relative churn risk reduction"
  success_rate: "> 80% of analyzed customers"
  avg_search_time: "< 5 seconds per customer"
  
model_performance:
  surrogate_accuracy: "R² > 0.85 for probability approximation"
  calibration_quality: "Isotonic regression with 3-fold CV"
  feature_selection: "Policy-constrained feature space only"
```

### **Business Impact Metrics:**
```yaml
roi_distribution:
  highly_recommended: "> 30% of customers (ROI > 3.0)"
  recommended: "> 50% of customers (ROI > 1.0)"
  actionable: "> 70% of customers (ROI > 0.0)"
  
cost_effectiveness:
  avg_implementation_cost: "< 10% of customer lifetime value"
  avg_potential_savings: "20% of CLV for prevented churns"  
  net_benefit_positive: "> 60% of recommendations"
```

## 🔧 **TROUBLESHOOTING FOR AI-AGENTS**

### **Common Issues:**
```yaml
policy_configuration_errors:
  symptom: "No suitable policy features found"
  solution: "Verify cf_cost_policy.json contains features present in customer_details"
  validation: "Check feature_mapping.json for raw-to-engineered mappings"
  
counterfactual_search_failures:
  symptom: "No counterfactuals found (no_cf_found = true)"
  solution: "Adjust target reduction (currently 20%) or increase max_iterations"
  optimization: "Review policy constraints (min/max values too restrictive)"
  
model_training_errors:  
  symptom: "Surrogate model fails to converge"
  solution: "Check data quality in customer_details table"
  fallback: "System automatically falls back to binary classification model"
  
database_persistence_errors:
  symptom: "JSON-Database write failures"
  solution: "Verify churn_json_database.py write permissions"
  recovery: "Check Trash potential/ for database backups"
```

### **Performance Optimization:**
```yaml
optimization_strategies:
  customer_filtering: "Focus on customers with 0.4 ≤ churn_prob ≤ 0.9"
  feature_selection: "Use only policy-defined features (no fallbacks)"
  batch_processing: "Process customers in batches to manage memory"
  early_stopping: "Stop search when target reduction achieved"
```

## 💡 **BUSINESS INTELLIGENCE EXAMPLES**

### **High-Impact Customer Analysis:**
```sql
-- Top ROI opportunities
SELECT customer_id, roi, net_benefit, recommendation
FROM cf_business_metrics
WHERE recommendation IN ('highly_recommended', 'recommended')
ORDER BY net_benefit DESC
LIMIT 20;

-- Feature impact analysis  
SELECT raw_feature, engineered_count, weight, feature_type
FROM cf_cost_analysis
ORDER BY weight * engineered_count DESC;

-- Cost-effective recommendations
SELECT 
    customer_id,
    feature_name,
    delta,
    feature_cost,
    cost_per_reduction
FROM cf_feature_recommendations
WHERE cost_per_reduction < 5.0  -- Good cost efficiency
ORDER BY cost_per_reduction ASC;
```

### **Strategic Feature Planning:**
```sql
-- Most frequently recommended features
SELECT 
    feature,
    count,
    median_abs_delta,
    RANK() OVER (ORDER BY count DESC) as impact_rank
FROM cf_aggregate  
ORDER BY count DESC;

-- Customer segmentation by intervention complexity
SELECT 
    CASE 
        WHEN implementation_cost < 50 THEN 'Low Cost'
        WHEN implementation_cost < 200 THEN 'Medium Cost'  
        ELSE 'High Cost'
    END as cost_segment,
    COUNT(*) as customer_count,
    AVG(roi) as avg_roi,
    AVG(net_benefit) as avg_net_benefit
FROM cf_business_metrics
GROUP BY cost_segment
ORDER BY avg_net_benefit DESC;
```

## 📋 **AI-AGENT MAINTENANCE CHECKLIST**

### **After Code Changes:**
```yaml
validation_steps:
  - "Test: counterfactuals_cli.py with sample data"
  - "Verify: All 5 CF tables populated in JSON-Database"
  - "Check: ROI calculations produce reasonable values"
  - "Validate: Feature recommendations are actionable"
  
performance_validation:  
  - "Benchmark: CF search completes within 5s per customer"
  - "Check: Success rate > 80% for CF generation"
  - "Monitor: Memory usage during large batch processing"
  - "Verify: Surrogate model accuracy R² > 0.85"
```

### **Policy Configuration Updates:**
```yaml
policy_maintenance:
  - "Review: cf_cost_policy.json feature weights quarterly"
  - "Update: Customer lifetime value and churn cost factors"
  - "Validate: Feature constraints reflect business reality"
  - "Test: Policy changes with small sample before full deployment"
  
feature_mapping_sync:
  - "Verify: feature_mapping.json includes all new engineered features"
  - "Check: Raw feature names consistent across systems"
  - "Update: Feature mappings when data dictionary changes"
```

### **Business Impact Monitoring:**
```yaml
roi_tracking:
  - "Monitor: Percentage of highly_recommended customers"
  - "Track: Average net benefit per recommended intervention"  
  - "Validate: Business assumptions (CLV, churn costs) annually"
  - "Report: CF success stories and implemented interventions"
```

---

**📅 Last Updated:** 2025-09-18  
**🤖 Optimized for:** AI-Agent maintenance and usage  
**🎯 Status:** Production-ready advanced analytics component  
**🔗 Related:** config/cf_cost_policy.json, config/feature_mapping.json
