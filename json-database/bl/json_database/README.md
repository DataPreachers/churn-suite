# JSON Database & SQL Query Interface - Business Logic Module

```yaml
module_info:
  name: "Enterprise JSON Database with DuckDB SQL Interface"
  purpose: "Unified data persistence and SQL analytics for Churn & Cox systems"
  status: "PRODUCTION"
  integration_level: "CORE_INFRASTRUCTURE"
  performance_target: "< 2s complex queries"
  last_updated: "2025-09-21"
  ai_agent_optimized: true
```

## 🎯 **MODULE OVERVIEW**

### **Primary Functions:**
- **JSON Database Management** - Unified storage for all Churn & Cox data
- **DuckDB SQL Interface** - Native SQL queries with enterprise performance
- **Schema Management** - 20 optimized tables with referential integrity
- **Data Leakage Protection** - Temporal validation and access control
- **OLAP Analytics** - Multi-dimensional business intelligence queries

### **Business Impact:**
- **Unified Data Architecture** - Single source of truth for all predictions
- **Real-time Analytics** - Sub-2-second complex query performance
- **Business Intelligence** - SQL-based reporting and dashboard capabilities
- **Data Governance** - Structured schemas with validation and integrity
- **Enterprise Scalability** - Handles 900K+ records seamlessly

## 🏗️ **ARCHITECTURE COMPONENTS**

### **Core Classes:**
```python
# Primary Database Components
ChurnJSONDatabase()          # Main database management and persistence
SQLQueryInterface()          # DuckDB-powered SQL query engine
LeakageGuard()              # Data leakage prevention and temporal validation

# Utility Components  
query_churn_database.py     # CLI tool for direct SQL queries
```

### **Database Schema (20 Tables):**
```yaml
core_tables:
  files: "File metadata and hash tracking"
  rawdata: "Stage0 raw data cache"
  experiments: "Experiment definitions and status tracking"
```

## 🚀 **QUICK START**

### **End-to-End Ingestion → rawdata (Union)**
```bash
# 1) CSV→Stage0→Outbox (über bl-input) + 2) Outbox→rawdata (Union, replace) in einem Schritt
make -C bl-workspace ingest
```

### **Programmatic API (Union-Import aus Outbox):**
```python
from bl.json_database.churn_json_database import ChurnJSONDatabase

db = ChurnJSONDatabase()
added = db.import_from_outbox_stage0_union(replace=True)
db.save()
print(f"Imported rows: {added}")
```

## 📊 **CONFIGURATION & SCHEMA**

-### **Type System (Data Dictionary ist Quelle der Wahrheit)**
- Typregeln (wirksam ab Stage0, keine Laufzeit-Casts erzwungen):
  - `Kunde`: INTEGER
  - `i_*`: INTEGER (inkl. `I_Alive` als 1=alive, 0=churned)
  - `I_TIMEBASE`: INTEGER (Format YYYYMM)
  - `n_*`: DOUBLE (float)
- Datei: `json-database/config/shared/config/data_dictionary_optimized.json`
- Das SQL-Interface validiert Typen optional, erzwingt aber keine Konvertierung (kein „hard cast“).

### **Key Configuration:**
```yaml
database_file: "dynamic_system_outputs/churn_database.json"
backup_strategy: "Automatic backup before major operations"
schema_validation: "Type validation based on Data Dictionary"
performance_optimization: "In-memory DuckDB with columnar storage"

connection_settings:
  duckdb_memory: "Auto-allocated based on query complexity"
  timeout: "30 seconds for complex queries"
  concurrent_access: "Multi-user safe with read locks"
```

### **SQL Views & Functions:**
```yaml
enterprise_views:
  Churn_Cox_Fusion: "Integrated churn and survival analysis"
  cox_survival_enhanced: "Multi-horizon survival probabilities"
  customer_risk_profile: "Risk categorization and business actions"
  cox_performance_summary: "Experiment performance aggregation"
  experiments_timebase: "Experiments with YYYYMM time functions"
  
utility_functions:
  yyyymm_to_year(): "Convert YYYYMM to year"
  yyyymm_to_month(): "Convert YYYYMM to month"
  yyyymm_diff_months(): "Calculate month difference"
  yyyymm_between(): "Check if YYYYMM in range"
```

## 🔗 **SYSTEM INTEGRATION**

### **Data Flow:**
```yaml
input_sources:
  - "Stage0 cache → rawdata table (via Outbox union import)"
  - "Churn pipeline → customer_churn_details, backtest_results"
  - "Cox pipeline → customer_cox_details, cox_survival"
  - "Counterfactuals → cf_* tables"
  
output_interfaces:
  - "DuckDB SQL queries → Business intelligence"
  - "Management Studio → Web-based analytics"
  - "Python API → Programmatic access"
  - "CLI tools → Administrative operations"
  
integration_patterns:
  foreign_keys: "id_experiments links all analysis tables"
  status_tracking: "experiments.status workflow management"
  temporal_validation: "LeakageGuard prevents data leakage"
```

### **Dependencies:**
```yaml
internal_dependencies:
  - "config/paths_config.py"
  - "config/data_dictionary_optimized.json"
  
external_dependencies:
  - "duckdb >= 0.8.0"
  - "pandas >= 1.3"
  - "tabulate >= 0.9"
  - "json-schema >= 4.0"
```

## 📈 **PERFORMANCE & MONITORING**

### **Current Performance (Production):**
```yaml
database_metrics:
  total_records: 919579
  active_tables: 20
  database_size: "587MB (optimized from 650MB+)"
  query_performance: "< 2s for complex analytical queries"
  
table_metrics:
  customer_churn_details: "4,742 records (22 columns)"
  customer_cox_details: "6,287 records (13 columns)"  
  cox_survival: "6,287 records with survival analysis"
  experiments: "10+ experiments tracked"
  
system_health:
  data_integrity: "100% referential integrity"
  backup_success_rate: "100%"
  concurrent_user_support: "Multi-user ready"
  memory_efficiency: "Columnar storage optimization"
```

### **SQL Query Performance:**
```yaml
query_benchmarks:
  simple_select: "< 50ms"
  complex_joins: "< 500ms"
  aggregations: "< 1s"
  datacube_analytics: "< 2s"
  
optimization_features:
  - "DuckDB query planner optimization"
  - "Columnar storage for analytics"
  - "In-memory processing"
  - "Automatic indexing on foreign keys"
```

## 🔧 **TROUBLESHOOTING FOR AI-AGENTS**

### **Common Issues:**
```yaml
database_corruption:
  symptom: "JSON parse errors or malformed data"
  solution: "Restore from backup in Trash potential/ directory"
  prevention: "Regular validation with db.validate_schema()"
  
sql_query_errors:
  symptom: "DuckDB execution failures"
  solution: "Check table existence with \\tables command"
  validation: "Verify foreign key relationships"
  
performance_issues:
  symptom: "Slow query execution > 5s"
  solution: "Check query complexity and add appropriate filters"
  optimization: "Use indexed columns (id_experiments, Kunde)"
  
memory_errors:
  symptom: "Out of memory during large queries"
  solution: "Use LIMIT clauses and batch processing"
  monitoring: "Monitor system memory during operations"
```

### **Data Integrity Checks:**
```python
# AI-Agent validation script
from bl.json_database.churn_json_database import ChurnJSONDatabase

db = ChurnJSONDatabase()
integrity_report = db.validate_schema()
table_counts = db.get_table_statistics()
foreign_key_integrity = db.validate_foreign_keys()
```

## 📊 **BUSINESS INTELLIGENCE EXAMPLES**

### **Executive Dashboard Queries:**
```sql
-- Customer Value Analysis
SELECT 
    COUNT(*) as total_customers,
    SUM(CUSTOMER_VALUE) as total_value,
    AVG(RF_PROB) as avg_churn_risk
FROM Churn_Cox_Fusion;

-- High-Risk Customer Identification  
SELECT Kunde, RF_PROB, CUSTOMER_VALUE, P_EVENT_6M
FROM Churn_Cox_Fusion 
WHERE RF_PROB > 0.5 AND CUSTOMER_VALUE > 15000
ORDER BY RF_PROB DESC, CUSTOMER_VALUE DESC;

-- Cox Performance Monitoring
SELECT 
    id_experiments,
    total_customers,
    high_risk_count,
    ROUND(high_risk_count * 100.0 / total_customers, 1) as high_risk_percentage
FROM cox_performance_summary
ORDER BY id_experiments;
```

### **Operational Analytics:**
```sql
-- Silent Churn Detection
SELECT 
    COUNT(*) as silent_churns,
    AVG(duration) as avg_survival_months
FROM cox_survival cs
WHERE cs.event = 1 
  AND cs.Kunde NOT IN (SELECT DISTINCT Kunde FROM customer_churn_details);

-- Experiment Performance Comparison
SELECT 
    e.experiment_id,
    e.experiment_name,
    COUNT(cd.Kunde) as churn_customers,
    COUNT(cox.Kunde) as cox_customers
FROM experiments e
LEFT JOIN customer_churn_details cd ON e.experiment_id = cd.id_experiments
LEFT JOIN customer_cox_details cox ON e.experiment_id = cox.id_experiments
GROUP BY e.experiment_id, e.experiment_name
ORDER BY e.experiment_id;
```

## 📋 **AI-AGENT MAINTENANCE CHECKLIST**

### **Daily Operations:**
```yaml
health_checks:
  - "Run: python bl/json_database/query_churn_database.py '\\tables'"
  - "Verify: All 20 tables accessible"
  - "Check: Foreign key integrity"
  - "Monitor: Query performance < 2s"
  
data_validation:
  - "Validate: No duplicate experiment records"
  - "Check: Customer record consistency"
  - "Verify: Status workflow integrity"
  - "Monitor: Database size growth"
```

### **After System Changes:**
```yaml
schema_updates:
  - "New tables → Update table count in this README"
  - "Schema changes → Update documentation"
  - "Performance changes → Update benchmarks"
  - "View changes → Update SQL examples"
  
integration_validation:
  - "Test: All pipelines write to correct tables"
  - "Verify: Foreign key relationships maintained"
  - "Check: SQL views return expected data"
  - "Validate: LeakageGuard prevents temporal violations"
```

### **Backup & Recovery:**
```yaml
backup_strategy:
  automatic_backup: "Before major schema changes"
  location: "Trash potential/churn_database_backup_<timestamp>.json"
  retention: "Keep last 5 backups"
  
recovery_procedure:
  1. "Stop all database connections"
  2. "Copy backup to churn_database.json"  
  3. "Restart applications"
  4. "Validate data integrity"
```

## 🧪 **VALIDATION**
- Interaktiv: `python bl/json_database/query_churn_database.py`, dann `\tables`, `\describe rawdata`, `\raw_profile`.
- DuckDB-Checks: `SELECT typeof(Kunde), typeof(I_TIMEBASE), typeof(I_Alive) FROM rawdata LIMIT 1;` (nur Ansicht, keine Datentypänderung). Erwartet: `I_Alive` ist INTEGER.

---

**📅 Last Updated:** 2025-09-21  
**🤖 Optimized for:** AI-Agent maintenance and usage  
**🎯 Status:** Production-ready core infrastructure  
**🔗 Related:** docs/SQL_QUERY_INTERFACE_GUIDE.md, docs/CHURN_DATABASE_SCHEMA.md
