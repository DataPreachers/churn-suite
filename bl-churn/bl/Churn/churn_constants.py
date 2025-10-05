#!/usr/bin/env python3
"""
Churn Random Forest Pipeline - Constants
========================================

Zentrale Konstanten für alle Churn-Module.
Definiert alle Magic Numbers und Konfigurationen für die neue modulare Churn-Pipeline.

Autor: AI Assistant
Datum: 2025-01-27
"""

# ==========================================
# MODEL CONFIGURATION
# ==========================================

# Random Forest Default Parameters
DEFAULT_RANDOM_FOREST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1
}

# Alternative Models Parameters
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0
}

LIGHTGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': -1
}

CATBOOST_PARAMS = {
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': False
}

# Cross-Validation Configuration
CV_FOLDS = 5
CV_RANDOM_STATE = 42
CV_SHUFFLE = True

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Rolling Window Configurations
ROLLING_WINDOWS = [6, 12, 18]  # Monate
DEFAULT_FEATURE_WINDOW = 6
DEFAULT_TREND_WINDOW = 12
DEFAULT_ACTIVITY_WINDOW = 18

# Feature Types
FEATURE_TYPES = {
    'ROLLING': 'rolling',
    'STATIC': 'static', 
    'ENHANCED': 'enhanced',
    'CATEGORICAL': 'categorical',
    'TARGET': 'target'
}

# Feature Categories
FEATURE_CATEGORIES = {
    'BUSINESS_ACTIVITY': 'business_activity',
    'DIGITALIZATION': 'digitalization',
    'FINANCIAL': 'financial',
    'DEMOGRAPHIC': 'demographic',
    'BEHAVIORAL': 'behavioral'
}

# Enhanced Features Configuration
ENHANCED_FEATURES = [
    'business_activity_enhanced',
    'digitalization_indicators',
    'activity_trend_6m',
    'activity_trend_12m',
    'rolling_variance_6m',
    'has_digitalization'
]

# Feature Selection Thresholds
MIN_FEATURE_VARIANCE = 0.01
MAX_FEATURE_CORRELATION = 0.95
MIN_FEATURE_IMPORTANCE = 0.001

# ==========================================
# SAMPLING STRATEGIES
# ==========================================

# Sampling Methods
SAMPLING_METHODS = {
    'NONE': 'none',
    'SMOTE': 'smote',
    'BORDERLINE_SMOTE': 'borderline_smote',
    'ADASYN': 'adasyn',
    'RANDOM_UNDER': 'random_under',
    'TOMEK_LINKS': 'tomek_links',
    'SMOTEENN': 'smoteenn',
    'SMOTETOMEK': 'smotetomek'
}

# Default Sampling Configuration
DEFAULT_SAMPLING_CONFIG = {
    'strategy': SAMPLING_METHODS['BORDERLINE_SMOTE'],
    'random_state': 42,
    'k_neighbors': 5
}

# ==========================================
# PERFORMANCE METRICS & TARGETS
# ==========================================

# Target Performance Metrics
TARGET_METRICS = {
    'auc': 0.99,
    'precision': 0.90,
    'recall': 0.95,
    'f1_score': 0.92
}

# Performance Thresholds
MINIMUM_AUC = 0.95
MINIMUM_PRECISION = 0.85
MINIMUM_RECALL = 0.90
MINIMUM_F1_SCORE = 0.87

# Threshold Optimization
THRESHOLD_OPTIMIZATION_METHODS = [
    'roc_optimal',
    'f1_optimal', 
    'precision_optimal',
    'percentile_75',
    'cost_benefit',
    'elbow'
]

DEFAULT_THRESHOLD = 0.5
MIN_RECALL_FOR_PRECISION_THRESHOLD = 0.5

# Cross-Validation Thresholds
CV_THRESHOLD_FOLDS = 5
BOOTSTRAP_ITERATIONS = 1000

# ==========================================
# BUSINESS METRICS
# ==========================================

# Customer Value Configuration
AVERAGE_CUSTOMER_VALUE = 2500.00
PREVENTION_COST_RATIO = 0.10
ROI_THRESHOLD = 3.0
PREVENTION_SUCCESS_RATE = 0.70

# Risk Categories
RISK_CATEGORIES = {
    'LOW': 'low',
    'MEDIUM': 'medium', 
    'HIGH': 'high'
}

# Risk Thresholds
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.6
HIGH_RISK_THRESHOLD = 0.8

# Business Impact Time Horizons
TIME_HORIZONS_MONTHS = [6, 12, 18, 24]
DEFAULT_TIME_HORIZON = 12

# ==========================================
# DATA HANDLING
# ==========================================

# Data Quality Thresholds
MAX_MISSING_VALUES_RATIO = 0.20
MIN_SAMPLES_PER_CLASS = 50
MAX_OUTLIER_RATIO = 0.05

# Data Types
NUMERIC_DTYPES = ['int64', 'float64']
CATEGORICAL_DTYPES = ['object', 'category']
BOOLEAN_DTYPES = ['bool']

# Data Validation
FLOAT64_TYPE = 'float64'
INT64_TYPE = 'int64'
OBJECT_TYPE = 'object'
BOOL_TYPE = 'bool'

# ==========================================
# TIMEBASE CONFIGURATION  
# ==========================================

# YYYYMM Format Configuration
TIMEBASE_FORMAT = 'YYYYMM'
TIMEBASE_REGEX = r'^\d{6}$'

# Default Time Periods
DEFAULT_TRAINING_MONTHS = 24
DEFAULT_BACKTEST_MONTHS = 6
MIN_TRAINING_MONTHS = 12

# Time Validation
MIN_TIMEBASE_VALUE = 200001
MAX_TIMEBASE_VALUE = 205012

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

# JSON Database Tables
CHURN_TABLES = [
    'churn_training_data',
    'churn_predictions', 
    'churn_feature_importance',
    'churn_business_metrics'
]

# Foreign Key Configuration
EXPERIMENT_FK_FIELD = 'id_experiments'
CUSTOMER_ID_FIELD = 'Kunde'

# Database Schema Validation
REQUIRED_EXPERIMENT_FIELDS = [
    'experiment_id',
    'experiment_name', 
    'model_type',
    'status',
    'training_from',
    'training_to'
]

# Experiment Status
EXPERIMENT_STATUS = {
    'CREATED': 'created',
    'PROCESSING': 'processing',
    'PROCESSED': 'processed',
    'FAILED': 'failed'
}

# ==========================================
# LOGGING & MONITORING
# ==========================================

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance Monitoring
MAX_EXECUTION_TIME_SECONDS = 1800  # 30 minutes
MEMORY_WARNING_THRESHOLD_GB = 8.0
CPU_WARNING_THRESHOLD_PERCENT = 90

# Progress Reporting
PROGRESS_REPORT_INTERVAL = 100  # Records
PERFORMANCE_REPORT_INTERVAL = 1000  # Records

# ==========================================
# FILE HANDLING
# ==========================================

# File Extensions
JSON_EXTENSION = '.json'
CSV_EXTENSION = '.csv'
PKL_EXTENSION = '.pkl'
JOBLIB_EXTENSION = '.joblib'

# Model Persistence
MODEL_SERIALIZATION_METHOD = 'joblib'
MODEL_COMPRESSION_LEVEL = 3

# File Operations
DEFAULT_ENCODING = 'utf-8'
CHUNK_SIZE = 10000
MAX_FILE_SIZE_MB = 500

# ==========================================
# ERROR HANDLING
# ==========================================

# Retry Configuration
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 1

# Error Tolerance
MAX_FAILED_RECORDS_RATIO = 0.05
CONTINUE_ON_ERROR = True

# Warning Thresholds
CORRELATION_WARNING_THRESHOLD = 0.9
VARIANCE_WARNING_THRESHOLD = 0.001
MISSING_DATA_WARNING_THRESHOLD = 0.1

# ==========================================
# MATHEMATICAL CONSTANTS
# ==========================================

# Numerical Constants
EPSILON = 1e-8
INFINITY_REPLACEMENT = 999999
NAN_REPLACEMENT = 0

# Probability Bounds
MIN_PROBABILITY = 0.0001
MAX_PROBABILITY = 0.9999

# Statistical Constants
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05
Z_SCORE_95_PERCENT = 1.96

# ==========================================
# VALIDATION RULES
# ==========================================

# Data Validation Rules
VALIDATION_RULES = {
    'min_samples': 1000,
    'min_features': 5,
    'max_features': 500,
    'min_positive_class_ratio': 0.01,
    'max_positive_class_ratio': 0.50
}

# Feature Validation
FEATURE_NAME_MAX_LENGTH = 100
FEATURE_NAME_PATTERN = r'^[a-zA-Z][a-zA-Z0-9_]*$'

# Experiment Validation  
EXPERIMENT_NAME_MAX_LENGTH = 200
EXPERIMENT_NAME_PATTERN = r'^[a-zA-Z0-9_-]+$'

# ==========================================
# INTEGRATION SETTINGS
# ==========================================

# Paths Integration
USE_PATHS_CONFIG = True
DYNAMIC_OUTPUTS_SUBDIR = 'churn_experiments'

# Data Dictionary Integration
USE_DATA_DICTIONARY = True
DATA_DICTIONARY_VALIDATION = True

# Stage0 Integration
STAGE0_CACHE_ENABLED = True
STAGE0_HASH_VALIDATION = True

# ==========================================
# MODULE VERSIONS
# ==========================================

# Module Versions for Tracking
MODULE_VERSIONS = {
    'churn_constants': '1.0.0',
    'churn_data_loader': '1.0.0',
    'churn_feature_engine': '1.0.0', 
    'churn_model_trainer': '1.0.0',
    'churn_evaluator': '1.0.0',
    'churn_working_main': '1.0.0',
    'churn_auto_processor': '1.0.0'
}

# Model Version Format
MODEL_VERSION_FORMAT = 'random_forest_v{major}.{minor}'
DEFAULT_MODEL_VERSION = 'random_forest_v1.0'

# ==========================================
# DEPRECATED/LEGACY MAPPING
# ==========================================

# Legacy Module Mapping (for migration reference)
LEGACY_MODULE_MAPPING = {
    'step1_input_handler.py': 'churn_data_loader.py',
    'enhanced_early_warning.py': 'churn_feature_engine.py',
    'step1_backtest_engine.py': 'churn_model_trainer.py',
    'step1_validation_engine.py': 'churn_evaluator.py',
    'step1_orchestrator.py': 'churn_working_main.py',
    'auto_rolling_window_pipeline.py': 'churn_auto_processor.py'
}

# Legacy Feature Names (for backward compatibility)
LEGACY_FEATURE_MAPPING = {
    'I_Alive': 'target',
    'customer_id': 'Kunde'
}
