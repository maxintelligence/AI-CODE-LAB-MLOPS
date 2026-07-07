"""
1. Create function (tune_hperparameters) for hyperparameter tuning using Optuna
Using objectives function and study.
2. Optimization of pipeline performance using the best hyperparameters obtained from the study.    
"""

# ==========================================================
# SECTION 1: Importing the necessary variables and libraries
# ==========================================================
from src.pipeline import (
    data,
    X_train, X_test, y_train, y_test,
    High_Skewed_Transformer, Robust_scaling, Categorical_encoding
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score
)
