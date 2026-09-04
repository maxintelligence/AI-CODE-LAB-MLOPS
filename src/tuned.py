"""
1. Create function (tune_hperparameters) for hyperparameter tuning using Optuna
Using objectives function and study.
2. Optimization of pipeline performance using the best hyperparameters obtained from the study.    
"""

# ==========================================================
# SECTION 1: Importing the necessary variables and libraries
# ==========================================================
from pathlib import Path
import joblib
from src.pipeline import (
    data,
    X_train, X_test, y_train, y_test,
    High_Skewed_Transformer, Robust_scaling, Categorical_encoding
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score
)

import optuna

# =============================================================================
# SECTION 2: Creating tuned_model function for optimization of hyperparameters
# =============================================================================

def tune_model(X, y):
    def objective(trial):
        params ={
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        pipeline_tuned = Pipeline(steps=[
            ("high_skewed", High_Skewed_Transformer),
            ("categorical", Categorical_encoding),
            ("robust", Robust_scaling),
            ("classifier", XGBClassifier(**params))
        ])
        
        # Using cross-validation to evaluate the models performance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(pipeline_tuned, X, y, cv=cv, scoring='recall', n_jobs=-1)
        return scores['test_score'].mean()
    # running optuna study to find the best hyperparamters
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    # printing and returning the best hyperparameter
    print("Best Hyperparameters:", study.best_params)
    return study.best_params
# =============================================================================================================================
# SECTION 3: Splitting data into feature matrix and target variable and getting the best optimized hyperparamters for recall
# =============================================================================================================================
X = data.drop(columns=["churn"], axis='columns')
y = data["churn"]

best_params=tune_model(X, y)

# ==================================================================================================
# SECTION 3: Optimizing pipeline performance using the best hyperparameters obtained from the study
# ===================================================================================================
# adding the scale_pos_weight parameter to the best_params dictionary for handling class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
best_params.update({'scale_pos_weight': scale_pos_weight,
                    'random_state': 1333,
                    "n_jobs": -1,
                    "eval_metric": "logloss"
                    
})

pipeline_tuned = Pipeline(steps=[
            ("high_skewed", High_Skewed_Transformer),
            ("categorical", Categorical_encoding),
            ("robust", Robust_scaling),
            ("classifier", XGBClassifier(**best_params))
        ])
# fitting the optimized pipeline to the training data
pipeline_tuned.fit(X_train, y_train)

# making predictions using the optimized pipeline
proba_tuned =pipeline_tuned.predict_proba(X_test)[:, 1]
THRESHOLD = 0.3
y_pred_tuned = (proba_tuned >= THRESHOLD).astype(int)

# Evaluating and saving the optimized pipeline
if __name__ == "__main__":
    print("Best Hyperparameters:", best_params)
    print("Balanced Accuracy Score for Tuned Model:", balanced_accuracy_score(y_test, y_pred_tuned))
    print("ROC AUC SCORE for Tuned Model:", roc_auc_score(y_test, proba_tuned))
    print("Classification Report for the Tuned Model:\n", classification_report(y_test, y_pred_tuned))
    model_path = Path("models/tuned_pipeline_model.pkl")
    joblib.dump(pipeline_tuned, model_path)
    print(f"Tuned pipeline saved to {model_path}")
