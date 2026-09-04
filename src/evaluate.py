"""
1. Importing necessary libraries
2. Create a function(evaluate_pipeline) for evaluating the saved pipeline
3. Loading the saved pipeline and making predictions on the test set and evaluating the performances
"""

# =============================================
# SECTION 1: IMPORTING THE NECESSARY LIBRARIES
# =============================================
import joblib
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score
)
from src.pipeline import X_test, y_test
# ===================================================================================
# SECTION 2: Create a function(evaluate_pipeline) for evaluating the saved pipeline
# ===================================================================================
def evaluate_pipeline(pipeline, X_test, y_test):
    Threshold = 0.3
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= Threshold).astype(int)
    
    # Evauating the pipeline
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("ROC AUC SCORE:", roc_auc_score(y_test, proba))
    print("Classification Report:", classification_report(y_test, y_pred))

# ============================================================================================================
# SECTION 3: Loading the saved pipeline and making predictions on the test set and evaluating the performances
# ============================================================================================================
pipeline = joblib.load("models/xgb_pipeline.pkl")
pipeline_tuned = joblib.load("models/tuned_pipeline_model.pkl")

# Storing both pipeline in a list for evaluation
pipelines = [pipeline, pipeline_tuned]
names = [" Original Pipeline", " Tuned Pipeline"]

if __name__ == "__main__":
    for idx, pipe in enumerate (pipelines):
        print(f"Evaluating{names[idx]}:\n")
        evaluate_pipeline(pipe, X_test, y_test)
        print("\n" + "n"*50 + "\n")
    