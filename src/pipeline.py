"""
1. importing the necessary libraries
2. loading data by importing load_data function from the preprocessed.py
3. create a custom transformer for high skewed numerical features using yeo-jonson transformation
4. create a custom transformer for selective one-hot encoding
5. create a custom transformer for robust scaling(Robustscaler)
6. create a function for data splitting and model training
7. complete preprocessing + model pipeline    
"""
# ==========================================================================================
# SECTION 1: Importing the necessary libraries
# =========================================================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocessed import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from .customcolumntransformer import CustomColumnTransformer
from sklearn.preprocessing import (
    
    OneHotEncoder,
    RobustScaler,
    PowerTransformer
)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')
# ===============================
# SECTION 2: Loading data
# ===============================
data = load_data("data/processed/Telco-Customer-Churn-Cleaned.csv")

# ===========================================================================================================
# SECTION 3: Creating a custom transformer for high skewed numerical feature
# ===========================================================================================================
High_Skewed_Transformer = CustomColumnTransformer(
    columns=["total_charges"],
    transformer=PowerTransformer(method="yeo-johnson")
)

# =========================================================================
# SECTION 4: Creating a custom transformer for selective one-hot encoding
# =========================================================================
Categorical_encoding = CustomColumnTransformer(
    columns=[
        "online_security", "online_backup", "device_protection", "tech_support",  "streaming_tv",
        "streaming_movies", "contract", "payment_method", "internet_service", "multiple_lines"],
    transformer=OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)

# =========================================================================
# SECTION 5: Creating a custom transformer for selective robust scaling
# =========================================================================
Robust_scaling = CustomColumnTransformer(
    columns=["monthly_charges", "tenure", "total_charges"],
    transformer=RobustScaler()
)

# =========================================================================
# SECTION 6: Creating functionn for data splitting and model training
# =========================================================================
def train_model(df: pd.DataFrame, target_col:str):
    X = df.drop(columns=[target_col], axis='columns')
    y = df[target_col]
    
    # splitting our data into training set and validation set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )
    # calculating the scale_pos_weight for the imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # creating an instance of XGBClassifier with scale_pos_weight
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    return X_train, X_test, y_train, y_test, xgb

# calling the train_model function to split the data and get the model
X_train, X_test, y_train, y_test, xgb_model = train_model(data, target_col="churn")


# =====================================================
# SECTION 7: complete preprocessing + model pipeline
# =====================================================
pipeline =Pipeline(steps=(
    ("high_skewed", High_Skewed_Transformer),
    ("categorical", Categorical_encoding),
    ("robust", Robust_scaling),
    ("classifier", xgb_model)
))

# fitting the pipeline to training data
pipeline.fit(X_train, y_train)

# making predictions
proba = pipeline.predict_proba(X_test)[:, 1]

# Setting the threshold to 0.3 to convert the probabilities to binary class predictions
THRESHOLD = 0.3
y_pred = (proba >= THRESHOLD).astype(int)

if __name__ == "__main__":
    # evaluating the model
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, proba))

    # saving the trained pipeline to a file
    model_path = Path("models/xgb_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Trained pipeline saved to {model_path}")