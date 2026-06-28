"""
1. Creating function for data validation
2. Schema Definition
3. Schema Validation
4. Validation Result    
"""
# Adding the project root to python's import path so modules can be imported
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing necessary libraries for data validation
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check
from pandera.errors import SchemaErrors
from src.preprocessed import clean_total_charges, load_data

# functions for validation
def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    print("Starting data validation with pandera...")
    failed_checks = []
    
    # Schema definition
    # cleaning the total charges column
    df = clean_total_charges(df)
    # renaming columns to match schema expectation
    df = df.rename(columns={"Churn": "churn"})
    # dropping duplicate charges if it exists
    if "total_charges" in df.columns:
        df = df.drop(columns=("total_charges"))
    
    schema = DataFrameSchema(
        {
            # Customer identifier
            "customerID": Column(str, nullable=False, unique=True),
            # Demographic features
            "gender": Column(str, checks=Check.isin(["Male","Female"])),
            "Partner": Column(str, checks=Check.isin(["Yes","No"])),
            "Dependents": Column(str, checks=Check.isin(["Yes","No"])),
            "SeniorCitizen": Column(int, checks=Check.isin([0,1])),
            # Service Features
            "PhoneService": Column(str, checks=Check.isin(["Yes", "No"])),
            "PaperlessBilling": Column(str, checks=Check.isin(["Yes", "No"])),
            "Contract": Column(str, checks=Check.isin(["Month-to-month", "One-year", "Two-year", "One year", "Two year"])),
            "MultipleLines": Column(str, checks=Check.isin(["Yes", "No", "No phone service"])),
            "InternetService": Column(str, checks=Check.isin(["DSL", "Fiber optic", "No"])),
            "OnlineSecurity": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "OnlineBackup": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "DeviceProtection": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "TechSupport": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "StreamingTV": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "StreamingMovies": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "PaymentMethod": Column(str, checks=Check.isin(["Electronic check", "Mail check","Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])),
            # Numerical Features
            "tenure": Column(int, nullable=False, checks=[Check.ge(0), Check.le(72)]),
            "MonthlyCharges": Column(float, nullable=False, checks=[Check.ge(0), Check.le(200)]),
            "TotalCharges": Column(float, nullable=True, checks=[Check.ge(0)]),
            "churn": Column(str, checks=Check.isin(["Yes", "No"]))
        },strict=True, coerce=True)
    
    # Schema Validation
    try:
        print("Validation Schema...")
        schema.validate(df, lazy=True)
    except SchemaErrors as e:
        # Column name is "check" not "checks" in newer pandera versions
        check_col = "check" if "check" in e.failure_cases.columns else "checks"
        failed_checks.extend(e.failure_cases["checks"].astype(str).unique().tolist())
        
    # BUSINESS RULE: TotalCharges >= MonthlyCharges
    print("Validating businesss rules")
    consistency_ratio = (df["TotalCharges"] >= df["MonthlyCharges"]).mean()
    if consistency_ratio < 0.95:
        failed_checks.append("TotalCharges should generally be greater than or equal to MonthlyCharges")
    
    # Validation result
    success = len(failed_checks) == 0
    if success:
        print("DATA VALIDATION PASSED")
    else:
        print("Data Validation Failed")
        print("\nFailed checks")
        for check in failed_checks:
            print(f"   o{check}")
        
    return success, failed_checks
    
if __name__ == "__main__":
    data = load_data("data/raw/Telco-Customer-Churn(1).csv")
    success, failed_checks = validate_data(data)