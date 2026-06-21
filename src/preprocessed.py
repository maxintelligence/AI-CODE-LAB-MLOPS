"""
1. Creating a function to load our data
2. Creating a function for data standardization
3. Creating a function to clean the total_charges column
4. Creating a function ton identify categorical column
5. Creating a function to identify numerical column
6. Creating a function tp perform binary encoding on binary column(i.e column with cardinality of two)
7. Creating a function to save the preprocessed data inside the processed subfolder in the data folder
"""


# Importing the necessary libraries for Data preprocessing
import pandas as pd
import numpy as np
import os

# function for loading raw data into a pandas dataframe
def load_data(file_path:str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File Not Found: {file_path}")
    return pd.read_csv(file_path)

# Functions for standardizing columns names and removing leading and trailing whitespaces and droppping the customer ID
def standardize_columns(df:pd.DataFrame) -> pd.DataFrame:
    standardize = {
    'customerID': 'customer_id',
    'gender': 'gender',
    'SeniorCitizen': 'senior_citizen',
    'Partner': 'partner',
    'Dependents': 'dependents',
    'tenure': 'tenure',
    'PhoneService': 'phone_service',
    'MultipleLines': 'multiple_lines',
    'InternetService': 'internet_service',
    'OnlineSecurity': 'online_security',
    'OnlineBackup': 'online_backup',
    'DeviceProtection': 'device_protection',
    'TechSupport': 'tech_support',
    'StreamingTV': 'streaming_tv',
    'StreamingMovies': 'streaming_movies',
    'Contract': 'contract',
    'PaperlessBilling': 'paperless_billing',
    'PaymentMethod': 'payment_method',
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges': 'total_charges',
    'Churn': 'churn'
    }
    
    #   Creating a copy of the original dataframe
    df = df.copy()
    # renaming the column names
    df.rename(columns=standardize, inplace = True)
    # dropping the customer ID
    df.drop(columns="customer_id", axis="columns", inplace=True)
    # remove the leading and trailing whitespaces from strings columns
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    return df

# function for cleaning and inputing the total charges column
def clean_total_charges(df:pd.DataFrame) -> pd.DataFrame:
    # creating a copy of the df
    df = df.copy()
    blank_values = df["total_charges"].astype(str).str.strip().eq("")
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors="coerce")
    # replace missing values with 0 for customers whose tenure is 0
    df.loc[(df['total_charges'].isna()) & (df['tenure'] == 0), 'total_charges'] = 0
    
    return df

# function for getting categorical columns in the data
def get_categorical_cols(df:pd.DataFrame, max_unique_values: int = 10) -> list[str]:
    categorical_cols = []
    for col in df.columns:
        if(
            pd.api.types.is_object_dtypes(df[col])
            or pd.api.types.is_strings_dtypes(df[col])
            or pd.api.types.is_categorical_dtypes(df[col])
            or df[col].nunique <= max_unique_values
        ):
            df[col] = df[col].astype("category")
            categorical_cols.append(col)
    return categorical_cols

# functiom for getting numerical columns in the data
def get_numerical_cols(df:pd.DataFrame) -> list[str]:
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return numerical_cols
        
             
