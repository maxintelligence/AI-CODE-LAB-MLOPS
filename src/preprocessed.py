"""
1. Creating a function to load our data
2. Creating a function for data standardization
3. Creating a function to clean total charge column
4. Creating a function to identify categorical column
5. Creating a function to identify numerical column
6. Creating a function to perform binary encoding on binary columns (columns with cardinality of 2)
7. Creating a function to save the preprocess data inside the processed sub-folder in the data folder
"""
# Importing Necessary Libraries For Data Preprocessing
import pandas as pd
import numpy as np
import os

# Function for loading raw data into a pandas dataframe
def load_data(file_path:str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

# Function for standardizing column names, removing leading and trailing whitespaces and dropping the customer_id
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
    # creating a copy of the original dataframe
    df = df.copy()
    # renaming the columns names
    df.rename(columns = standardize, inplace = True)
    # dropping the customer_id
    df.drop(columns="customer_id", axis="columns", inplace=True)
    # Removing the leading and trailing whitespaces from string columns
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
        
    return df

# Function for cleaning and imputing the total_charges column
def clean_total_charges(df:pd.DataFrame) -> pd.DataFrame:
    # creating a copy of df
    df = df.copy()
    # identifying blank values in the total_charges column
    blank_values = df["total_charges"].astype(str).str.strip().eq("")
    # converting invalid values to NaN
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors = "coerce")
    # replace missing values with 0 for customers whose tenure is zero
    df.loc[(df['total_charges'].isna()) & (df['tenure']==0), 'total_charges'] = 0
    
    return df

# Function for getting categorical columns in the data
def get_categorical_columns(df: pd.DataFrame, max_unique_values: int = 10) -> list[str]:
    categorical_cols = []
    for col in df.columns:
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or (df[col].nunique() <= max_unique_values and not pd.api.types.is_numeric_dtype(df[col]))
        ):
            df[col] = df[col].astype("category")
            categorical_cols.append(col)
    
    return categorical_cols

# Function for getting numerical columns in the data
def get_numeric_columns(df:pd.DataFrame) -> list[str]:
    numeric_cols = (df.select_dtypes(include=["int64", "float64"]).columns.tolist())
    return numeric_cols 

# Function for encoding binary categorical columns into numericl vlues
def encode_binary_columns(df):
    binary_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if df[col].nunique() == 2
    ]
    # Converting columns to string type and normalizing case
    df[binary_cols] = df[binary_cols].astype(str).apply(lambda x: x.str.lower())
    # Mapping dictionary for binary categories
    binary_mapping = {"yes": 1, "no": 0, "male": 1, "female": 0}
    # Replacing the categorical values with their numerical equivalents
    df[binary_cols] = df[binary_cols].replace(binary_mapping)
    # Converting encoded columns to integer type
    df[binary_cols] = df[binary_cols].astype(int)
    
    return df

# Function for saving the preprocessed to data to csv file 
def save_data(df:pd.DataFrame, file_path:str) -> None:
    # Ensuring that the parent directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Saving the Dataframe to csv withouth the index column
    df.to_csv(file_path, index = False)
    print(f"Data successfully saved to : {file_path}")

if __name__ == "__main__":    
    # Calling all the preprocessed for raw data preprocessing
    # Data Loading
    data = load_data("../data/raw/Telco-Customer-Churn(1).csv")
    # Column Standardization
    data = standardize_columns(data)
    # Cleaning the total charge column
    data = clean_total_charges(data)
    # Showcasing Categorical Columns
    cat_cols = get_categorical_columns(data)
    print(cat_cols)
    # Showcasing Numerical Columns
    num_cols = get_numeric_columns(data)
    print(num_cols)
    # Encoding Binary columns
    data = encode_binary_columns(data)
    # Saving data to csv
    save_data(data, "../data/processed/Telco-Customer-Churn-Cleaned.csv")