"""
1. creating the CustomColumnTransformer class
2. Creating the constructor
3. Defining class attributes
4. Defining method(fit, transform)
"""
# Importing necessaries libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Creating the CustomColumnTransformer
class CustomColumnTransformer:
    def __init__(self, columns: list[str], transformer):
        self.columns = columns
        self.transformer = transformer
    # Creating the fit method
    def fit(self, X:pd.DataFrame, y = None):
        self.transformer.fit(X[self.columns])
        # obtaining output features name
        if hasatrr(self.transformer, "get_features_name_out"):
            self.feature_names_out_ = self.transformer.get_features_names_out(self.columns().tolist())
        else:
            self.features_names_out_ = self.columns
        return self
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X =X.copy()
        # Applying yeo-jonson transformation
        transformed = self.transformer.transform(X[self.columns])
        # converting transform output to Dataframe
        transformed_df =pd.DataFrame(transformed, columns=self.feature_names_out_, index=X.index)
        X =X.drop(columns=self.columns)
        # concatenating the transform columns with the remaining columns
        X = pd.concat([X, transformed_df])
        return X