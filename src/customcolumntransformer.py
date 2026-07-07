"""
1. creating the CustomColumnTransformer class
2. Creating the constructor
3. Defining class attributes
4. Defining method(fit, transform)
"""
# Importing necessaries libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Creating the CustomColumnTransformer Class
import scipy.sparse
class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], transformer):
        self.columns = columns
        self.transformer = transformer
    # Creating the fit method
    def fit(self, X:pd.DataFrame, y = None):
        self.transformer.fit(X[self.columns])
        # obtaining output features name
        if hasattr(self.transformer, "get_feature_names_out"):
            self.feature_names_out_ = self.transformer.get_feature_names_out(self.columns)
        else:
            self.feature_names_out_ = self.columns
        return self
    
    # Creating the transform method
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Applying yeo-jonson transformation
        transformed = self.transformer.transform(X[self.columns])
        # converting sparse matrix to dense array only if applicable
        if isinstance(transformed, scipy.sparse.spmatrix):
            transformed = transformed.toarray()
            
        if transformed.shape[1] != len(self.feature_names_out_):
            raise ValueError(
                f"Transformer returned {transformed.shape[1]} columns"
                f" but feature_names_out_ has {len(self.feature_names_out_)} names."
            )
       
       
        transformed_df =pd.DataFrame(transformed, columns=self.feature_names_out_, index=X.index)
        X =X.drop(columns=self.columns)
        # concatenating the transform columns with the remaining columns
        X = pd.concat([X, transformed_df], axis=1)
        return X