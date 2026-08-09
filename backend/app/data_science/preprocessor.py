import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from typing import Dict, Any, List, Tuple

def cast_to_string_array(X):
    import pandas as pd
    return pd.DataFrame(X).fillna("missing").astype(str).values

class DataPreprocessor:
    def __init__(self, target_column: str, problem_type: str):
        self.target_column = target_column
        self.problem_type = problem_type
        self.pipeline = None
        self.feature_names_out = []
        self.numerical_cols = []
        self.categorical_cols = []
        
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        numerical_columns: List[str], 
        categorical_columns: List[str],
        id_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Fits prep pipelines on train, returns preprocessed train and validation matrices,
        plus target vectors.
        """
        # Clean targets case-insensitively
        df = df.copy()
        target_lower = self.target_column.lower()
        actual_target_col = None
        for col in df.columns:
            if col.lower() == target_lower:
                actual_target_col = col
                break
                
        if actual_target_col:
            df = df.dropna(subset=[actual_target_col])
            y_series = df[actual_target_col]
        else:
            df = df.dropna(subset=[self.target_column])
            y_series = df[self.target_column]
            
        if self.problem_type == "regression":
            # Coerce target to numeric for regression
            y_numeric = pd.to_numeric(y_series, errors='coerce')
            valid_idx = y_numeric.notnull()
            df = df[valid_idx]
            y = y_numeric[valid_idx].values
        else:
            y = y_series.values
            
        if len(df) == 0:
            raise ValueError(f"No valid rows remaining after filtering missing or non-numeric target values for target column: '{self.target_column}'!")
        
        # Split features and target case-insensitively
        id_lowers = [c.lower() for c in id_columns]
        drop_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == target_lower or col_lower in id_lowers:
                drop_cols.append(col)
                
        X = df.drop(columns=drop_cols, errors='ignore')
        
        # Target leakage invariant check
        assert not any(col.lower() == target_lower for col in X.columns), f"Target column '{self.target_column}' leaked into features X!"
        
        # Robustly detect numerical and categorical columns from X only
        detected_num, detected_cat = [], []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                detected_num.append(col)
            else:
                # Try to coerce to numeric to support numeric values stored as strings
                non_nulls = X[col].dropna()
                if len(non_nulls) > 0:
                    coerced = pd.to_numeric(non_nulls, errors='coerce')
                    valid_num_pct = coerced.notnull().sum() / len(non_nulls)
                    if valid_num_pct > 0.8:
                        detected_num.append(col)
                    else:
                        detected_cat.append(col)
                else:
                    detected_cat.append(col)
                    
        self.numerical_cols = detected_num
        self.categorical_cols = detected_cat
        
        # Ensure numerical columns are coercion-converted to numeric, and categorical columns are uniformly strings
        X = X.copy()
        for col in self.numerical_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        for col in self.categorical_cols:
            X[col] = X[col].astype(str).fillna('missing')
            
        # 1. Stratified split for classification, standard split for regression
        stratify = y if self.problem_type == "classification" else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Fit LabelEncoder ONLY on training target if classification
        if self.problem_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            
            # Safe transform validation targets (mapping unseen values if any to -1)
            y_val_list = []
            for val in y_val:
                try:
                    y_val_list.append(self.label_encoder.transform([val])[0])
                except Exception:
                    y_val_list.append(-1)
            y_val = np.array(y_val_list)
            
            # Store labels mapping
            self.target_mapping = {int(i): str(c) for i, c in enumerate(self.label_encoder.classes_)}
        else:
            self.label_encoder = None
            self.target_mapping = None
        
        # 2. Build Pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('caster', FunctionTransformer(cast_to_string_array, validate=False)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64))
        ])
        
        transformers = []
        if self.numerical_cols:
            transformers.append(('num', num_transformer, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', cat_transformer, self.categorical_cols))
            
        self.pipeline = ColumnTransformer(transformers=transformers, remainder='drop')
        
        # 3. Transform
        X_train_trans = self.pipeline.fit_transform(X_train)
        X_val_trans = self.pipeline.transform(X_val)
        
        # Extract feature names for explainability
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols:
            try:
                onehot_encoder = self.pipeline.named_transformers_['cat'].named_steps['onehot']
                cat_names = list(onehot_encoder.get_feature_names_out(self.categorical_cols))
                feature_names.extend(cat_names)
            except Exception:
                # Fallback in case of mapping difficulties
                feature_names.extend([f"cat_{c}" for c in self.categorical_cols])
                
        self.feature_names_out = feature_names
        
        return X_train_trans, X_val_trans, y_train, y_val, feature_names
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.pipeline:
            raise ValueError("Preprocessor not yet fitted!")
        target_lower = self.target_column.lower()
        drop_cols = [c for c in df.columns if c.lower() == target_lower]
        X = df.drop(columns=drop_cols, errors='ignore')
        X = X.copy()
        for col in self.numerical_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('missing')
        return self.pipeline.transform(X)
        
    def decode_target(self, y: np.ndarray) -> np.ndarray:
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y)
        return y

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        return joblib.load(filepath)
