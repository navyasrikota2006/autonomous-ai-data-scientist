import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

def validate_csv(filepath: str) -> Tuple[int, int, List[str]]:
    """
    Validates that a file is a valid CSV and is not empty.
    Returns: (row_count, col_count, columns)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        # Read only a chunk first to check validity and speed up big files
        df_head = pd.read_csv(filepath, nrows=5)
        if df_head.empty:
            raise ValueError("The provided CSV file contains no data (empty).")
            
        # Count rows in a memory-efficient way
        row_count = 0
        for chunk in pd.read_csv(filepath, chunksize=10000):
            row_count += len(chunk)
            
        columns = list(df_head.columns)
        return row_count, len(columns), columns
        
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty or has invalid formatting.")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")

def profile_dataframe(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Profiles a pandas DataFrame and returns a detailed JSON report.
    """
    rows, cols = df.shape
    warnings = []
    
    # 1. Classify Columns
    numerical_columns = []
    categorical_columns = []
    id_like_columns = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)
        
        # ID-like columns (high cardinality, index-like)
        is_id_name = "id" in col.lower() or "key" in col.lower() or "code" in col.lower() or "hash" in col.lower()
        if unique_count == rows and col != target_column and (is_id_name or not ("int" in dtype or "float" in dtype)):
            id_like_columns.append(col)
            
        if "int" in dtype or "float" in dtype:
            # Low cardinality numeric could be categorical, but we treat numeric as numeric unless it has <= 5 unique values
            if rows > 10 and unique_count <= 5 and col != target_column:
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            categorical_columns.append(col)
            
    # 2. Target Column & Problem Type Auto-Detection
    if not target_column:
        # Default heuristics for target column
        target_keywords = ["churn", "target", "label", "price", "status", "class", "outcome", "revenue"]
        found_target = None
        for kw in target_keywords:
            for col in df.columns:
                if kw in col.lower():
                    found_target = col
                    break
            if found_target:
                break
        if not found_target:
            # Fall back to the last column
            found_target = df.columns[-1]
        target_column = found_target

    # Determine problem type based on target cardinality and type
    target_unique_val = df[target_column].nunique()
    target_dtype = df[target_column].dtype
    
    if "int" in str(target_dtype) or "float" in str(target_dtype):
        if target_unique_val <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
    else:
        problem_type = "classification"
        
    # Standard warning checks
    missing_columns = []
    column_statistics = {}
    
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = float(null_count / rows)
        unique_count = int(df[col].nunique())
        
        col_stats: Dict[str, Any] = {
            "type": "numerical" if col in numerical_columns else "categorical",
            "missing_count": null_count,
            "missing_pct": null_pct,
            "distinct_count": unique_count
        }
        
        if null_pct > 0.0:
            missing_columns.append(col)
            if null_pct > 0.5:
                warnings.append(f"Column '{col}' has high missing values ({null_pct:.1%}). Consider dropping.")
                
        if unique_count == 1:
            warnings.append(f"Column '{col}' is constant (only has 1 unique value) and will be removed.")
            
        # Calculate stats
        if col in numerical_columns:
            desc = df[col].describe()
            q1 = float(desc.get("25%", 0))
            q3 = float(desc.get("75%", 0))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlies = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            
            col_stats.update({
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
                "median": float(desc.get("50%", 0)),
                "outliers_count": outlies
            })
            if outlies > 0:
                warnings.append(f"Column '{col}' has {outlies} outliers based on IQR boundary.")
        else:
            top_value = df[col].mode().iloc[0] if not df[col].empty and not df[col].mode().empty else None
            top_freq = int(df[col].value_counts().iloc[0]) if not df[col].empty and len(df[col].value_counts()) > 0 else 0
            col_stats.update({
                "top_value": str(top_value),
                "top_frequency": top_freq,
                "top_percentage": float(top_freq / rows)
            })
            
        column_statistics[col] = col_stats
        
    # 3. Handle Correlations
    # We only compute Pearson correlation index for numerical features
    corr_df = df[numerical_columns].corr().fillna(0.0)
    correlations = corr_df.to_dict()
    
    # Check for target leakage
    if problem_type == "classification" and target_column in numerical_columns:
        for col in numerical_columns:
            if col != target_column:
                c_val = abs(correlations.get(col, {}).get(target_column, 0.0))
                if c_val > 0.95:
                    warnings.append(f"ALERT: Column '{col}' is extremely highly correlated with target '{target_column}' ({c_val:.2f}). Possible Target Leakage.")

    return {
        "rows": rows,
        "columns": cols,
        "target_candidate": target_column,
        "problem_type": problem_type,
        "missing_columns": missing_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "id_like_columns": id_like_columns,
        "warnings": warnings,
        "correlations": correlations,
        "column_statistics": column_statistics
    }
