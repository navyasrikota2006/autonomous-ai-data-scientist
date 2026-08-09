import os
import pytest
import pandas as pd
import numpy as np
from app.tools.data_tools import validate_csv, profile_dataframe

def test_validate_csv(tmp_path):
    # Test valid CSV
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    csv_file = tmp_path / "valid.csv"
    df.to_csv(csv_file, index=False)
    
    rows, cols, col_names = validate_csv(str(csv_file))
    assert rows == 3
    assert cols == 2
    assert col_names == ["A", "B"]

    # Test empty CSV
    empty_file = tmp_path / "empty.csv"
    with open(empty_file, "w") as f:
        f.write("")
    with pytest.raises(ValueError, match="empty"):
        validate_csv(str(empty_file))

def test_profile_dataframe():
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Age": [23, 45, 12, 34, 56],
        "Score": [90.5, np.nan, 80.0, 75.5, 60.1],
        "Group": ["A", "B", "A", "B", "A"],
        "Target": [1, 0, 1, 0, 1]
    })
    
    profile = profile_dataframe(df, target_column="Target")
    assert profile["rows"] == 5
    assert profile["columns"] == 5
    assert profile["target_candidate"] == "Target"
    assert profile["problem_type"] == "classification"
    assert "Score" in profile["missing_columns"]
    assert "Age" in profile["numerical_columns"]
    assert "Group" in profile["categorical_columns"]
    assert "ID" in profile["id_like_columns"]
