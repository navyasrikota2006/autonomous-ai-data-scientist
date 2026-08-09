import os
import pytest
import numpy as np
import pandas as pd
from app.data_science.preprocessor import DataPreprocessor
from app.data_science.trainer import ModelTrainer
from app.data_science.explainer import ModelExplainer
from app.agents.quality_critic import QualityCritic

@pytest.fixture
def synthetic_classification_data():
    np.random.seed(42)
    rows = 60
    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, size=rows),
        "Income": np.random.normal(50000, 15000, size=rows),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=rows),
        "Gender": np.random.choice(["Male", "Female"], size=rows),
        "Target": np.random.randint(0, 2, size=rows)
    })
    # Add a missing value to Income
    df.loc[10, "Income"] = np.nan
    return df

def test_full_ml_lifecycle(synthetic_classification_data, tmp_path):
    df = synthetic_classification_data
    target_col = "Target"
    
    # 1. Preprocessing
    preprocessor = DataPreprocessor(target_column=target_col, problem_type="classification")
    numerical_cols = ["Age", "Income"]
    categorical_cols = ["Contract", "Gender"]
    
    X_train, X_val, y_train, y_val, feature_names = preprocessor.fit_transform(
        df=df,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        id_columns=[]
    )
    
    # Dimensions assert
    assert X_train.shape[1] > 0
    assert len(feature_names) == 2 + 3 + 2 # Age + Income + 3 contract categories + 2 gender categories = 7
    
    # 2. HPO & Training
    trainer = ModelTrainer(problem_type="classification", primary_metric="f1", mode="quick")
    results = trainer.train_candidates(X_train, y_train, X_val, y_val)
    
    assert len(results) > 0
    success_results = [r for r in results if r["status"] == "success"]
    assert len(success_results) > 0
    
    champion_res = success_results[0]
    
    # 3. Model Explainability
    explainer = ModelExplainer(
        model=champion_res["model_instance"],
        feature_names=feature_names,
        problem_type="classification"
    )
    explanations = explainer.generate_explanations(X_train, str(tmp_path))
    assert "importance" in explanations
    assert len(explanations["importance"]) == len(feature_names)
    assert os.path.exists(explanations["chart_path"])
    
    # 4. Critic agent check
    critic_outcome = QualityCritic.inspect_model(
        model_name=champion_res["model_name"],
        problem_type="classification",
        primary_metric="f1",
        metrics=champion_res["metrics"],
        cv_metrics=champion_res["cv_metrics"]
    )
    assert critic_outcome.status in ["PASS", "WARNING", "RETRY"]
