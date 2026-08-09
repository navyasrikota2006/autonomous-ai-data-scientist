import os
import pytest
import numpy as np
import pandas as pd
import zipfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database.models import Base, Dataset, AnalysisRun, Experiment, Report
from app.data_science.preprocessor import cast_to_string_array, DataPreprocessor
from app.tools.pdf_generator import generate_model_card_pdf
from app.main import app

# InMemory SQLite for testing database operations
DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def db_session():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_cast_to_string_array_mixed_types():
    """
    Ensures that mixed type inputs cast correctly to string arrays without errors,
    preventing scikit-learn OneHotEncoder string/float mixed errors.
    """
    mixed_data = np.array(["Male", 1.0, "Female", np.nan, 2.5], dtype=object)
    casted = cast_to_string_array(mixed_data)
    assert casted.shape == (5, 1)
    # Check that everything inside is converted to a string
    for val in casted.flatten():
        assert isinstance(val, str)

def test_data_preprocessor_mixed_type_categorical(tmp_path):
    """
    Ensures DataPreprocessor fit_transform is robust to categoricals with mixed types.
    """
    df = pd.DataFrame({
        "Numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "MixedCategorical": ["A", 2, "C", np.nan, "E", "A", 2, "C", np.nan, "E"],
        "Target": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    })
    
    preprocessor = DataPreprocessor(target_column="Target", problem_type="classification")
    X_train, X_val, y_train, y_val, feature_names = preprocessor.fit_transform(
        df=df,
        numerical_columns=["Numeric"],
        categorical_columns=["MixedCategorical"],
        id_columns=[]
    )
    assert X_train.shape[0] == 8 # 80% of 10 records
    assert X_train.shape[1] > 0
    
    # Check simple transform works
    val_trans = preprocessor.transform(df.iloc[:2])
    assert val_trans.shape == (2, X_train.shape[1])


def test_model_card_pdf_generation(db_session, tmp_path):
    """
    Ensures that PDF compiles to a valid size > 0 file on disk and embeds all tables.
    """
    # 1. Setup mock records
    mock_dataset = Dataset(
        id="test-dataset-id",
        name="test_data.csv",
        filepath="dummy_path.csv",
        file_size=1024,
        row_count=100,
        column_count=5,
        columns_metadata={
            "target_candidate": "target",
            "problem_type": "classification",
            "numerical_columns": ["age", "income"],
            "categorical_columns": ["contract"],
            "id_like_columns": [],
            "warnings": ["Column 'income' has outliers."]
        }
    )
    db_session.add(mock_dataset)
    db_session.commit()

    mock_run = AnalysisRun(
        id="test-run-id",
        dataset_id="test-dataset-id",
        business_objective="Optimize churn rates",
        mode="standard",
        target_column="target",
        problem_type="classification",
        primary_metric="f1",
        status="completed",
        best_experiment_id="test-exp-id"
    )
    db_session.add(mock_run)
    db_session.commit()

    # Generate dummy feature importance chart
    model_dir = os.path.join(tmp_path, "test-run-id", "models", "random_forest")
    os.makedirs(model_dir, exist_ok=True)
    chart_path = os.path.join(model_dir, "feature_importance.png")
    
    # Generate a real chart image using matplotlib to avoid PIL decoding errors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([1, 2], [3, 4])
    plt.savefig(chart_path)
    plt.close()

    mock_exp = Experiment(
        id="test-exp-id",
        analysis_run_id="test-run-id",
        model_name="random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 5},
        metrics={
            "train": {"accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1": 0.95},
            "val": {"accuracy": 0.88, "precision": 0.86, "recall": 0.90, "f1": 0.88}
        },
        cv_metrics=[0.88, 0.89, 0.87],
        overfitting_risk="low",
        status="success",
        features_used=["age", "income", "contract"],
        model_path=os.path.join(model_dir, "model.joblib")
    )
    db_session.add(mock_exp)
    db_session.commit()

    output_pdf = os.path.join(tmp_path, "model_card.pdf")
    generate_model_card_pdf(experiment_id="test-exp-id", db=db_session, output_path=output_pdf)
    
    assert os.path.exists(output_pdf)
    assert os.path.getsize(output_pdf) > 0

def test_zip_packaging_integrity(tmp_path):
    """
    Validates ZIP file logic, relative path formatting nested within prefix folder,
    and physical zip validation (testzip).
    """
    run_id = "mock-run-uuid"
    prefix = f"autonomous_ai_data_scientist_run_{run_id}/"
    zip_path = os.path.join(tmp_path, "test_check.zip")
    
    # 1. Create a dummy structure
    run_dir = os.path.join(tmp_path, run_id)
    os.makedirs(os.path.join(run_dir, "eda"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    
    # Add dummy report
    report_file = os.path.join(tmp_path, f"report_{run_id}.html")
    with open(report_file, "w") as f:
        f.write("<html>Report</html>")
    
    # Add dummy eda image
    eda_corr = os.path.join(run_dir, "eda", "correlation_heatmap.png")
    with open(eda_corr, "w") as f:
        f.write("EDA content")
        
    # Write details to zip
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(report_file, (prefix + "report/research_report.html"))
        zipf.write(eda_corr, (prefix + "eda/correlation_heatmap.png"))
        zipf.writestr(prefix + "README.txt", "Readme content info")
        
    # Test physical validation
    with zipfile.ZipFile(zip_path, 'r') as verify_zip:
        assert verify_zip.testzip() is None
        namelist = verify_zip.namelist()
        assert prefix + "report/research_report.html" in namelist
        assert prefix + "eda/correlation_heatmap.png" in namelist
        assert prefix + "README.txt" in namelist

def test_health_route():
    """
    Test standard client call.
    """
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_mixed_type_regression(tmp_path):
    """
    Verifies that a mixed-type regression pipeline handles numeric, categorical,
    and mixed-type columns, asserting target columns are drop-excluded and all features
    reaching estimators (Ridge, RandomForestRegressor, XGBRegressor) are float/numeric.
    """
    df = pd.DataFrame({
        "age": [25, 30, 45, 22, 35, 40, 50, 28, 38, 42],
        "income": [50000, 60000, 80000, 45000, 70000, 75000, 95000, 52000, 68000, 82000],
        "education": ["A", "B", "C", "A", "B", "C", "B", "A", "C", "B"],
        "region": ["North", "South", "East", "North", "South", "East", "South", "North", "East", "South"],
        "experience": [3, 5, 12, 1, 8, 10, 15, 4, 7, 11],
        "mixed": ["A", "B", 1, 2, None, "A", "B", 1, 2, None],
        "target": [150.0, 180.0, 250.0, 130.0, 210.0, 220.0, 290.0, 160.0, 200.0, 240.0]
    })

    dp = DataPreprocessor(target_column="target", problem_type="regression")
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=["age", "income", "experience"],
        categorical_columns=["education", "region", "mixed"],
        id_columns=[]
    )

    # Invariant: Target not in features check
    assert "target" not in features
    assert not any(col.lower() == "target" for col in dp.numerical_cols)
    assert not any(col.lower() == "target" for col in dp.categorical_cols)

    # Verify X_train has all float dtypes
    assert np.issubdtype(X_train.dtype, np.number)
    assert X_train.dtype != object

    # Train candidates
    from app.data_science.trainer import ModelTrainer
    trainer = ModelTrainer(problem_type="regression", primary_metric="r2", mode="quick")
    results = trainer.train_candidates(X_train, y_train, X_val, y_val)

    assert len(results) > 0
    success = [r for r in results if r["status"] == "success"]
    assert len(success) > 0, f"No model trained successfully: {[r.get('error_message') for r in results]}"


def test_categorical_regression(tmp_path):
    """
    Verifies that a dataset with purely categorical features can be fits and transformed for regression.
    """
    df = pd.DataFrame({
        "education": ["A", "B", "C", "A", "B", "C", "B", "A", "C", "B"],
        "region": ["North", "South", "East", "North", "South", "East", "South", "North", "East", "South"],
        "target": [10.0, 20.0, 30.0, 10.0, 25.0, 32.0, 22.0, 12.0, 28.0, 21.0]
    })
    dp = DataPreprocessor(target_column="target", problem_type="regression")
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=[],
        categorical_columns=["education", "region"],
        id_columns=[]
    )
    assert "target" not in features
    assert np.issubdtype(X_train.dtype, np.number)


def test_numeric_and_categorical_features(tmp_path):
    """
    Checks that numerical and categorical items are correctly classified inside DataPreprocessor
    and targets are removed.
    """
    df = pd.DataFrame({
        "num_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "cat_col": ["high", "low", "high", "low", "medium", "high", "low", "high", "low", "medium"],
        "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })
    dp = DataPreprocessor(target_column="target", problem_type="classification")
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=["num_col"],
        categorical_columns=["cat_col"],
        id_columns=[]
    )
    assert "num_col" in dp.numerical_cols
    assert "cat_col" in dp.categorical_cols
    assert "target" not in dp.numerical_cols
    assert "target" not in dp.categorical_cols


def test_target_not_in_features(tmp_path):
    """
    Validates case-insensitive target leakage checks.
    """
    df = pd.DataFrame({
        "Age": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "Target_Col": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })
    dp = DataPreprocessor(target_column="target_col", problem_type="classification")
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=["Age"],
        categorical_columns=[],
        id_columns=[]
    )
    assert "target_col" not in [f.lower() for f in features]
    assert "Target_Col" not in dp.numerical_cols


def test_holdout_metrics(tmp_path):
    """
    Verifies that real holdout metrics (MAE, MSE, RMSE, R2) are computed for regression estimators.
    """
    df = pd.DataFrame({
        "Age": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        "Score": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
    })
    dp = DataPreprocessor(target_column="Score", problem_type="regression")
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=["Age"],
        categorical_columns=[],
        id_columns=[]
    )
    from app.data_science.trainer import ModelTrainer
    trainer = ModelTrainer(problem_type="regression", primary_metric="r2", mode="quick")
    results = trainer.train_candidates(X_train, y_train, X_val, y_val)
    
    success = [r for r in results if r["status"] == "success"]
    assert len(success) > 0
    champ = success[0]
    
    metrics = champ["metrics"]
    assert "mae" in metrics["val"]
    assert "r2" in metrics["val"]
    assert isinstance(metrics["val"]["mae"], float)


def test_classification_pipeline(tmp_path):
    """
    Ensures classification fit_transform works fine and y is LabelEncoded.
    """
    df = pd.DataFrame({
        "Age": [25, 30, 45, 22, 35, 40, 50, 28, 38, 42],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month"],
        "Target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    })
    dp = DataPreprocessor(target_column="Target", problem_type="classification")
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=["Age"],
        categorical_columns=["Contract"],
        id_columns=[]
    )
    # Target values should be converted to 0/1 integers
    assert set(y_train).issubset({0, 1})
    assert np.issubdtype(X_train.dtype, np.number)


def test_preprocessor_serialization(tmp_path):
    """
    Regression test for serialization (pickling) of the data preprocessor and its pipeline.
    Ensures that the fitted DataPreprocessor and combined model pipeline can be joblib dumped and loaded.
    """
    import joblib
    # 1. Create a small mixed-type dataset
    df = pd.DataFrame({
        "Age": [25, 30, 45, 22, 35, 40, 50, 28, 38, 42],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month"],
        "MixedCol": ["A", 2, "C", np.nan, "E", "A", 2, "C", np.nan, "E"],
        "Target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    })
    
    # 2. Build the DataPreprocessor
    dp = DataPreprocessor(target_column="Target", problem_type="classification")
    
    # 3. Fit/transform the data
    X_train, X_val, y_train, y_val, features = dp.fit_transform(
        df=df,
        numerical_columns=["Age"],
        categorical_columns=["Contract", "MixedCol"],
        id_columns=[]
    )
    
    # 4. Serialize the resulting preprocessing pipeline using joblib.dump()
    preprocessor_path = os.path.join(tmp_path, "preprocessor.joblib")
    joblib.dump(dp, preprocessor_path)
    
    # Ensure raw pipeline within it can also be dumped (like combined pipeline does)
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    combined_pipeline = SklearnPipeline(steps=[
        ('preprocessor', dp.pipeline),
        ('estimator', clf)
    ])
    combined_path = os.path.join(tmp_path, "combined.joblib")
    joblib.dump(combined_pipeline, combined_path)
    
    # 5. Load it again using joblib.load()
    loaded_dp = joblib.load(preprocessor_path)
    loaded_combined = joblib.load(combined_path)
    
    # 6. Transform compatible data with the loaded pipeline
    test_df = pd.DataFrame({
        "Age": [28, np.nan],
        "Contract": ["One year", "Month-to-month"],
        "MixedCol": [1, np.nan]
    })
    
    transformed_dp = loaded_dp.transform(test_df)
    transformed_combined = loaded_combined.named_steps['preprocessor'].transform(test_df)
    
    # 7. Assert that serialization and transformation succeed.
    assert transformed_dp.shape == (2, X_train.shape[1])
    assert transformed_combined.shape == (2, X_train.shape[1])
    assert np.allclose(transformed_dp, transformed_combined)
    
    # Also assert that predicting through combined pipeline works
    preds = loaded_combined.predict(test_df)
    assert preds.shape == (2,)


