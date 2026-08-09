import os
import sys
import logging
from sqlalchemy.orm import Session

# Configure python path to find app package
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from app.core.config import settings
from app.database.connection import engine, Base, SessionLocal
from app.database.models import Dataset, AnalysisRun, Experiment, Report
from app.agents.orchestrator import execute_analysis_workflow
from app.tools.data_tools import validate_csv, profile_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("E2E_Backend_Test")

def main():
    logger.info("Initializing database schemas on SQLite...")
    Base.metadata.create_all(bind=engine)
    
    db: Session = SessionLocal()
    
    # 1. Check if demo dataset is generated on disk
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(workspace_dir, "backend", "datasets", "sample", "churn_sample.csv")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Sample dataset not found at {dataset_path}. Run generate_sample_data.py first.")
        sys.exit(1)
        
    logger.info(f"Dataset verified at {dataset_path}")
    
    # 2. Register dataset in DB
    row_count, col_count, columns = validate_csv(dataset_path)
    
    # Let's read and profile it
    import pandas as pd
    df = pd.read_csv(dataset_path)
    profile = profile_dataframe(df, target_column="Churn")
    
    # Clean old datasets if exist to prevent duplicate key errors in demo
    db.query(Dataset).filter(Dataset.name == "churn_sample.csv").delete()
    db.commit()
    
    db_dataset = Dataset(
        name="churn_sample.csv",
        filepath=dataset_path,
        file_size=os.path.getsize(dataset_path),
        row_count=row_count,
        column_count=col_count,
        columns_metadata=profile
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    logger.info(f"Dataset registered with ID: {db_dataset.id}")
    
    # 3. Create AnalysisRun
    run = AnalysisRun(
        dataset_id=db_dataset.id,
        business_objective="Optimize subscriber retention by identifying churn trends",
        mode="quick", # Use Quick mode so it completes fast (5 Optuna trials)
        target_column="Churn",
        problem_type="classification",
        status="pending",
        current_stage="planner"
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    
    logger.info(f"Analysis scheduled. Run ID: {run.id}. Starting agent workflow...")
    
    # 4. Run Workflow Synchronously
    # We pass a lambda creating sessions to replicate how background processes create db scopes
    execute_analysis_workflow(run_id=run.id, db_session_factory=lambda: SessionLocal())
    
    # 5. Fetch updated run and verify outcomes
    db.refresh(run)
    logger.info(f"Workflow ended with Status: {run.status}")
    
    # Clear session caches to hit sqlite table updates
    db.expire_all()
    
    experiments = db.query(Experiment).filter(Experiment.analysis_run_id == run.id).all()
    report = db.query(Report).filter(Report.analysis_run_id == run.id).first()
    
    logger.info(f"Experiments trained: {len(experiments)}")
    for exp in experiments:
        logger.info(f"  - Model: {exp.model_name}, Status: {exp.status}, F1 score: {exp.metrics.get('val', {}).get('f1', 0):.4f}")
        
    if run.status != "completed":
        logger.error(f"E2E Run failed or is in state {run.status}! Check SQLite databases.")
        sys.exit(1)
        
    assert len(experiments) > 0, "No models were trained!"
    assert report is not None, "Report was not generated!"
    assert os.path.exists(report.report_path), f"HTML research report file does not exist at {report.report_path}!"
    assert run.best_experiment_id is not None, "Champion model id is missing!"
    
    best_exp = db.query(Experiment).filter(Experiment.id == run.best_experiment_id).first()
    assert os.path.exists(best_exp.model_path), f"Serialized joblib binary not found at {best_exp.model_path}!"
    
    logger.info("=========================================")
    logger.info("🎉 E2E BACKEND INTEGRATION SUCCESSFUL! 🎉")
    logger.info("=========================================")
    
    db.close()

if __name__ == "__main__":
    main()
