import os
import shutil
import zipfile
import logging
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, status, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings
from app.database.connection import get_db
from app.database.models import Dataset, AnalysisRun, AgentRunLog, Experiment, Report
from app.schemas.dataset import DatasetResponse, ProfileResponse
from app.schemas.analysis import AnalysisStartRequest, AnalysisRunResponse, AgentLogResponse
from app.schemas.model import ExperimentResponse, ReportResponse
from app.tools.data_tools import validate_csv, profile_dataframe
from app.agents.orchestrator import execute_analysis_workflow

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/datasets/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Ingests and validates a raw CSV dataset. Calculates rows, columns, 
    and checks if it fits basic security limits.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    # Sanitize filename to prevent path traversal
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
    if not safe_name or safe_name.startswith('.'):
        safe_name = "uploaded_dataset_" + safe_name.lstrip('.')
        if not safe_name.endswith(".csv"):
            safe_name += ".csv"
            
    dataset_path = os.path.join(settings.DATASETS_DIR, safe_name)
    
    # Save file of chunk stream to verify size limits
    total_size = 0
    try:
        with open(dataset_path, "wb") as buffer:
            while chunk := await file.read(8192):
                total_size += len(chunk)
                if total_size > settings.MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413, 
                        detail=f"Upload size exceeded. Max size allowed is {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
                    )
                buffer.write(chunk)
    except Exception as e:
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to write file stream: {str(e)}")
        
    # Validate CSV properties
    try:
        row_count, col_count, columns = validate_csv(dataset_path)
        df = pd.read_csv(dataset_path)
        profile = profile_dataframe(df)
    except Exception as csv_err:
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        raise HTTPException(status_code=400, detail=f"CSV Validation/Profiling failed: {str(csv_err)}")
        
    # Ingest record in DB
    db_dataset = Dataset(
        name=safe_name,
        filepath=dataset_path,
        file_size=total_size,
        row_count=row_count,
        column_count=col_count,
        columns_metadata=profile
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    return db_dataset

@router.post("/datasets/demo/{type}", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def load_demo_dataset(type: str, db: Session = Depends(get_db)):
    """
    Spawns sample classification or regression data directly from pre-generated CSV directories.
    """
    if type == "classification":
        filename = "churn_sample.csv"
    elif type == "regression":
        filename = "housing_sample.csv"
    else:
        raise HTTPException(status_code=400, detail="Unknown demo format. Options are: 'classification', 'regression'.")
        
    demo_path = os.path.join(settings.DATASETS_DIR, "sample", filename)
    if not os.path.exists(demo_path):
        raise HTTPException(
            status_code=404, 
            detail="Demo dataset not generated. Run data generation script in workspace."
        )
        
    # Copy dataset file from sample into datasets directory
    target_path = os.path.join(settings.DATASETS_DIR, filename)
    shutil.copyfile(demo_path, target_path)
    
    # Calculate stats
    row_count, col_count, columns = validate_csv(target_path)
    df = pd.read_csv(target_path)
    profile = profile_dataframe(df)
    
    # Write to database
    db_dataset = Dataset(
        name=filename,
        filepath=target_path,
        file_size=os.path.getsize(target_path),
        row_count=row_count,
        column_count=col_count,
        columns_metadata=profile
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    return db_dataset

@router.get("/datasets/{id}", response_model=DatasetResponse)
def get_dataset(id: str, db: Session = Depends(get_db)):
    db_dataset = db.query(Dataset).filter(Dataset.id == id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return db_dataset

@router.post("/analysis/start", response_model=AnalysisRunResponse, status_code=status.HTTP_201_CREATED)
def start_analysis(
    req: AnalysisStartRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """
    Submits a run objective and starts the multi-agent orchestrator 
    in a background worker thread.
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset metadata not found.")
        
    if not req.target_column or req.target_column.strip() == "":
        raise HTTPException(status_code=400, detail="Please select a target variable before starting the ML experiment.")
        
    # Create AnalysisRun model
    run = AnalysisRun(
        dataset_id=req.dataset_id,
        business_objective=req.business_objective,
        mode=req.mode,
        target_column=req.target_column,
        problem_type=req.problem_type,
        status="pending",
        current_stage="planner"
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    
    # Delegate orchestration loop to background worker thread
    background_tasks.add_task(execute_analysis_workflow, run.id, lambda: Session(bind=db.bind))
    
    return run

@router.get("/analysis/{id}/status", response_model=AnalysisRunResponse)
def get_run_status(id: str, db: Session = Depends(get_db)):
    run = db.query(AnalysisRun).filter(AnalysisRun.id == id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run reference not found.")
    return run

@router.get("/analysis/{id}/experiments", response_model=List[ExperimentResponse])
def get_run_experiments(id: str, db: Session = Depends(get_db)):
    experiments = db.query(Experiment).filter(Experiment.analysis_run_id == id).all()
    return experiments

@router.get("/analysis/{id}/results")
def get_run_results(id: str, db: Session = Depends(get_db)):
    """
    Retrieves comparative breakdown metrics plus details of the champion model.
    """
    run = db.query(AnalysisRun).filter(AnalysisRun.id == id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found.")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Workflow run is still processing or has failed.")
        
    best_exp = db.query(Experiment).filter(Experiment.id == run.best_experiment_id).first()
    experiments = db.query(Experiment).filter(Experiment.analysis_run_id == id).all()
    
    return {
        "run_id": id,
        "problem_type": run.problem_type,
        "target_column": run.target_column,
        "primary_metric": run.primary_metric,
        "champion": best_exp,
        "all_experiments": experiments
    }

@router.get("/analysis/{id}/report", response_model=ReportResponse)
def get_run_report(id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.analysis_run_id == id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Research report not compiled or run incomplete.")
    return report

@router.get("/analysis/{id}/download-report")
def download_report(id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.analysis_run_id == id).first()
    if not report or not report.report_path or not os.path.isfile(report.report_path) or os.path.getsize(report.report_path) == 0:
        raise HTTPException(status_code=404, detail="Report download not ready or invalid.")
    return FileResponse(
        path=report.report_path, 
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="research_report_{id}.html"'}
    )

@router.get("/models/{id}/download")
def download_model(id: str, db: Session = Depends(get_db)):
    """
    Retrieves the serialized joblib pipeline of a specific trained experiment.
    """
    exp = db.query(Experiment).filter(Experiment.id == id).first()
    if not exp or not exp.model_path or not os.path.isfile(exp.model_path) or os.path.getsize(exp.model_path) == 0:
        raise HTTPException(status_code=404, detail="Trained model binary not found or empty.")
    return FileResponse(
        path=exp.model_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="model_{exp.model_name}_{id}.joblib"'}
    )

@router.get("/models/{id}/model-card")
def download_model_card(id: str, db: Session = Depends(get_db)):
    """
    Generates and returns the PDF model card for the given experiment model ID.
    """
    exp = db.query(Experiment).filter(Experiment.id == id).first()
    if not exp or not exp.model_path:
        raise HTTPException(status_code=404, detail="Model Experiment not found.")
        
    model_dir = os.path.dirname(exp.model_path)
    pdf_path = os.path.join(model_dir, "model_card.pdf")
    
    try:
        from app.tools.pdf_generator import generate_model_card_pdf
        generate_model_card_pdf(id, db, pdf_path)
    except Exception as e:
        logger.exception("Failed to compile model card PDF:")
        raise HTTPException(status_code=500, detail=f"Failed to generate model card PDF: {str(e)}")
        
    if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) == 0:
        raise HTTPException(status_code=404, detail="Generated model card is empty or invalid.")
        
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="model_card_{exp.model_name}_{id}.pdf"'}
    )

@router.get("/analysis/{id}/download-all")
def download_all_artifacts(id: str, db: Session = Depends(get_db)):
    """
    Zips raw artifacts (EDA charts, report, preprocessor, model pipeline)
    for easy migration/download.
    """
    run = db.query(AnalysisRun).filter(AnalysisRun.id == id).first()
    if not run or run.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment run not successfully complete.")
        
    run_dir = os.path.join(settings.RUNS_DIR, id)
    report_obj = db.query(Report).filter(Report.analysis_run_id == id).first()
    
    os.makedirs(os.path.join(settings.BASE_DIR, "reports"), exist_ok=True)
    zip_path = os.path.join(settings.BASE_DIR, "reports", f"autonomous_ai_artifacts_{id}.zip")
    
    prefix = f"autonomous_ai_data_scientist_run_{id}/"
    
    try:
        # Generate champion model card on the fly if needed
        champion = db.query(Experiment).filter(Experiment.id == run.best_experiment_id).first()
        if champion and champion.model_path:
            champ_dir = os.path.dirname(champion.model_path)
            pdf_path = os.path.join(champ_dir, "model_card.pdf")
            try:
                from app.tools.pdf_generator import generate_model_card_pdf
                generate_model_card_pdf(champion.id, db, pdf_path)
            except Exception as p_err:
                logger.warning(f"Could not generate model card during ZIP packaging: {p_err}")
                
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. Add all EDA output images
            eda_folder = os.path.join(run_dir, "eda")
            if os.path.exists(eda_folder):
                for root, _, files in os.walk(eda_folder):
                    for file in files:
                        f_path = os.path.join(root, file)
                        if os.path.isfile(f_path) and os.path.getsize(f_path) > 0:
                            arcname = os.path.join(prefix, "eda", file)
                            # Convert to forward slash for zip uniformity
                            zipf.write(f_path, arcname.replace("\\", "/"))
                            
            # 2. Add champion model (.joblib) as best_model.joblib
            if champion and champion.model_path and os.path.isfile(champion.model_path):
                zipf.write(champion.model_path, (prefix + "models/best_model.joblib").replace("\\", "/"))
                            
            # 3. Add preprocessor (.joblib) as preprocessing_pipeline.joblib
            prep_file = os.path.join(run_dir, "preprocessing_pipeline.joblib")
            if os.path.exists(prep_file) and os.path.isfile(prep_file) and os.path.getsize(prep_file) > 0:
                zipf.write(prep_file, (prefix + "models/preprocessing_pipeline.joblib").replace("\\", "/"))
                
            # 4. Add final HTML report and Model Card PDF
            if report_obj and report_obj.report_path and os.path.isfile(report_obj.report_path) and os.path.getsize(report_obj.report_path) > 0:
                zipf.write(report_obj.report_path, (prefix + "report/research_report.html").replace("\\", "/"))
                
            if champion and champion.model_path:
                pdf_path = os.path.join(os.path.dirname(champion.model_path), "model_card.pdf")
                if os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0:
                    zipf.write(pdf_path, (prefix + "report/model_card.pdf").replace("\\", "/"))
                    
            # 5. Add metrics.json
            if champion and champion.metrics:
                import json
                metrics_data = json.dumps(champion.metrics, indent=2)
                zipf.writestr((prefix + "metrics/metrics.json").replace("\\", "/"), metrics_data)
                
            # 6. Add experiment_metadata.json
            all_exps = db.query(Experiment).filter(Experiment.analysis_run_id == id).all()
            metadata_list = []
            for ex in all_exps:
                metadata_list.append({
                    "id": ex.id,
                    "model_name": ex.model_name,
                    "overfitting_risk": ex.overfitting_risk,
                    "status": ex.status,
                    "metrics": ex.metrics,
                    "hyperparameters": ex.hyperparameters
                })
            zipf.writestr((prefix + "experiments/experiment_metadata.json").replace("\\", "/"), json.dumps(metadata_list, indent=2))
            
            # 7. Add explainability files (SHAP charts)
            if champion and champion.model_path:
                feat_chart = os.path.join(os.path.dirname(champion.model_path), "feature_importance.png")
                if os.path.isfile(feat_chart) and os.path.getsize(feat_chart) > 0:
                    zipf.write(feat_chart, (prefix + "explainability/feature_importance.png").replace("\\", "/"))
                    
            # 8. Add README.txt
            dataset_name = "N/A"
            if run.dataset_id:
                dataset_obj = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
                if dataset_obj:
                    dataset_name = dataset_obj.name
                    
            readme_content = f"""Autonomous AI Data Scientist Run {id} Output Artifacts
========================================================
Project Name: Autonomous AI Data Scientist Lab
Execution Timestamp: {datetime.utcnow().isoformat()} UTC
Business Objective: {run.business_objective or "General optimization"}
Selected Dataset: {dataset_name}
Selected Target: {run.target_column}
Problem Type: {run.problem_type}
Champion Model: {champion.model_name if champion else "None"}

To load python model machine artifact, execute:
import joblib
model = joblib.load("models/best_model.joblib")
"""
            zipf.writestr((prefix + "README.txt").replace("\\", "/"), readme_content)
            
        # ZIP Validation Checks
        if not os.path.exists(zip_path):
            raise ValueError("ZIP file was not created on disk.")
        if os.path.getsize(zip_path) == 0:
            raise ValueError("ZIP file is 0 bytes (corrupted/empty).")
            
        with zipfile.ZipFile(zip_path, 'r') as verify_zip:
            test_res = verify_zip.testzip()
            if test_res is not None:
                raise ValueError(f"ZIP verification failed: physical corruption in {test_res}")
                
            namelist = verify_zip.namelist()
            expected = [prefix + "README.txt"]
            if report_obj and report_obj.report_path and os.path.isfile(report_obj.report_path) and os.path.getsize(report_obj.report_path) > 0:
                expected.append(prefix + "report/research_report.html")
            for exp_f in expected:
                exp_zip_path = exp_f.replace("\\", "/")
                if exp_zip_path not in namelist:
                    raise ValueError(f"ZIP validation failed: missing expected file: {exp_zip_path}")
                    
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="autonomous_ai_artifacts_{id}.zip"'}
        )
    except Exception as e:
        logger.exception("Failed to compile artifacts zip:")
        raise HTTPException(status_code=500, detail=f"Failed to compile artifacts zip: {str(e)}")


@router.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow()}
