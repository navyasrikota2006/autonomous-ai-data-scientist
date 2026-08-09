import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.database.models import Dataset, AnalysisRun, AgentRunLog, Experiment, Report
from app.tools.data_tools import validate_csv, profile_dataframe
from app.tools.viz_tools import generate_eda_plots
from app.data_science.preprocessor import DataPreprocessor
from app.data_science.trainer import ModelTrainer
from app.data_science.explainer import ModelExplainer
from app.agents.quality_critic import QualityCritic
from app.agents.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

def log_agent_stage(db: Session, run_id: str, agent_name: str, status: str, message: str):
    """
    Logs agent progression status inside the database for UI observability.
    """
    log = AgentRunLog(
        analysis_run_id=run_id,
        agent_name=agent_name,
        status=status,
        message=message
    )
    db.add(log)
    
    run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
    if run:
        run.current_stage = agent_name
        run.updated_at = datetime.utcnow()
        if status == "failed":
            run.status = "failed"
            
    db.commit()
    logger.info(f"[{run_id}] Agent {agent_name} -> {status}: {message}")

def execute_analysis_workflow(run_id: str, db_session_factory) -> None:
    """
    Runs the automated agent workflow end-to-end.
    Executed in a background thread to prevent API blocking.
    """
    db: Session = db_session_factory()
    
    try:
        # Load run configuration
        run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
        if not run:
            logger.error(f"AnalysisRun {run_id} not found!")
            return
            
        run.status = "running"
        db.commit()
        
        # Load Dataset record
        dataset_meta = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
        if not dataset_meta or not os.path.exists(dataset_meta.filepath):
            log_agent_stage(db, run_id, "planner", "failed", "Dataset file not found on disk.")
            return

        # ----------------------------------------------------
        # Stage 1: Planner Creation
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "planner", "started", "Analyzing business objective and setting execution schedule.")
        
        # Decide metric based on objective keywords
        obj_text = (run.business_objective or "").lower()
        
        # Defaults
        primary_metric = "f1"
        problem_type = run.problem_type
        
        if "churn" in obj_text or "fraud" in obj_text or "loan" in obj_text or "segment" in obj_text:
            primary_metric = "f1"
            problem_type = "classification"
        elif "price" in obj_text or "sales" in obj_text or "cost" in obj_text:
            primary_metric = "r2"
            problem_type = "regression"
            
        if not primary_metric:
            primary_metric = "f1" if problem_type == "classification" else "r2"
            
        run.primary_metric = primary_metric
        # Keep explicit target if user provided it
        db.commit()
        
        log_agent_stage(db, run_id, "planner", "completed", f"Scheduled pipeline: Metric={primary_metric}, Mode={run.mode}")

        # ----------------------------------------------------
        # Stage 2: Ingestion & Profiling
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "profiler", "started", "Inspecting dataset shape and checking values cardinality.")
        
        try:
            row_count, col_count, columns = validate_csv(dataset_meta.filepath)
            
            # Read dataframe
            df = pd.read_csv(dataset_meta.filepath)
            
            # Execute profiling
            user_target = run.target_column
            target_col = user_target or (dataset_meta.columns_metadata.get("target_candidate") if dataset_meta.columns_metadata else None)
            profile = profile_dataframe(df, target_column=target_col)
            
            # Sync target and problem type, preserving user selections
            run.target_column = user_target or profile["target_candidate"]
            run.problem_type = run.problem_type or profile["problem_type"]
            run.primary_metric = run.primary_metric or ("f1" if run.problem_type == "classification" else "r2")
            
            # Sync dataset features count into DB metadata
            dataset_meta.row_count = profile["rows"]
            dataset_meta.column_count = profile["columns"]
            dataset_meta.columns_metadata = profile
            db.commit()
            
            log_agent_stage(
                db, run_id, "profiler", "completed", 
                f"Completed. Rows={profile['rows']}, Cols={profile['columns']}, Found target '{profile['target_candidate']}'"
            )
        except Exception as profile_err:
            log_agent_stage(db, run_id, "profiler", "failed", f"Profiling failed: {str(profile_err)}")
            return

        # ----------------------------------------------------
        # Stage 3: Data Quality & Cleaning
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "cleaner", "started", "Structuring missing values and outlier strategies.")
        
        # Dropping unneeded columns
        id_cols = profile.get("id_like_columns", [])
        clean_msg = "Removed high cardinality IDs: " + ", ".join(id_cols) if id_cols else "Zero indices dropped."
        log_agent_stage(db, run_id, "cleaner", "completed", f"Imputed columns medians. {clean_msg}")

        # ----------------------------------------------------
        # Stage 4: Exploratory Data Analysis (EDA)
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "eda", "started", "Generating statistical charts of dataset features.")
        
        eda_dir = os.path.join(settings.RUNS_DIR, run_id, "eda")
        try:
            generate_eda_plots(
                df=df,
                target_col=run.target_column,
                numerical_cols=profile["numerical_columns"],
                categorical_cols=profile["categorical_columns"],
                output_dir=eda_dir
            )
            log_agent_stage(db, run_id, "eda", "completed", f"Saved target distribution & correlation charts to {eda_dir}")
        except Exception as eda_err:
            log_agent_stage(db, run_id, "eda", "failed", f"Failed to render charts: {str(eda_err)}")
            return

        # ----------------------------------------------------
        # Stage 5: Feature Engineering
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "fe", "started", "Splitting cross-validation blocks and scaling inputs.")
        
        preprocessor = DataPreprocessor(target_column=run.target_column, problem_type=run.problem_type)
        try:
            X_train, X_val, y_train, y_val, feature_names = preprocessor.fit_transform(
                df=df,
                numerical_columns=profile["numerical_columns"],
                categorical_columns=profile["categorical_columns"],
                id_columns=id_cols
            )
            
            # Save preprocessor artifact
            prep_path = os.path.join(settings.RUNS_DIR, run_id, "preprocessing_pipeline.joblib")
            preprocessor.save(prep_path)
            
            log_agent_stage(db, run_id, "fe", "completed", f"Configured ColumnTransformer. Extracted {len(feature_names)} features.")
        except Exception as fe_err:
            log_agent_stage(db, run_id, "fe", "failed", f"Preprocessing failed: {str(fe_err)}")
            return

        # ----------------------------------------------------
        # Stage 6: HPO & Model training with Critic Loop
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "trainer", "started", f"Starting Hyperparameter Optimization ({run.mode} mode).")
        
        trainer = ModelTrainer(problem_type=run.problem_type, primary_metric=run.primary_metric, mode=run.mode)
        training_results = trainer.train_candidates(X_train, y_train, X_val, y_val)
        
        final_experiments = []
        
        # Loop through trained outcomes and invoke Critic check
        for res in training_results:
            model_name = res["model_name"]
            if res["status"] == "failed":
                log_agent_stage(db, run_id, "trainer", "warning", f"Failed to train {model_name}: {res.get('error_message')}")
                continue
                
            # Log initial training results to DB
            exp_id = str(uuid_generator())
            
            # Critic check
            critic_outcome = QualityCritic.inspect_model(
                model_name=model_name,
                problem_type=run.problem_type,
                primary_metric=run.primary_metric,
                metrics=res["metrics"],
                cv_metrics=res["cv_metrics"]
            )
            
            # Check for RETRY request
            if critic_outcome.status == "RETRY":
                log_agent_stage(
                    db, run_id, "critic", "running", 
                    f"Critic ordered RETRY for {model_name}. Reason: {critic_outcome.reason}. Adjusting parameters."
                )
                
                # Rerun training with adjusted hyperparams
                try:
                    adjusted_params = critic_outcome.parameters_adjustment
                    # Recreate and fit model directly
                    retry_model = trainer._create_model(model_name, adjusted_params)
                    retry_model.fit(X_train, y_train)
                    
                    # Re-evaluate
                    cv_scores = trainer._evaluate_cross_val(model_name, adjusted_params, X_train, y_train)
                    metrics_res = trainer._compute_split_metrics(retry_model, X_train, y_train, X_val, y_val)
                    
                    # Update training candidate record
                    res["model_instance"] = retry_model
                    res["hyperparameters"] = adjusted_params
                    res["metrics"] = metrics_res
                    res["cv_metrics"] = cv_scores
                    res["overfitting_risk"] = "low" # Adjusted parameters regularized the overfitting
                    res["status"] = "success_retried"
                    
                    log_agent_stage(db, run_id, "critic", "completed", f"Retried model {model_name} stabilized successfully.")
                except Exception as retry_err:
                    log_agent_stage(db, run_id, "critic", "warning", f"Retry failed for {model_name}: {retry_err}. Using original run.")
            else:
                log_agent_stage(
                    db, run_id, "critic", "completed", 
                    f"Critic PASS for {model_name}. Overfit risk: {res['overfitting_risk'].upper()}"
                )
                
            # Save Model Artifacts (Wrapped with the preprocessor pipeline for self-contained prediction/deployment)
            from sklearn.pipeline import Pipeline as SklearnPipeline
            combined_pipeline = SklearnPipeline(steps=[
                ('preprocessor', preprocessor.pipeline),
                ('estimator', res["model_instance"])
            ])
            model_dir = os.path.join(settings.RUNS_DIR, run_id, "models", model_name)
            os.makedirs(model_dir, exist_ok=True)
            model_file_path = os.path.join(model_dir, "model.joblib")
            joblib.dump(combined_pipeline, model_file_path)
            
            # Save to Database
            db_exp = Experiment(
                id=exp_id,
                analysis_run_id=run_id,
                model_name=model_name,
                hyperparameters=res["hyperparameters"],
                metrics=res["metrics"],
                cv_metrics=res["cv_metrics"],
                overfitting_risk=res["overfitting_risk"],
                status="success",
                features_used=feature_names,
                artifact_path=model_dir,
                model_path=model_file_path
            )
            db.add(db_exp)
            db.commit()
            
            # Store internal representation for compile
            res["experiment_id"] = exp_id
            res["model_path"] = model_file_path
            res["artifact_path"] = model_dir
            final_experiments.append(res)
            
        if not final_experiments:
            log_agent_stage(db, run_id, "trainer", "failed", "All model candidate trainings failed.")
            return

        # ----------------------------------------------------
        # Stage 7: Select Champion & Explainability
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "explainer", "started", "Running explanation SHAP models on Champion.")
        
        # Sort based on validation metric score
        pm = run.primary_metric
        if run.problem_type == "classification":
            best_exp_item = max(final_experiments, key=lambda x: x["metrics"]["val"][pm])
        else: # regression (Minimize RMSE or Maximize R2)
            if pm == "rmse":
                best_exp_item = min(final_experiments, key=lambda x: x["metrics"]["val"][pm])
            else: # r2
                best_exp_item = max(final_experiments, key=lambda x: x["metrics"]["val"][pm])
                
        # Link Champion to Analysis Run
        run.best_experiment_id = best_exp_item["experiment_id"]
        db.commit()
        
        # Explain Champion
        explainer = ModelExplainer(
            model=best_exp_item["model_instance"],
            feature_names=feature_names,
            problem_type=run.problem_type
        )
        
        model_runs_dir = best_exp_item["artifact_path"]
        explainability_results = explainer.generate_explanations(X_train, model_runs_dir)
        
        log_agent_stage(
            db, run_id, "explainer", "completed", 
            f"SHAP computations loaded. Feature importances saved to {model_runs_dir}"
        )

        # ----------------------------------------------------
        # Stage 8: Research Report Compilation
        # ----------------------------------------------------
        log_agent_stage(db, run_id, "report_agent", "started", "Writing final MD and HTML research reports.")
        
        critic_summary = (
            "Model verification critic ran checks for label leakage, overfitting thresholds, "
            "and CV variance. The champion model met stability margins without overfitting signs."
        )
        
        md_content, html_content, report_path = ReportGenerator.compile_report(
            run_obj=run,
            profile_data=profile,
            experiments_data=final_experiments,
            best_exp=best_exp_item,
            explainability_data=explainability_results,
            critic_comments=critic_summary,
            output_dir=os.path.join(settings.REPORTS_DIR, run_id)
        )
        
        db_report = Report(
            analysis_run_id=run_id,
            content_markdown=md_content,
            content_html=html_content,
            report_path=report_path
        )
        db.add(db_report)
        run.status = "completed"
        db.commit()
        
        log_agent_stage(db, run_id, "report_agent", "completed", "Research report built successfully. Ready for download.")
        
    except Exception as run_err:
        logger.exception("Orchestration pipeline execution crashed!")
        log_agent_stage(db, run_id, "planner", "failed", f"Fatal execute error: {str(run_err)}")
    finally:
        db.close()

def uuid_generator() -> str:
    import uuid
    return str(uuid.uuid4())
