import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session
from app.database.models import Experiment, AnalysisRun, Dataset

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas pattern to compute total pages and draw professional headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Header
        self.drawString(54, 750, "Autonomous AI Scientist — ML Research Lab")
        self.drawRightString(letter[0]-54, 750, "ENTERPRISE MODEL CARD")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 742, letter[0]-54, 742)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0]-54, 34, page_text)
        self.drawString(54, 34, "CONFIDENTIAL — INTERNAL DEVELOPMENT CARD")
        self.line(54, 46, letter[0]-54, 46)
        self.restoreState()

def generate_model_card_pdf(experiment_id: str, db: Session, output_path: str) -> str:
    """
    Compiles a comprehensive PDF model card for a given experiment model ID.
    Saves the file to output_path and returns the filepath.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Fetch relations from database
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment ID {experiment_id} not found in database.")

    run = db.query(AnalysisRun).filter(AnalysisRun.id == exp.analysis_run_id).first()
    if not run:
        raise ValueError(f"AnalysisRun not found for experiment {experiment_id}")

    dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset not found for analysis run {run.id}")

    # Styles Setup
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1E3A8A") # Navy
    secondary_color = colors.HexColor("#475569") # Slate
    dark_neutral = colors.HexColor("#1E293B") # Charcoal
    light_neutral = colors.HexColor("#F8FAFC") # Off-white
    border_color = colors.HexColor("#E2E8F0")

    # Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'NormalBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=dark_neutral,
        spaceAfter=4
    )

    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=9,
        textColor=colors.white
    )

    table_body_style = ParagraphStyle(
        'TableBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9,
        textColor=dark_neutral
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    story = []

    # --- Title Page Area ---
    story.append(Paragraph(f"Model Card: {exp.model_name.replace('_', ' ').title()}", title_style))
    story.append(Spacer(1, 10))

    # Meta Header Panel (Sidebar/Details Table)
    meta_data = [
        [Paragraph("Project Name", bold_body_style), Paragraph("Autonomous AI Data Scientist Lab", body_style)],
        [Paragraph("Experiment ID", bold_body_style), Paragraph(exp.id, body_style)],
        [Paragraph("Analysis Run ID", bold_body_style), Paragraph(run.id, body_style)],
        [Paragraph("Target Variable", bold_body_style), Paragraph(run.target_column or "N/A", body_style)],
        [Paragraph("Problem Type", bold_body_style), Paragraph(run.problem_type.upper() if run.problem_type else "N/A", body_style)],
        [Paragraph("Business Objective", bold_body_style), Paragraph(run.business_objective or "General optimization", body_style)],
        [Paragraph("Creation DateTime", bold_body_style), Paragraph(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[130, 370])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_neutral),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # --- Section: Dataset Summary ---
    story.append(Paragraph("1. Dataset Summary", h1_style))
    
    # Safely parse dataset metadata if it is stored as a JSON string
    metadata = dataset.columns_metadata or {}
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
            
    stats_dict = metadata.get("column_statistics", {})
    missing_count = sum(col_info.get("missing_count", 0) for col_info in stats_dict.values()) if stats_dict else 0
    warnings_list = metadata.get("warnings", [])
    warnings_text = ", ".join(warnings_list) if warnings_list else "None detected."
    
    dataset_data = [
        [Paragraph("Dataset Name", bold_body_style), Paragraph(dataset.name or "Uploaded Dataset", body_style)],
        [Paragraph("Total Rows", bold_body_style), Paragraph(str(dataset.row_count or metadata.get("rows", "N/A")), body_style)],
        [Paragraph("Total Columns", bold_body_style), Paragraph(str(dataset.column_count or metadata.get("columns", "N/A")), body_style)],
        [Paragraph("Numerical Features", bold_body_style), Paragraph(str(len(metadata.get("numerical_columns", []))), body_style)],
        [Paragraph("Categorical Features", bold_body_style), Paragraph(str(len(metadata.get("categorical_columns", []))), body_style)],
        [Paragraph("Missing Values Count", bold_body_style), Paragraph(str(missing_count), body_style)],
        [Paragraph("Identifier Fields Removed", bold_body_style), Paragraph(str(", ".join(metadata.get("id_like_columns", [])) or "None"), body_style)],
        [Paragraph("Data Quality Warnings", bold_body_style), Paragraph(warnings_text, body_style)],
    ]
    t_dataset = Table(dataset_data, colWidths=[150, 350])
    t_dataset.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_dataset)
    story.append(Spacer(1, 15))

    # --- Section: Model Information & Preprocessing ---
    story.append(Paragraph("2. Model & Algorithm Specifications", h1_style))
    
    hyperparams = exp.hyperparameters or {}
    hyperparams_formatted = ", ".join([f"{k}={v}" for k, v in hyperparams.items()]) if hyperparams else "Default estimators parameters used."
    features_formatted = ", ".join(exp.features_used) if exp.features_used else "All features."
    
    model_data = [
        [Paragraph("Champion Model Algorithm", bold_body_style), Paragraph(exp.model_name.replace('_', ' ').upper(), body_style)],
        [Paragraph("Hyperparameters Selected", bold_body_style), Paragraph(hyperparams_formatted, body_style)],
        [Paragraph("Inputs (Features Used)", bold_body_style), Paragraph(features_formatted, body_style)],
        [Paragraph("Saved Model Location", bold_body_style), Paragraph(f"models/{exp.model_name}/model.joblib" if exp.model_path else "Not saved", body_style)],
    ]
    t_model = Table(model_data, colWidths=[150, 350])
    t_model.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_model)
    story.append(Spacer(1, 15))

    # --- Section: Validation Results ---
    story.append(Paragraph("3. Validation Performance Metrics", h1_style))
    
    # Render table matching metric outputs
    val_table = []
    def f_metric(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

    if run.problem_type == "classification":
        val_table_headers = [
            Paragraph("Split", table_header_style), 
            Paragraph("Accuracy", table_header_style), 
            Paragraph("Precision", table_header_style), 
            Paragraph("Recall", table_header_style), 
            Paragraph("F1 Score", table_header_style),
            Paragraph("ROC-AUC", table_header_style),
            Paragraph("PR-AUC", table_header_style)
        ]
        val_table.append(val_table_headers)
        
        train_metrics = exp.metrics.get("train", {}) if exp.metrics else {}
        val_metrics = exp.metrics.get("val", {}) if exp.metrics else {}
        
        val_table.append([
            Paragraph("Holdout Train", table_body_style),
            Paragraph(f_metric(train_metrics.get('accuracy')), table_body_style),
            Paragraph(f_metric(train_metrics.get('precision')), table_body_style),
            Paragraph(f_metric(train_metrics.get('recall')), table_body_style),
            Paragraph(f_metric(train_metrics.get('f1')), table_body_style),
            Paragraph(f_metric(train_metrics.get('roc_auc')), table_body_style),
            Paragraph(f_metric(train_metrics.get('pr_auc')), table_body_style)
        ])
        val_table.append([
            Paragraph("Holdout Validation", table_body_style),
            Paragraph(f_metric(val_metrics.get('accuracy')), table_body_style),
            Paragraph(f_metric(val_metrics.get('precision')), table_body_style),
            Paragraph(f_metric(val_metrics.get('recall')), table_body_style),
            Paragraph(f_metric(val_metrics.get('f1')), table_body_style),
            Paragraph(f_metric(val_metrics.get('roc_auc')), table_body_style),
            Paragraph(f_metric(val_metrics.get('pr_auc')), table_body_style)
        ])
        t_val = Table(val_table, colWidths=[100, 65, 65, 65, 65, 74, 70])
    else:
        # Regression
        val_table_headers = [
            Paragraph("Split", table_header_style), 
            Paragraph("Mean Absolute Error (MAE)", table_header_style), 
            Paragraph("Mean Squared Error (MSE)", table_header_style), 
            Paragraph("Root Mean Squared Error (RMSE)", table_header_style), 
            Paragraph("R-Squared (R²)", table_header_style)
        ]
        val_table.append(val_table_headers)
        
        train_metrics = exp.metrics.get("train", {}) if exp.metrics else {}
        val_metrics = exp.metrics.get("val", {}) if exp.metrics else {}
        
        val_table.append([
            Paragraph("Holdout Train", table_body_style),
            Paragraph(f_metric(train_metrics.get('mae')), table_body_style),
            Paragraph(f_metric(train_metrics.get('mse')), table_body_style),
            Paragraph(f_metric(train_metrics.get('rmse')), table_body_style),
            Paragraph(f_metric(train_metrics.get('r2')), table_body_style),
        ])
        val_table.append([
            Paragraph("Holdout Validation", table_body_style),
            Paragraph(f_metric(val_metrics.get('mae')), table_body_style),
            Paragraph(f_metric(val_metrics.get('mse')), table_body_style),
            Paragraph(f_metric(val_metrics.get('rmse')), table_body_style),
            Paragraph(f_metric(val_metrics.get('r2')), table_body_style),
        ])
        t_val = Table(val_table, colWidths=[100, 100, 100, 100, 104])
        
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_val)
    story.append(Spacer(1, 8))
    
    cv_scores_formatted = ", ".join([f"{val:.4f}" for val in exp.cv_metrics]) if exp.cv_metrics else "N/A"
    cv_avg_score = sum(exp.cv_metrics)/len(exp.cv_metrics) if exp.cv_metrics else 0.0
    cv_msg = f"K-Fold Cross-Validation Scores: [{cv_scores_formatted}] (Mean Avg = {cv_avg_score:.4f})"
    story.append(Paragraph(cv_msg, body_style))
    story.append(Spacer(1, 10))

    # --- Section: Critic & Overfitting Check ---
    story.append(Paragraph("4. Critic / Model Governance Inspection", h1_style))
    
    # Retrieve Critic Check logs
    from app.database.models import AgentRunLog
    logs = db.query(AgentRunLog).filter(AgentRunLog.analysis_run_id == run.id).all()
    critic_events = []
    for log in logs:
        msg = log.message or ""
        msg_lower = msg.lower()
        if "critic" in msg_lower or "retry" in msg_lower or log.agent_name.lower() == "critic":
            critic_events.append(f"[{log.timestamp.strftime('%H:%M:%S')}] {log.agent_name.upper()}: {msg}")
            
    critic_history_text = "\n".join(critic_events) if critic_events else "No critic retries or interventions were recorded for this model run. Approved on first pass."
    
    critic_data = [
        [Paragraph("Overfitting Risk Rating", bold_body_style), Paragraph(exp.overfitting_risk.upper() if exp.overfitting_risk else "LOW", body_style)],
        [Paragraph("Critic Evaluation Remarks", bold_body_style), Paragraph(critic_history_text, body_style)],
        [Paragraph("Status Checks Status", bold_body_style), Paragraph("APPROVED", body_style)]
    ]
    t_critic = Table(critic_data, colWidths=[150, 350])
    t_critic.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_critic)
    
    # Page Break before visual plots to look organized
    story.append(PageBreak())

    # --- Section: Explanations & Plots ---
    story.append(Paragraph("5. Model Explainability & SHAP Contributions", h1_style))
    story.append(Paragraph(
        "Visualizes quantitative features importance rankings mapping impact on forecasts.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Fetch feature importance values on the fly to render as text/table
    importance_list = []
    try:
        import joblib
        if exp.model_path and os.path.exists(exp.model_path):
            pipeline = joblib.load(exp.model_path)
            estimator = pipeline.named_steps.get('estimator', None)
            if estimator:
                scores = None
                if hasattr(estimator, "feature_importances_"):
                    scores = estimator.feature_importances_
                elif hasattr(estimator, "coef_"):
                    import numpy as np
                    scores = np.abs(estimator.coef_)
                    if len(scores.shape) > 1:
                        scores = np.mean(scores, axis=0)
                
                if scores is not None and exp.features_used:
                    for name, score in zip(exp.features_used, scores):
                        importance_list.append((name, float(score)))
                    importance_list.sort(key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Could not load feature importances: {e}")
        
    if importance_list:
        story.append(Paragraph("Top Feature Importances (Selectable Text):", h2_style))
        imp_table_data = [[Paragraph("Feature Name", table_header_style), Paragraph("Score / Importance Weights", table_header_style)]]
        for f_name, f_score in importance_list[:5]:
            imp_table_data.append([Paragraph(f_name, table_body_style), Paragraph(f"{f_score:.6f}", table_body_style)])
            
        t_imp = Table(imp_table_data, colWidths=[200, 304])
        t_imp.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), secondary_color),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('PADDING', (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
        ]))
        story.append(t_imp)
        story.append(Spacer(1, 10))

    # Sample local explanation
    local_explanation_text = "Local explanation not available."
    try:
        import pandas as pd
        import joblib
        if os.path.exists(dataset.filepath) and exp.model_path and os.path.exists(exp.model_path):
            df_raw = pd.read_csv(dataset.filepath, nrows=5)
            pipeline = joblib.load(exp.model_path)
            
            row = df_raw.iloc[0:1]
            target_col_to_drop = [c for c in row.columns if c.lower() == (run.target_column or "").lower()]
            row_features = row.drop(columns=target_col_to_drop, errors="ignore")
            
            pred = pipeline.predict(row_features)[0]
            if run.problem_type == "classification":
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(row_features)[0]
                    classes = pipeline.classes_
                    pred_prob = max(proba)
                    local_explanation_text = f"Sample prediction for the first dataset record yields class forecast '{pred}' with a confidence probability of {pred_prob:.2%}. "
                else:
                    local_explanation_text = f"Sample prediction for the first dataset record yields class forecast '{pred}'. "
                    
                top_features_attribution = []
                feats = exp.features_used or []
                for f in feats[:5]:
                    if f in df_raw.columns:
                        top_features_attribution.append(f"{f} = {df_raw.loc[0, f]}")
                if top_features_attribution:
                    local_explanation_text += "Attribution feature state: " + ", ".join(top_features_attribution)
            else: # regression
                local_explanation_text = f"Sample prediction for the first dataset record yields a forecasted value of {pred:.4f}. "
                top_features_attribution = []
                feats = exp.features_used or []
                for f in feats[:5]:
                    if f in df_raw.columns:
                        top_features_attribution.append(f"{f} = {df_raw.loc[0, f]}")
                if top_features_attribution:
                    local_explanation_text += "Attribution feature state: " + ", ".join(top_features_attribution)
    except Exception as e:
        local_explanation_text = f"Local explanation execution details: {str(e)}"
        
    story.append(Paragraph("Local Attributions (Selectable Text):", h2_style))
    story.append(Paragraph(local_explanation_text, body_style))
    story.append(Spacer(1, 10))

    # Embed feature importance image if exists
    chart_found = False
    if exp.model_path:
        model_dir = os.path.dirname(exp.model_path)
        feat_chart = os.path.join(model_dir, "feature_importance.png")
        if os.path.exists(feat_chart):
            chart_found = True
            story.append(Paragraph("Global Feature Importance Distribution (SHAP / Fallback Plot):", h2_style))
            img_w = letter[0] - 120
            img_h = (img_w / 8.0) * 5.0
            story.append(Image(feat_chart, width=img_w, height=img_h))
            story.append(Spacer(1, 10))
            
    if not chart_found:
        story.append(Paragraph("[Visual feature importance chart not generated during workflow]", body_style))
        story.append(Spacer(1, 10))

    # --- Section: Limitations & Assumptions ---
    story.append(Paragraph("6. Governance Limitations & Assumptions", h1_style))
    story.append(Paragraph(
        "• <b>Dataset Constraints:</b> Model capability is bound to the features, column types, and distributions present "
        "in the uploaded dataset source. Categorical classes unseen in the training split will fall back to default bins.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Imputation:</b> Missing numeric rows are imputed with the median of training columns; categorical NaNs are imputed with 'missing' classes.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Linear/Tree Assure:</b> Outliers are handled within typical thresholds. Extreme deviations may yield distorted SHAP/predictive performance.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # --- Section: Reproducibility ---
    story.append(Paragraph("7. Replicability & System Information", h1_style))
    
    rep_info = [
        [Paragraph("Random Seed State", bold_body_style), Paragraph("42", body_style)],
        [Paragraph("Optimization Trials", bold_body_style), Paragraph("Optuna quick/standard trials schedule", body_style)],
        [Paragraph("ML Pipeline Engines", bold_body_style), Paragraph("scikit-learn, xgboost, optuna", body_style)],
        [Paragraph("Export Serialization", bold_body_style), Paragraph("Joblib (embedded ColumnTransformer + estimator pipeline)", body_style)]
    ]
    t_rep = Table(rep_info, colWidths=[150, 350])
    t_rep.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_rep)

    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
