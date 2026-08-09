import os
import base64
import logging
from typing import Dict, Any, List, Tuple
from jinja2 import Template
from app.core.config import settings

logger = logging.getLogger(__name__)

class ReportGenerator:
    @staticmethod
    def compile_report(
        run_obj: Any,
        profile_data: Dict[str, Any],
        experiments_data: List[Dict[str, Any]],
        best_exp: Dict[str, Any],
        explainability_data: Dict[str, Any],
        critic_comments: str,
        output_dir: str
    ) -> Tuple[str, str, str]:
        """
        Uses data inputs to construct Markdown and HTML reports.
        Returns: (markdown_content, html_content, filepath)
        """
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "research_report.html")
        
        # 1. Prepare Base64 Charts to Embed Inline (if available)
        charts_b64 = {}
        for plot_key in ["target_distribution", "correlation_heatmap", "outliers_boxplots", "relationships"]:
            # Check experiment run directories
            p = os.path.join(output_dir, f"{plot_key}.png")
            if not os.path.exists(p):
                # Fallback to general output_dir
                p = os.path.join(settings.BASE_DIR, "mlruns", run_obj.id, "eda", f"{plot_key}.png")
                # Or correlation_heatmap / target_distribution / outliers_boxplots / relationships_countplot
                if plot_key == "relationships":
                    # Check scatter or countplot
                    test_p1 = os.path.join(settings.BASE_DIR, "mlruns", run_obj.id, "eda", "relationships_countplot.png")
                    test_p2 = os.path.join(settings.BASE_DIR, "mlruns", run_obj.id, "eda", "relationships_scatterplot.png")
                    p = test_p1 if os.path.exists(test_p1) else test_p2
                    
            if os.path.exists(p):
                try:
                    with open(p, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        charts_b64[plot_key] = f"data:image/png;base64,{encoded_string}"
                except Exception as e:
                    logger.warning(f"Could not encode EDA chart {plot_key}: {e}")
                    
        # Feature Importance Chart embedding
        shap_path = os.path.join(settings.BASE_DIR, "mlruns", run_obj.id, "models", best_exp["model_name"], "feature_importance.png")
        if os.path.exists(shap_path):
            try:
                with open(shap_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    charts_b64["explain_importance"] = f"data:image/png;base64,{encoded_string}"
            except Exception as e:
                logger.warning(f"Could not encode explain feature importance: {e}")

        # 2. Compile Markdown Content
        markdown_template = """# ML Research Report: {{ run.business_objective or 'Autonomous Model Analysis' }}
**Experiment Run ID:** `{{ run.id }}`  
**Date:** {{ created_at }}  
**Methodology:** Autonomous Multi-Agent Machine Learning Pipeline

---

## 1. Executive Summary
This research report presents an end-to-end machine learning study executing on the ingested dataset. The target variable selected was **{{ run.target_column }}**, executing a **{{ run.problem_type.upper() }}** problem format. Based on statistical evaluations, a final optimized model was selected using Cross-Validation metrics.

## 2. Dataset Overview
- **Total Rows Ingested:** {{ profile.rows }}
- **Total Columns Ingested:** {{ profile.columns }}
- **Problem Type:** {{ run.problem_type }}
- **Numeric Features Identified:** {{ profile.numerical_columns | length }} ($${{ profile.numerical_columns | join(', ') }}$$)
- **Categorical Features Identified:** {{ profile.categorical_columns | length }} ($${{ profile.categorical_columns | join(', ') }}$$)
- **Identified ID/Key Columns:** {{ profile.id_like_columns | join(', ') or 'None' }}

## 3. Data Quality & Preprocessing Decisions
The data critic applied the following imputations:
- Missing values in numerical variables were imputed using the median to protect against outliers.
- Categorical features were missing value imputed with a constant representation and encoded using a One-Hot transformer.
- Constant variables (if any) and ID-like variables were dropped to avoid overfitting or leakage.

## 4. Models Evaluation Summary
Below is a comparative breakdown of all candidate models trained using Stratified/Standard Cross-Validation and Optuna parameter tuning:

| Model Candidate | CV Mean ({{ run.primary_metric }}) | Val Split ({{ run.primary_metric }}) | Overfitting Risk | Status |
|---|---|---|---|---|
{% for exp in trials %}
| {{ exp.model_name }} | {{ exp.cv_mean | round(4) }} | {{ exp.val_score | round(4) }} | {{ exp.overfitting_risk }} | {{ exp.status }} |
{% endfor %}

### Final Selected Model
- **Best Model:** `{{ best.model_name }}`
- **Hyperparameters:** `{{ best.hyperparameters | tojson }}`
- **Evaluation Score:** {{ best.metrics.val[run.primary_metric] | round(4) }}

## 5. Model Explainability & Feature Contribution
Based on Shapley Additive exPlanations (SHAP) or Permutation Importance weights, the following features had the largest global contribution influencing model outcomes:

{% for name, score in importances.items() %}
- **{{ name }}:** {{ score | round(5) }}
{% endfor %}

---
*Report generated automatically by the Autonomous AI Data Scientist Platform.*
"""

        # Render Markdown
        import datetime
        from jinja2 import Environment
        env = Environment()
        env.filters['tojson'] = lambda d: d if isinstance(d, str) else str(d)
        
        md_tmpl = env.from_string(markdown_template)
        
        # Format candidate data for template
        trials_list = []
        for exp in experiments_data:
            cv_scores = exp.get("cv_metrics", [])
            cv_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
            val_score = exp.get("metrics", {}).get("val", {}).get(run_obj.primary_metric, 0.0)
            trials_list.append({
                "model_name": exp["model_name"],
                "cv_mean": cv_mean,
                "val_score": val_score,
                "overfitting_risk": exp.get("overfitting_risk", "low"),
                "status": exp["status"]
            })
            
        md_content = md_tmpl.render(
            run=run_obj,
            profile=profile_data,
            trials=trials_list,
            best=best_exp,
            importances=dict(list(explainability_data.get("importance", {}).items())[:10]),
            created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 3. Compile HTML Content (Embedded Styling)
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Research Report - {{ run.id }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #2D3748;
            line-height: 1.6;
            margin: 0;
            padding: 40px;
            background: #F7FAFC;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        h1 {
            color: #1A365D;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 12px;
            margin-top: 0;
        }
        h2 {
            color: #2B6CB0;
            margin-top: 30px;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 8px;
        }
        h3 {
            color: #2D3748;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #E2E8F0;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #F7FAFC;
            color: #4A5568;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .badge-success { background-color: #C6F6D5; color: #22543D; }
        .badge-warning { background-color: #FEFCBF; color: #744210; }
        .badge-danger { background-color: #FED7D7; color: #742A2A; }
        .chart-container {
            text-align: center;
            margin: 30px 0;
        }
        .chart-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        }
        .alert-box {
            background-color: #EBF8FF;
            border-left: 4px solid #3182CE;
            color: #2B6CB0;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .footer {
            margin-top: 50px;
            border-top: 1px solid #E2E8F0;
            padding-top: 20px;
            font-size: 0.85em;
            color: #A0AEC0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Autonomous ML Research Report</h1>
        <p><strong>Run Reference:</strong> <code>{{ run.id }}</code> | <strong>Date:</strong> {{ date }}</p>
        
        <div class="alert-box">
            <strong>Business Objective:</strong> {{ run.business_objective or "None specified." }}
        </div>
        
        <h2>1. Executive Summary</h2>
        <p>
            An autonomous multi-agent machine learning execution run was requested for dataset <strong>{{ run.dataset.name }}</strong>. 
            The system evaluated predicting target column <strong>{{ run.target_column }}</strong>, analyzing it as a 
            <strong>{{ run.problem_type }}</strong> problem. Multiple algorithms were preprocessed, trained, and optimized using Optuna, 
            with a champion selected based on cross-validation metrics.
        </p>

        <h2>2. Data Profiling & Quality Assessment</h2>
        <p>The profile analysis scanned <strong>{{ profile.rows }}</strong> rows and <strong>{{ profile.columns }}</strong> columns.</p>
        
        <h3>Dataset Column Breakdown</h3>
        <ul>
            <li><strong>Numerical Features:</strong> {{ profile.numerical_columns | length }}</li>
            <li><strong>Categorical Features:</strong> {{ profile.categorical_columns | length }}</li>
            <li><strong>Skipped index/ID columns:</strong> {{ profile.id_like_columns | join(', ') or 'None' }}</li>
        </ul>
        
        {% if profile.warnings %}
        <div class="alert-box" style="background-color: #FFF5F5; border-left-color: #E53E3E; color: #9B2C2C;">
            <strong>Profiling Alerts:</strong>
            <ul>
                {% for w in profile.warnings %}
                <li>{{ w }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if charts.target_distribution %}
        <div class="chart-container">
            <h3>Target Value Distribution</h3>
            <img class="chart-image" src="{{ charts.target_distribution }}" alt="Target Distribution">
        </div>
        {% endif %}

        {% if charts.correlation_heatmap %}
        <div class="chart-container">
            <h3>Numerical Feature Interactions (Correlation Map)</h3>
            <img class="chart-image" src="{{ charts.correlation_heatmap }}" alt="Correlation Map">
        </div>
        {% endif %}

        <h2>3. Preprocessing Configuration</h2>
        <p>
            Numerical features were normalized utilizing a robust StandardScaler pipeline and missing fields completed using median values. 
            Categorical columns were completed using constant replacement and encoded via a sparse One-Hot matrix.
        </p>

        <h2>4. Model Comparison Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Candidate</th>
                    <th>Cross-Validation Mean ({{ run.primary_metric }})</th>
                    <th>Validation Holdout ({{ run.primary_metric }})</th>
                    <th>Overfitting Risk</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for exp in trials %}
                <tr>
                    <td><strong>{{ exp.model_name }}</strong></td>
                    <td>{{ exp.cv_mean | round(4) }}</td>
                    <td>{{ exp.val_score | round(4) }}</td>
                    <td>
                        <span class="badge {% if exp.overfitting_risk == 'high' %}badge-danger{% elif exp.overfitting_risk == 'moderate' %}badge-warning{% else %}badge-success{% endif %}">
                            {{ exp.overfitting_risk | upper }}
                        </span>
                    </td>
                    <td>{{ exp.status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="alert-box" style="background-color: #F0FFF4; border-left-color: #38A169; color: #276749;">
            <strong>Champion Model Selected:</strong> {{ best.model_name }}<br>
            <strong>Best Hyperparameters:</strong> <code>{{ best.hyperparameters | tojson }}</code>
        </div>
        
        <h2>5. Validation Critic Inspection</h2>
        <p>{{ critic_comments }}</p>

        <h2>6. Model Explainability</h2>
        <p>Below is the statistical significance representation of feature importance computed by SHAP/feature coefficients:</p>

        {% if charts.explain_importance %}
        <div class="chart-container">
            <img class="chart-image" src="{{ charts.explain_importance }}" alt="Feature Importance">
        </div>
        {% endif %}

        <div class="footer">
            Report autonomously compiled by ML Research Lab.
        </div>
    </div>
</body>
</html>
"""
        html_tmpl = env.from_string(html_template)
        html_content = html_tmpl.render(
            run=run_obj,
            profile=profile_data,
            trials=trials_list,
            best=best_exp,
            charts=charts_b64,
            critic_comments=critic_comments,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Compiled final research report: {report_path}")
        return md_content, html_content, report_path

import numpy as np
import datetime
