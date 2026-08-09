from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.core.llm import LLMProvider

class CriticResponse(BaseModel):
    status: str = Field(description="Decision status: PASS, WARNING, or RETRY")
    reason: str = Field(description="Detailed reason justifying the critic's decision")
    recommended_action: str = Field(description="Advice for the orchestrator, e.g. add regularization, drop features, or continue")
    parameters_adjustment: Dict[str, Any] = Field(
        default={}, 
        description="Suggested parameter adjustments to overlay in HPO/preprocessing"
    )

class QualityCritic:
    @staticmethod
    def inspect_model(
        model_name: str, 
        problem_type: str, 
        primary_metric: str, 
        metrics: Dict[str, Any], 
        cv_metrics: List[float]
    ) -> CriticResponse:
        """
        Critiques model performance and returns a structured response.
        Uses LLM if available, otherwise executes deterministic checks.
        """
        # Collect diagnostic statistics
        train_score = metrics["train"].get(primary_metric, 0.0)
        val_score = metrics["val"].get(primary_metric, 0.0)
        gap = train_score - val_score
        
        cv_mean = float(np.mean(cv_metrics)) if cv_metrics else val_score
        cv_std = float(np.std(cv_metrics)) if cv_metrics else 0.0
        
        system_prompt = (
            "You are a Senior ML Critic in an Autonomous Data Science Laboratory. "
            "Your job is to inspect model training results, check for overfitting, "
            "target leaks, validation instability, and decide if the run passes, "
            "needs a Warning, or must be retried with adjustments."
        )
        
        user_prompt = (
            f"Model Trained: {model_name}\n"
            f"Problem Type: {problem_type}\n"
            f"Primary Metric Target: {primary_metric}\n"
            f"Training Split Score: {train_score:.4f}\n"
            f"Validation Split Score: {val_score:.4f}\n"
            f"Validation Gap (Train - Val): {gap:.4f}\n"
            f"Cross-Validation Scores: {cv_metrics}\n"
            f"CV Score Mean: {cv_mean:.4f}\n"
            f"CV Score Std Dev: {cv_std:.4f}\n\n"
            "Analyze these results. Overfitting occurs if the gap is large (typically > 0.15 for classification, > 0.20 for regression).\n"
            "Unstable models have high CV standard deviation (> 0.12).\n"
            "Target leakage is suspicious if validation scores are 1.000.\n\n"
            "Return JSON matching: \n"
            "{'status': 'PASS'|'WARNING'|'RETRY', 'reason': '...', 'recommended_action': '...', 'parameters_adjustment': {}}"
        )
        
        # 1. Call LLM if enabled
        if settings.LLM_PROVIDER.lower() != "none":
            try:
                res = LLMProvider.call_llm(system_prompt, user_prompt, json_schema=CriticResponse)
                if isinstance(res, CriticResponse):
                    return res
            except Exception as e:
                print(f"Critic Agent LLM call failed: {e}. Running fallback rule engine.")
                
        # 2. Rule Engine Fallback (Deterministic checks)
        status = "PASS"
        reason = "Model metrics fall within acceptable thresholds."
        recommended_action = "Accept the current model run and save it to the experiment store."
        parameters_adjustment = {}
        
        # Check target leakage
        if val_score >= 0.999 and train_score >= 0.999:
            status = "WARNING"
            reason = "Model scored a perfect 1.0. This strongly suggests target leakage (features reflecting the target itself)."
            recommended_action = "Check for identifiers, redundant target columns, or temporal leakage in dataset features."
        # Check high overfitting
        elif gap > 0.18:
            status = "RETRY"
            reason = f"High overfitting observed. The gap between training ({train_score:.2f}) and validation ({val_score:.2f}) is too large ({gap:.2f})."
            recommended_action = "Retrying the training with increased regularization constraints and decreased feature capacity."
            
            # Suggest regularization params for HPO retry
            if "forest" in model_name:
                parameters_adjustment = {"max_depth": 6, "min_samples_split": 8}
            elif "xgb" in model_name:
                parameters_adjustment = {"max_depth": 4, "learning_rate": 0.05}
            elif "regression" in model_name:
                parameters_adjustment = {"alpha": 10.0} # Increase penalty
        # Check validation stability
        elif cv_std > 0.12:
            status = "WARNING"
            reason = f"Validation score is unstable. Cross-validation standard deviation is high ({cv_std:.4f})."
            recommended_action = "Consider collecting more training rows or applying cross-validation stratifying."
            
        return CriticResponse(
            status=status,
            reason=reason,
            recommended_action=recommended_action,
            parameters_adjustment=parameters_adjustment
        )

# Simple importing import numpy to support mathematical operations in client
import numpy as np
