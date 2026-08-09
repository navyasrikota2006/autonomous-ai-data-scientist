from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class ExperimentResponse(BaseModel):
    id: str
    analysis_run_id: str
    model_name: str
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    cv_metrics: Optional[List[float]] = None
    overfitting_risk: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    features_used: Optional[List[str]] = None
    artifact_path: Optional[str] = None
    model_path: Optional[str] = None
    created_at: datetime
    
    @validator("hyperparameters", "metrics", "cv_metrics", "features_used", pre=True, always=False, check_fields=False)
    def parse_json_fields(cls, v):
        import json
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v

    class Config:
        from_attributes = True

class ReportResponse(BaseModel):
    id: str
    analysis_run_id: str
    content_markdown: str
    content_html: str
    report_path: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ModelRegistryResponse(BaseModel):
    id: str
    run_id: str
    model_name: str
    problem_type: str
    target_column: str
    metrics: Dict[str, Any]
    created_at: datetime
