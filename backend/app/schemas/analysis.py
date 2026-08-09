from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class AnalysisStartRequest(BaseModel):
    dataset_id: str
    business_objective: Optional[str] = ""
    mode: str = "standard" # quick, standard, research
    target_column: Optional[str] = None
    problem_type: Optional[str] = None # classification, regression

class AgentLogResponse(BaseModel):
    id: int
    agent_name: str
    status: str
    message: Optional[str] = None
    timestamp: datetime
    
    class Config:
        from_attributes = True

class AnalysisRunResponse(BaseModel):
    id: str
    dataset_id: str
    business_objective: Optional[str] = None
    mode: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    primary_metric: Optional[str] = None
    status: str
    current_stage: str
    best_experiment_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    agent_logs: List[AgentLogResponse] = []
    
    class Config:
        from_attributes = True
