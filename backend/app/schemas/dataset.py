from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class DatasetMetadata(BaseModel):
    name: str
    row_count: int
    column_count: int
    missing_columns: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    warnings: List[str]

class DatasetCreate(BaseModel):
    name: str
    filepath: str
    file_size: int

class DatasetResponse(BaseModel):
    id: str
    name: str
    filepath: str
    file_size: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    @validator("columns_metadata", pre=True, always=False, check_fields=False)
    def parse_metadata_string(cls, v):
        import json
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v

    class Config:
        from_attributes = True

class ProfileResponse(BaseModel):
    rows: int
    columns: int
    target_candidate: str
    problem_type: str
    missing_columns: List[str]
    categorical_columns: List[str]
    numerical_columns: List[str]
    warnings: List[str]
    correlations: Dict[str, Dict[str, float]]
    column_statistics: Dict[str, Dict[str, Any]]
