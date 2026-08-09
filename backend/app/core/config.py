import os
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Autonomous AI Data Scientist ML Research Lab"
    API_V1_STR: str = "/api"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    RUNS_DIR: str = os.path.join(BASE_DIR, "mlruns")
    
    # DB URL - Defaults to sqlite inside the backend folder for easy local setup
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'db.sqlite3')}")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # LLM config
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "none") # "openai", "anthropic", "gemini", "none" (offline)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    LLM_MODEL: Optional[str] = os.getenv("LLM_MODEL", None)
    
    # Limits
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024 # 50 MB
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
for directory in [settings.DATASETS_DIR, settings.REPORTS_DIR, settings.RUNS_DIR]:
    os.makedirs(directory, exist_ok=True)
