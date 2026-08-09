import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.database.connection import engine, Base
from app.api.routes import router as api_router

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 1. Initialize Database Tables Automatically on Startup
try:
    logger.info("Initializing SQLite/PostgreSQL database schemas...")
    Base.metadata.create_all(bind=engine)
    logger.info("Schema tables complete.")
except Exception as e:
    logger.exception("Failed to initialize database tables:")

# 2. Boot FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multi-agent machine learning platform combining automated pipelines and validation critics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 3. Configure CORS policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Mount Static Directories for Web Artifact Rendering
# Serves the artifacts (EDA and SHAP plots) directly for frontend <img> binding.
if os.path.exists(settings.RUNS_DIR):
    app.mount("/mlruns", StaticFiles(directory=settings.RUNS_DIR), name="mlruns")
if os.path.exists(settings.REPORTS_DIR):
    app.mount("/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

# 5. Mount API Routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API Service",
        "health": "/api/health",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Execute standalone
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
