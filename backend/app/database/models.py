import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from app.database.connection import Base

def generate_uuid():
    return str(uuid.uuid4())

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns_metadata = Column(JSON, nullable=True) # Column list, nulls, types
    created_at = Column(DateTime, default=datetime.utcnow)
    
    runs = relationship("AnalysisRun", back_populates="dataset", cascade="all, delete-orphan")

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    business_objective = Column(Text, nullable=True)
    mode = Column(String(50), default="standard") # quick, standard, research
    target_column = Column(String(255), nullable=True)
    problem_type = Column(String(50), nullable=True) # classification, regression
    primary_metric = Column(String(50), nullable=True)
    status = Column(String(50), default="pending") # pending, running, completed, failed
    current_stage = Column(String(50), default="idle")
    best_experiment_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="runs")
    agent_logs = relationship("AgentRunLog", back_populates="analysis_run", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="analysis_run", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="analysis_run", cascade="all, delete-orphan")

class AgentRunLog(Base):
    __tablename__ = "agent_run_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_run_id = Column(String(36), ForeignKey("analysis_runs.id"), nullable=False)
    agent_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False) # started, running, completed, failed
    message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    analysis_run = relationship("AnalysisRun", back_populates="agent_logs")

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    analysis_run_id = Column(String(36), ForeignKey("analysis_runs.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True) # train metrics, test/val metrics
    cv_metrics = Column(JSON, nullable=True) # list of fold scores
    overfitting_risk = Column(String(50), nullable=True)
    status = Column(String(50), nullable=False) # success, failed
    error_message = Column(Text, nullable=True)
    features_used = Column(JSON, nullable=True)
    artifact_path = Column(String(512), nullable=True) # folder with plots
    model_path = Column(String(512), nullable=True) # trained joblib file path
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis_run = relationship("Experiment", secondary="analysis_runs", primaryjoin="Experiment.analysis_run_id==AnalysisRun.id", foreign_keys="[Experiment.analysis_run_id]", overlaps="experiments")
    analysis_run = relationship("AnalysisRun", back_populates="experiments")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    analysis_run_id = Column(String(36), ForeignKey("analysis_runs.id"), nullable=False)
    content_markdown = Column(Text, nullable=False)
    content_html = Column(Text, nullable=False)
    report_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis_run = relationship("AnalysisRun", back_populates="reports")
