export interface DatasetMeta {
  id: string;
  name: string;
  filepath: string;
  file_size: number;
  row_count: number | null;
  column_count: number | null;
  columns_metadata: any | null;
  created_at: string;
}

export interface AgentLog {
  id: number;
  agent_name: string;
  status: string;
  message: string | null;
  timestamp: string;
}

export interface AnalysisRun {
  id: string;
  dataset_id: string;
  business_objective: string | null;
  mode: string;
  target_column: string | null;
  problem_type: string | null;
  primary_metric: string | null;
  status: string;
  current_stage: string;
  best_experiment_id: string | null;
  created_at: string;
  updated_at: string;
  agent_logs: AgentLog[];
}

export interface Experiment {
  id: string;
  analysis_run_id: string;
  model_name: string;
  hyperparameters: any | null;
  metrics: any | null;
  cv_metrics: number[] | null;
  overfitting_risk: string | null;
  status: string;
  error_message: string | null;
  features_used: string[] | null;
  artifact_path: string | null;
  model_path: string | null;
  created_at: string;
}

export interface Report {
  id: string;
  analysis_run_id: string;
  content_markdown: string;
  content_html: string;
  report_path: string;
  created_at: string;
}

const API_BASE = '/api';

export const api = {
  uploadDataset: async (file: File): Promise<DatasetMeta> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Failed to upload dataset.');
    }
    return response.json();
  },

  startAnalysis: async (params: {
    dataset_id: string;
    business_objective?: string;
    mode: string;
    target_column?: string | null;
    problem_type?: string | null;
  }): Promise<AnalysisRun> => {
    const response = await fetch(`${API_BASE}/analysis/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Failed to start analysis.');
    }
    return response.json();
  },

  getRunStatus: async (runId: string): Promise<AnalysisRun> => {
    const response = await fetch(`${API_BASE}/analysis/${runId}/status`);
    if (!response.ok) throw new Error('Run not found.');
    return response.json();
  },

  getRunExperiments: async (runId: string): Promise<Experiment[]> => {
    const response = await fetch(`${API_BASE}/analysis/${runId}/experiments`);
    if (!response.ok) throw new Error('Failed to retrieve experiments.');
    return response.json();
  },

  getRunResults: async (runId: string): Promise<{
    run_id: string;
    problem_type: string;
    target_column: string;
    primary_metric: string;
    champion: Experiment | null;
    all_experiments: Experiment[];
  }> => {
    const response = await fetch(`${API_BASE}/analysis/${runId}/results`);
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Failed to retrieve results.');
    }
    return response.json();
  },

  getReport: async (runId: string): Promise<Report> => {
    const response = await fetch(`${API_BASE}/analysis/${runId}/report`);
    if (!response.ok) throw new Error('Report not ready.');
    return response.json();
  },

  getDownloadUrl: (id: string, type: 'model' | 'report' | 'all' | 'model-card'): string => {
    if (type === 'model') {
      return `/api/models/${id}/download`;
    } else if (type === 'model-card') {
      return `/api/models/${id}/model-card`;
    } else if (type === 'report') {
      return `/api/analysis/${id}/download-report`;
    } else {
      return `/api/analysis/${id}/download-all`;
    }
  }
};
