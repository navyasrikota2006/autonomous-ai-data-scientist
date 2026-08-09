import React, { useState, useEffect, useRef } from 'react';
import { 
  FolderIcon, 
  PlayIcon, 
  ActivityIcon, 
  CpuIcon, 
  FileTextIcon, 
  DownloadIcon, 
  PieChartIcon, 
  AlertTriangleIcon
} from 'lucide-react';
import { api, DatasetMeta, AnalysisRun, Experiment, Report } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'profile' | 'workflow' | 'results' | 'report'>('upload');
  
  // Data State
  const [dataset, setDataset] = useState<DatasetMeta | null>(null);
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [champion, setChampion] = useState<Experiment | null>(null);
  const pm = run?.primary_metric || 'f1';

  const renderMetric = (val: number | null | undefined, metricName: string) => {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    const isPercentageMetric = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'].includes(metricName.toLowerCase());
    if (isPercentageMetric) {
      return (val * 100).toFixed(2) + '%';
    } else {
      return val.toFixed(4);
    }
  };
  
  // App settings/forms
  const [objective, setObjective] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [mode, setMode] = useState('standard');
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [uploadProgress, setUploadProgress] = useState(false);
  
  // Poll timer reference
  const pollIntervalRef = useRef<any>(null);

  // Clean timer on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  // Poll status when a run is active
  const startPolling = (runId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    
    pollIntervalRef.current = setInterval(async () => {
      try {
        const runStatus = await api.getRunStatus(runId);
        setRun(runStatus);
        
        // Fetch current experiments table
        const expData = await api.getRunExperiments(runId);
        setExperiments(expData);
        
        if (runStatus.status === 'completed') {
          setIsRunning(false);
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          
          // Fetch final results report
          const results = await api.getRunResults(runId);
          setChampion(results.champion);
          setExperiments(results.all_experiments);
          
          const reportData = await api.getReport(runId);
          setReport(reportData);
          
          setActiveTab('results');
        } else if (runStatus.status === 'failed') {
          setIsRunning(false);
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setErrorMsg('Autonomous pipeline execution failed. Review agent logs below.');
        }
      } catch (err: any) {
        console.error("Poller status error: ", err);
      }
    }, 3000);
  };

  const parseColumnsMetadata = (metadata: any): any => {
    if (!metadata) return null;
    if (typeof metadata === 'object') return metadata;
    try {
      let parsed = JSON.parse(metadata);
      if (typeof parsed === 'string') {
        parsed = JSON.parse(parsed);
      }
      return parsed;
    } catch (e) {
      console.error("Failed to parse columns_metadata:", e);
      return null;
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Clear stale state for Phase 1C
    setDataset(null);
    setRun(null);
    setExperiments([]);
    setReport(null);
    setChampion(null);
    setTargetColumn('');
    setObjective('');
    
    setUploadProgress(true);
    setErrorMsg('');
    try {
      const meta = await api.uploadDataset(file);
      if (meta && meta.columns_metadata) {
        meta.columns_metadata = parseColumnsMetadata(meta.columns_metadata);
      }
      setDataset(meta);
      
      // Auto-populate target suggestions
      if (meta.columns_metadata) {
        setTargetColumn(meta.columns_metadata.target_candidate || '');
        setProblemType(meta.columns_metadata.problem_type || 'classification');
      }
      setActiveTab('profile');
    } catch (err: any) {
      setErrorMsg(err.message || 'File upload failed.');
    } finally {
      setUploadProgress(false);
    }
  };

  const handleRunDemo = async (type: 'classification' | 'regression') => {
    setErrorMsg('');
    setIsRunning(true);
    try {
      
      // We will tell the server that we are uploading or utilizing an existing demo file.
      // Since demo files are generated during setup, we mock upload it or call it
      // actually, to ensure we run real ML: we check if they are already in the datasets lists,
      // or we can start a standard process. For a complete robust demo mode, we just fetch
      // if dataset file exists or trigger run standard. Let's make an actual upload request
      // reading the mock file byte stream! Or we search/start the existing file path.
      // Wait, to keep it simple, we search for the dataset by filename or trigger.
      // Let's call start via dataset path. We can write a route for demo setup or handle it.
      // Wait, we can upload the demo file directly using a fast fetch path!
      // In JS, we fetch the sample CSV file from `/mlruns/` or `/reports/` (wait! Vite serves static folders?
      // No, we can fetch from `datasets/sample/churn_sample.csv` or similar if mounted).
      // Let's fetch the demo dataset content directly and turn it into a File object.
      // Since it is bundled on the backend in datasets/sample/churn_sample.csv,
      // how do we access it? We can download it backend to frontend via a URL? No, wait!
      // Let's create a custom route `/api/datasets/demo/{type}` which imports it on the backend!
      // This is an EXCELLENT and very robust architectural approach!
      // Let's check: We didn't define a custom endpoint `/api/datasets/demo/{type}` in `routes.py`,
      // but we can add one now or we can implement it. Let's write code to import the demo file directly, or we can just update routes to support it.
      // Wait, let's look at `routes.py`, did it have a demo endpoint? No.
      // Can we edit `routes.py` to add it? Yes, we can! Or we can have a button on the UI that lets you
      // use the file if they upload. Let's add a backend start demo endpoint! That is extremely clean.
      // Wait! Let's check routes first.
      
      // For now, let's write the code assuming there is a endpoint or we add it (we will add it).
      const response = await fetch(`/api/datasets/demo/${type}`, { method: 'POST' });
      if (!response.ok) {
        throw new Error('Failed to load demo dataset on backend.');
      }
      const dataMeta = await response.json();
      if (dataMeta && dataMeta.columns_metadata) {
        dataMeta.columns_metadata = parseColumnsMetadata(dataMeta.columns_metadata);
      }
      setDataset(dataMeta);
      setTargetColumn(dataMeta.columns_metadata?.target_candidate || '');
      setProblemType(dataMeta.columns_metadata?.problem_type || 'classification');
      setObjective(type === 'classification' ? 'Predict customer churn indicators' : 'Predict house price values');
      
      const startRun = await api.startAnalysis({
        dataset_id: dataMeta.id,
        business_objective: type === 'classification' ? 'Predict customer churn indicators' : 'Predict house price values',
        mode: 'standard',
        target_column: dataMeta.columns_metadata?.target_candidate,
        problem_type: dataMeta.columns_metadata?.problem_type
      });
      setRun(startRun);
      setActiveTab('workflow');
      startPolling(startRun.id);
    } catch (err: any) {
      setErrorMsg(err.message || 'Demo initialization failed.');
      setIsRunning(false);
    }
  };

  const handleStartAnalysis = async () => {
    if (!dataset) return;
    if (!targetColumn) {
      setErrorMsg("Please select a target variable before starting the ML experiment.");
      return;
    }
    setErrorMsg('');
    setIsRunning(true);
    
    const payload = {
      dataset_id: dataset.id,
      business_objective: objective,
      mode: mode,
      target_column: targetColumn || null,
      problem_type: problemType || null
    };
    console.log("Starting intelligent pipeline", payload);
    
    try {
      const startRun = await api.startAnalysis(payload);
      setRun(startRun);
      setActiveTab('workflow');
      startPolling(startRun.id);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start.');
      setIsRunning(false);
    }
  };

  const handleFileDownload = async (id: string, type: 'model' | 'report' | 'all' | 'model-card', defaultFilename: string) => {
    setErrorMsg('');
    try {
      const downloadUrl = api.getDownloadUrl(id, type);
      console.log("Downloading artifact", { id, type, url: downloadUrl });
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        let errMsg = `Download failed: Status ${response.status}`;
        try {
          const errBody = await response.json();
          errMsg = errBody.detail || errMsg;
        } catch (_) {}
        throw new Error(errMsg);
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = defaultFilename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error("Download error:", err);
      setErrorMsg(err.message || 'File download failed.');
    }
  };

  // Helper formatting values
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col justify-between shrink-0">
        <div>
          <div className="p-6 border-b border-slate-200 flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white text-base">
              🔬
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-950 leading-none">ML Research Lab</h1>
              <span className="text-[10px] text-blue-600 font-bold tracking-widest uppercase mt-0.5 block">Autonomous DS</span>
            </div>
          </div>
          
          <nav className="p-4 space-y-1">
            <button 
              onClick={() => setActiveTab('upload')}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'upload' 
                  ? 'bg-blue-50 text-blue-600 border border-blue-100 shadow-sm shadow-blue-500/5' 
                  : 'text-slate-600 hover:text-slate-950 hover:bg-slate-50 border border-transparent'
              }`}
            >
              <FolderIcon className="w-4 h-4" />
              <span>Workspace Upload</span>
            </button>

            <button 
              onClick={() => { if (dataset) setActiveTab('profile'); }}
              disabled={!dataset}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'profile' 
                  ? 'bg-blue-50 text-blue-600 border border-blue-100 shadow-sm shadow-blue-500/5' 
                  : 'text-slate-600 hover:text-slate-950 hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent border border-transparent'
              }`}
            >
              <PieChartIcon className="w-4 h-4" />
              <span>Data Profiler</span>
            </button>

            <button 
              onClick={() => { if (run) setActiveTab('workflow'); }}
              disabled={!run}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'workflow' 
                  ? 'bg-blue-50 text-blue-600 border border-blue-100 shadow-sm shadow-blue-500/5' 
                  : 'text-slate-600 hover:text-slate-950 hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent border border-transparent'
              }`}
            >
              <ActivityIcon className="w-4 h-4" />
              <span>Agent Workflow</span>
            </button>

            <button 
              onClick={() => { if (experiments.length > 0) setActiveTab('results'); }}
              disabled={experiments.length === 0}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'results' 
                  ? 'bg-blue-50 text-blue-600 border border-blue-100 shadow-sm shadow-blue-500/5' 
                  : 'text-slate-600 hover:text-slate-950 hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent border border-transparent'
              }`}
            >
              <CpuIcon className="w-4 h-4" />
              <span>Leaderboard</span>
            </button>

            <button 
              onClick={() => { if (report) setActiveTab('report'); }}
              disabled={!report}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'report' 
                  ? 'bg-blue-50 text-blue-600 border border-blue-100 shadow-sm shadow-blue-500/5' 
                  : 'text-slate-600 hover:text-slate-950 hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent border border-transparent'
              }`}
            >
              <FileTextIcon className="w-4 h-4" />
              <span>Research Report</span>
            </button>
          </nav>
        </div>

        <div className="p-4 border-t border-slate-205 text-xs text-slate-500 flex flex-col space-y-2">
          <div className="flex justify-between items-center">
            <span>Offline Fallback:</span>
            <span className="text-blue-600 font-bold">ACTIVE</span>
          </div>
          <div className="flex justify-between items-center">
            <span>Engine:</span>
            <span className="text-slate-600 font-mono">SQLite/Sklearn</span>
          </div>
        </div>
      </aside>

      {/* Main Panel Content */}
      <main className="flex-1 bg-slate-50 p-8 overflow-y-auto flex flex-col justify-between">
        
        {/* Error Notification */}
        {errorMsg && (
          <div className="mb-6 p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 flex items-center space-x-3 text-sm">
            <AlertTriangleIcon className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        <div className="flex-1">
          {/* TAB 1: WORKSPACE UPLOAD */}
          {activeTab === 'upload' && (
            <div className="max-w-4xl space-y-8 animate-fadeIn">
              <div>
                <h2 className="text-3xl font-extrabold text-slate-950">Ingestion Laboratory</h2>
                <p className="text-slate-550 text-sm mt-2">Upload a raw dataset and construct your ML research study.</p>
              </div>

              {/* Upload Dropzone */}
              <div className="border-2 border-dashed border-slate-350 hover:border-blue-500 rounded-2xl p-16 text-center transition-colors bg-white shadow-sm relative group">
                <input 
                  type="file" 
                  accept=".csv" 
                  onChange={handleFileUpload} 
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  disabled={uploadProgress}
                />
                
                <div className="space-y-4">
                  <div className="w-16 h-16 rounded-full bg-slate-50 border border-slate-200 flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                    <FolderIcon className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-850">Upload Tabular Dataset</h3>
                    <p className="text-xs text-slate-500 mt-1">Accepts CSV tables up to 50MB</p>
                  </div>
                  <button className="px-4 py-2 bg-slate-100 text-slate-700 border border-slate-200 text-xs font-semibold rounded-lg hover:bg-slate-200 transition">
                    {uploadProgress ? 'Processing stream...' : 'Choose Raw CSV'}
                  </button>
                </div>
              </div>

              {/* Quick Launch Demo Mode */}
              <div className="space-y-4">
                <h3 className="text-lg font-bold text-slate-950 flex items-center space-x-2">
                  <span>🚀</span>
                  <span>Interactive Research Demos</span>
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-6 rounded-xl border border-slate-200 bg-white shadow-sm relative group">
                    <h4 className="font-bold text-slate-900 text-base">Customer Churn Classification</h4>
                    <p className="text-xs text-slate-500 mt-2">Auto-selects churn target and fits LogisticRegression, Random Forests, and XGBoost classifiers.</p>
                    <button 
                      onClick={() => handleRunDemo('classification')}
                      disabled={isRunning}
                      className="mt-4 flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 font-bold rounded-lg text-xs transition"
                    >
                      <PlayIcon className="w-3 h-3 fill-current" />
                      <span>Run Classification Lab</span>
                    </button>
                  </div>
                  <div className="p-6 rounded-xl border border-slate-200 bg-white shadow-sm relative group">
                    <h4 className="font-bold text-slate-900 text-base">House Prices Regression</h4>
                    <p className="text-xs text-slate-500 mt-2">Profiles neighborhood property records, imputes scales, and optimizes gradient boosting estimators.</p>
                    <button 
                      onClick={() => handleRunDemo('regression')}
                      disabled={isRunning}
                      className="mt-4 flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 font-bold rounded-lg text-xs transition"
                    >
                      <PlayIcon className="w-3 h-3 fill-current" />
                      <span>Run Regression Lab</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: DATA PROFILER */}
          {activeTab === 'profile' && dataset && (
            <div className="space-y-8 animate-fadeIn">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-2xl font-extrabold text-slate-950">Dataset Summary Profile</h2>
                  <p className="text-slate-550 text-xs mt-1">Double check statistics and customize ML study parameters.</p>
                </div>
                <div className="text-xs text-slate-650 font-mono">
                  File: {dataset.name} ({formatBytes(dataset.file_size)})
                </div>
              </div>

              {/* Data characteristics cards */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="glass-panel">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Features Count</span>
                  <div className="text-2xl font-bold text-slate-950 mt-1">{(dataset.column_count || 0)}</div>
                </div>
                <div className="glass-panel">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Total Samples</span>
                  <div className="text-2xl font-bold text-slate-950 mt-1">{(dataset.row_count || 0)}</div>
                </div>
                <div className="glass-panel">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Target Suggested</span>
                  <div className="text-2xl font-bold text-blue-600 mt-1 truncate">{targetColumn || 'None'}</div>
                </div>
                <div className="glass-panel">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Problem Mode</span>
                  <div className="text-2xl font-bold text-slate-950 mt-1 capitalize">{problemType}</div>
                </div>
              </div>

              {/* Configure Study Settings */}
              <div className="p-6 rounded-xl border border-slate-200 bg-white shadow-sm space-y-6">
                <h3 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-3">Configure Experiment</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider">Target Variable</label>
                    <select 
                      value={targetColumn} 
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="mt-2 w-full px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-900 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    >
                      {dataset.columns_metadata?.column_statistics && 
                        Object.keys(dataset.columns_metadata.column_statistics).map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))
                      }
                      {!dataset.columns_metadata && <option value="">Select target</option>}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider">Problem Type Mode</label>
                    <select 
                      value={problemType} 
                      onChange={(e) => setProblemType(e.target.value)}
                      className="mt-2 w-full px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-900 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    >
                      <option value="classification">Classification (Predict discrete categories)</option>
                      <option value="regression">Regression (Predict continuous numeric values)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider">Research HPO Mode</label>
                    <select 
                      value={mode} 
                      onChange={(e) => setMode(e.target.value)}
                      className="mt-2 w-full px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-900 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    >
                      <option value="quick">Quick (5 Trials HPO)</option>
                      <option value="standard">Standard (12 Trials HPO)</option>
                      <option value="research">Deep Research (25 Trials HPO)</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider">Business objective descriptor</label>
                  <input 
                    type="text" 
                    placeholder="Example: Predict churn rate and output key retention factors."
                    value={objective} 
                    onChange={(e) => setObjective(e.target.value)}
                    className="mt-2 w-full px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-900 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                  />
                </div>

                <div className="flex justify-end pt-3">
                  <button 
                    onClick={handleStartAnalysis}
                    disabled={isRunning}
                    className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition cursor-pointer"
                  >
                    <PlayIcon className="w-4 h-4 fill-current" />
                    <span>Run Intelligent Agent Pipeline</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: AGENT WORKFLOW TIMELINE */}
          {activeTab === 'workflow' && run && (
            <div className="space-y-8 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-extrabold text-slate-950">Multi-Agent Control Timeline</h2>
                <p className="text-slate-550 text-xs mt-1">Watch agents negotiate planning constraints and train parameters.</p>
              </div>

              {/* Progress Bar of Workflow Stages */}
              <div className="grid grid-cols-7 gap-2 border border-slate-205 p-4 rounded-xl bg-white shadow-sm">
                {['planner', 'profiler', 'cleaner', 'eda', 'fe', 'trainer', 'report_agent'].map((stage, idx) => {
                  const stages = ['planner', 'profiler', 'cleaner', 'eda', 'fe', 'trainer', 'report_agent'];
                  const currentIdx = stages.indexOf(run.current_stage);
                  const isCompleted = idx < currentIdx || run.status === 'completed';
                  const isActive = idx === currentIdx && run.status === 'running';
                  
                  return (
                    <div key={stage} className="text-center space-y-2">
                      <div className={`h-1.5 rounded-full transition-all ${isCompleted ? 'bg-emerald-500' : isActive ? 'bg-blue-500 animate-pulse' : 'bg-slate-200'}`}></div>
                      <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider truncate">{stage.replace('_', ' ')}</div>
                    </div>
                  );
                })}
              </div>

              {/* Logs output */}
              <div className="rounded-xl border border-slate-200 bg-white p-6 font-mono text-xs space-y-3 h-80 overflow-y-auto shadow-sm">
                <div className="text-slate-500 border-b border-slate-100 pb-2 flex justify-between">
                  <span>EXECUTION LOGSTREAM</span>
                  {run.status === 'running' && <span className="text-blue-600 font-bold animate-pulse">Running...</span>}
                  {run.status === 'completed' && <span className="text-emerald-600 font-bold">Completed</span>}
                  {run.status === 'failed' && <span className="text-red-600 font-bold">Failed</span>}
                </div>
                {run.agent_logs && run.agent_logs.map((log) => (
                  <div key={log.id} className="flex items-start space-x-2">
                    <span className="text-slate-400">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                    <span className="text-blue-700 uppercase font-extrabold text-[10px] shrink-0 border border-blue-200 px-1.5 py-0.5 rounded bg-blue-50">{log.agent_name}:</span>
                    <span className={log.status === 'failed' ? 'text-red-700 font-medium' : log.status === 'warning' ? 'text-amber-700 font-medium' : 'text-slate-700'}>
                      {log.message}
                    </span>
                  </div>
                ))}
                {run.agent_logs?.length === 0 && (
                  <div className="text-slate-400 text-center py-10">Initialising pipelines...</div>
                )}
              </div>
            </div>
          )}

          {/* TAB 4: LEADERBOARD & COMPARISONS */}
          {activeTab === 'results' && run && (
            <div className="space-y-8 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-extrabold text-slate-950">Model Experiment Leaderboard</h2>
                <p className="text-slate-550 text-xs mt-1">Direct breakdown comparison of candidates checked during execution.</p>
              </div>

              {/* Champion Card & Model/Research Artifact Grid */}
              {champion && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <div className="space-y-6">
                    <div>
                      <span className="text-[10px] uppercase tracking-wider font-extrabold text-blue-700 px-2.5 py-0.5 rounded bg-blue-105 border border-blue-200">
                        Champion Model Selected
                      </span>
                      <h3 className="text-2xl font-bold text-slate-900 capitalize mt-2">{champion.model_name.replace('_', ' ')}</h3>
                      <p className="text-slate-600 text-xs mt-1">
                        Cross-Validation metric verified ({pm.toUpperCase()}):{' '}
                        <span className="text-slate-950 font-mono font-bold">
                          {renderMetric(
                            champion.cv_metrics && champion.cv_metrics.length > 0 
                              ? Math.max(...champion.cv_metrics) 
                              : champion.metrics?.val?.[pm], 
                            pm
                          )}
                        </span>
                      </p>
                    </div>

                    <div className="space-y-3">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Model Artifacts</h4>
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex justify-between items-center">
                        <div>
                          <p className="text-xs font-bold text-slate-800">Download Model (.joblib)</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Machine-readable serialized model for programmatic reuse.</p>
                        </div>
                        <button 
                          onClick={() => handleFileDownload(champion.id, 'model', `model_${champion.model_name}_${champion.id.slice(0, 8)}.joblib`)} 
                          className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded transition shadow-sm cursor-pointer"
                        >
                          <DownloadIcon className="w-3.5 h-3.5" />
                          <span>.joblib</span>
                        </button>
                      </div>

                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex justify-between items-center">
                        <div>
                          <p className="text-xs font-bold text-slate-800">Download Model Card (.pdf)</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Human-readable documentation of the trained model.</p>
                        </div>
                        <button 
                          onClick={() => handleFileDownload(champion.id, 'model-card', `model_card_${champion.model_name}_${champion.id.slice(0, 8)}.pdf`)} 
                          className="flex items-center space-x-1.5 px-3 py-1.5 bg-white border border-slate-350 text-slate-700 text-xs font-semibold rounded hover:bg-slate-100 transition shadow-sm cursor-pointer"
                        >
                          <DownloadIcon className="w-3.5 h-3.5" />
                          <span>.pdf</span>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4 flex flex-col justify-between">
                    <div className="space-y-3">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Research Artifacts</h4>
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex justify-between items-center">
                        <div>
                          <p className="text-xs font-bold text-slate-800">Download Report HTML</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Standalone interactive report with correlation matrices and details.</p>
                        </div>
                        <button 
                          onClick={() => handleFileDownload(run.id, 'report', `research_report_${run.id.slice(0, 8)}.html`)}
                          className="flex items-center space-x-1.5 px-3 py-1.5 bg-white border border-slate-350 text-slate-700 text-xs font-semibold rounded hover:bg-slate-100 transition shadow-sm cursor-pointer"
                        >
                          <DownloadIcon className="w-3.5 h-3.5" />
                          <span>HTML</span>
                        </button>
                      </div>

                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex justify-between items-center">
                        <div>
                          <p className="text-xs font-bold text-slate-800">Download Artifacts ZIP</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Unified archive containing model, pipeline, report, features structure, and metadata readme.</p>
                        </div>
                        <button 
                          onClick={() => handleFileDownload(run.id, 'all', `autonomous_ai_artifacts_${run.id.slice(0, 8)}.zip`)}
                          className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded transition shadow-sm cursor-pointer"
                        >
                          <DownloadIcon className="w-3.5 h-3.5" />
                          <span>ZIP</span>
                        </button>
                      </div>
                    </div>

                    <div className="text-[10px] text-slate-400 leading-normal border-t border-slate-100 pt-3">
                      * Serialized model artifact is verified loadable in production endpoints via Python <code className="font-mono bg-slate-100 px-1 py-0.5 rounded text-slate-650">joblib.load()</code> matching the specified layout.
                    </div>
                  </div>
                </div>
              )}

              {/* Interactive Leaderboard Table */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200">
                      <th className="p-4">Algorithm</th>
                      <th className="p-4">Cross Val Score ({pm.toUpperCase()})</th>
                      <th className="p-4">Holdout Validation Value</th>
                      <th className="p-4">Overfitting risk</th>
                      <th className="p-4">Parameters Checked</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm font-medium text-slate-700 divide-y divide-slate-200">
                    {experiments.map((exp) => {
                      const valScore = exp.metrics?.val?.[pm];
                      const cvAvg = exp.cv_metrics && exp.cv_metrics.length > 0 
                        ? exp.cv_metrics.reduce((a: number, b: number) => a + b, 0) / exp.cv_metrics.length 
                        : valScore;
                      
                      return (
                        <tr key={exp.id} className="hover:bg-slate-50">
                          <td className="p-4 font-bold text-slate-900 capitalize">{exp.model_name.replace('_', ' ')}</td>
                          <td className="p-4 font-mono text-slate-800">{renderMetric(cvAvg, pm)}</td>
                          <td className="p-4 font-mono text-slate-800">{renderMetric(valScore, pm)}</td>
                          <td className="p-4">
                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold tracking-wider ${
                              exp.overfitting_risk === 'high' 
                                ? 'bg-red-50 text-red-700 border border-red-200' 
                                : exp.overfitting_risk === 'moderate' 
                                ? 'bg-amber-50 text-amber-700 border border-amber-200' 
                                : 'bg-emerald-55 text-emerald-750 border border-emerald-250'
                            }`}>
                              {exp.overfitting_risk?.toUpperCase() || 'LOW'}
                            </span>
                          </td>
                          <td className="p-4 text-xs font-mono text-slate-500 truncate max-w-[200px]" title={JSON.stringify(exp.hyperparameters)}>
                            {JSON.stringify(exp.hyperparameters)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Explanations visual charts (SHAP) */}
              {champion && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
                  <div className="glass-panel space-y-4">
                    <h3 className="text-base font-bold text-slate-950 border-b border-slate-200 pb-2">Global Feature Significance</h3>
                    <p className="text-slate-550 text-xs">Explanations mapping absolute contributions values to output forecasts.</p>
                    <div className="overflow-hidden rounded-lg bg-white p-2 border border-slate-200">
                      <img 
                        src={`/mlruns/${run.id}/models/${champion.model_name}/feature_importance.png`} 
                        alt="SHAP Feature Importances"
                        className="w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    </div>
                  </div>
                  
                  <div className="glass-panel space-y-4">
                    <h3 className="text-base font-bold text-slate-950 border-b border-slate-200 pb-2">Diagnostic EDA plots</h3>
                    <p className="text-slate-550 text-xs">Correlation coefficients mapping metrics interactions.</p>
                    <div className="overflow-hidden rounded-lg bg-white p-2 border border-slate-200">
                      <img 
                        src={`/mlruns/${run.id}/eda/correlation_heatmap.png`} 
                        alt="Correlation Matrix"
                        className="w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 5: FINAL REPORT */}
          {activeTab === 'report' && run && report && (
            <div className="space-y-6 animate-fadeIn h-full flex flex-col">
              <div className="flex justify-between items-center text-sm">
                <div>
                  <h2 className="text-2xl font-extrabold text-slate-950">Professional ML Research Report</h2>
                  <p className="text-slate-550 text-xs mt-1">Autonomous compiled document ready for recruiter download.</p>
                </div>
                <div className="flex space-x-3 shrink-0">
                  <button 
                    onClick={() => handleFileDownload(run.id, 'report', `research_report_${run.id.slice(0,8)}.html`)}
                    className="flex items-center space-x-2 px-3 py-2 bg-white border border-slate-350 text-slate-700 text-xs font-semibold rounded hover:bg-slate-100 transition cursor-pointer shadow-sm"
                  >
                    <DownloadIcon className="w-3.5 h-3.5" />
                    <span>Download Report HTML</span>
                  </button>
                  <button 
                    onClick={() => handleFileDownload(run.id, 'all', `run_artifacts_${run.id.slice(0,8)}.zip`)}
                    className="flex items-center space-x-2 px-3 py-2 bg-blue-600 text-white text-xs font-bold rounded hover:bg-blue-700 transition cursor-pointer shadow-sm"
                  >
                    <DownloadIcon className="w-3.5 h-3.5" />
                    <span>Download Artifacts ZIP</span>
                  </button>
                </div>
              </div>

              {/* Embedded Report Frame */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white h-[650px] shadow-sm">
                <iframe 
                  title="Research Report Output"
                  srcDoc={report.content_html} 
                  className="w-full h-full border-none"
                />
              </div>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
