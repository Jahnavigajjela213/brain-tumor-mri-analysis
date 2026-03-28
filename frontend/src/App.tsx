import { useState } from 'react';
import { Brain, Activity, ShieldAlert, Cpu, Database, ChevronLeft } from 'lucide-react';
import FileUpload from './components/FileUpload';
import PredictionCard from './components/PredictionCard';
import SegmentationDisplay from './components/SegmentationDisplay';
import { uploadMRI, segmentMRI, predictSurvival, testDataset } from './services/api';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<{
    original_base64: string | null;
    mask_base64: string | null;
    overlay_base64: string | null;
    has_subregions?: boolean;
  } | null>(null);
  
  const [prediction, setPrediction] = useState<{ probability: number; days: number } | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isDatasetLoading, setIsDatasetLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisLabel, setAnalysisLabel] = useState<string>('Analyzing MRI...');

  const handleFileSelect = (file: File) => {
    console.log("File selected:", file.name);
    setSelectedFile(file);
    setResult(null);
    setPrediction(null);
    setError(null);
    
    const reader = new FileReader();
    reader.readAsDataURL(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      alert('Please select an MRI scan first.');
      return;
    }
    
    try {
      setResult(null);
      setPrediction(null);
      setLoading(true);
      setError(null);
      setAnalysisLabel('Uploading to server...');

      // 1. Upload
      const uploadData = await uploadMRI(selectedFile);
      console.log("Upload Response:", uploadData);

      // 2. Segment
      setAnalysisLabel('Segmenting tumor...');
      const segData = await segmentMRI(uploadData.upload_id);
      console.log("Segmentation Response (segData):", segData);
      
      // Update result state as requested
      setResult({
        original_base64: segData.original_base64,
        mask_base64: segData.mask_base64,
        overlay_base64: segData.overlay_base64,
        has_subregions: segData.has_subregions
      });

      // 3. Predict
      setAnalysisLabel('Analyzing clinical factors...');
      const predData = await predictSurvival(uploadData.upload_id);
      console.log("Prediction Response:", predData);

      setPrediction({
        probability: predData.survival_probability,
        days: predData.estimated_survival_days
      });
    } catch (err: any) {
      console.error("Analysis Pipeline Error:", err);
      setError(err.message || 'System error. Please ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadDatasetSample = async (index: number) => {
    try {
      setIsDatasetLoading(true);
      setError(null);
      setResult(null);
      setPrediction(null);
      
      console.log(`Loading dataset sample index: ${index}`);
      const data = await testDataset(index);
      console.log("Dataset Sample Response:", data);
      
      setResult({
        original_base64: data.original_base64,
        mask_base64: data.mask_base64,
        overlay_base64: data.overlay_base64,
        has_subregions: true // dataset samples have subregions in the new logic
      });

      setPrediction({
        probability: data.survival_probability,
        days: data.estimated_survival_days
      });
      
      setSelectedFile(new File([], data.dataset_image));
    } catch (err: any) {
      console.error("Dataset Explorer Error:", err);
      setError('Failed to load sample. Ensure BraTS dataset is processed.');
    } finally {
      setIsDatasetLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setPrediction(null);
    setLoading(false);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 pb-20 px-4 pt-10">
      {/* Header */}
      <nav className="max-w-7xl mx-auto flex items-center justify-between mb-16 px-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">NeuroGuard AI</h1>
            <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">MRI Analysis Suite</p>
          </div>
        </div>
        
        <div className="hidden lg:flex items-center space-x-6">
          <div className="flex items-center space-x-2 text-slate-400">
            <Cpu className="w-4 h-4 text-blue-500" />
            <span className="text-[10px] font-bold uppercase">U-Net v2.0</span>
          </div>
          <div className="flex items-center space-x-2 text-slate-400">
            <Database className="w-4 h-4 text-emerald-500" />
            <span className="text-[10px] font-bold uppercase">BraTS 2020 Basis</span>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto space-y-12">
        <header className="text-center space-y-4">
          <h2 className="text-4xl md:text-5xl font-extrabold text-white leading-tight">
            Clinical-Grade <span className="text-blue-500">MRI Analysis</span>
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto text-lg">
            Automated tumor segmentation and survival probability projections for clinical research support.
          </p>
        </header>

        {(!result && !prediction) || loading || isDatasetLoading ? (
          <div className="space-y-12">
            <FileUpload 
              onUpload={handleFileSelect} 
              onClear={handleReset}
              isLoading={loading || isDatasetLoading} 
              uploadedFile={selectedFile} 
            />

            {selectedFile && !loading && !isDatasetLoading && (
              <div className="flex flex-col items-center space-y-6 mt-8">
                <button
                  onClick={handleAnalyze}
                  className="px-12 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold transition-all transform hover:scale-105 active:scale-95 shadow-xl shadow-blue-500/20 flex items-center space-x-3"
                >
                  <Activity className="w-5 h-5" />
                  <span>Execute AI Pipeline</span>
                </button>
              </div>
            )}

            {(loading || isDatasetLoading) && (
              <div className="flex flex-col items-center space-y-4 py-12">
                <div className="w-12 h-12 border-4 border-blue-600/20 border-t-blue-600 rounded-full animate-spin"></div>
                <p className="text-blue-400 font-bold animate-pulse uppercase tracking-widest text-xs">{analysisLabel}</p>
              </div>
            )}

            {/* Dataset Explorer Section */}
            <section className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 mt-12">
               <div className="flex items-center justify-between mb-8">
                 <div className="flex items-center space-x-3">
                   <Database className="w-5 h-5 text-blue-500" />
                   <h3 className="text-lg font-bold text-white uppercase tracking-tight">Dataset Samples</h3>
                 </div>
                 <span className="px-3 py-1 bg-slate-800 rounded-full text-[10px] text-slate-500 font-bold uppercase">BraTS 2020 Explorer</span>
               </div>
               
               <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                 {[0, 2, 5, 8, 12].map((idx) => (
                   <button
                     key={idx}
                     onClick={() => handleLoadDatasetSample(idx)}
                     className="bg-slate-800/40 hover:bg-blue-900/20 border border-slate-700/50 hover:border-blue-500/50 p-4 rounded-xl transition-all text-center group"
                   >
                     <p className="text-xs font-bold text-slate-400 group-hover:text-blue-400 mb-2">Sample #{idx + 1}</p>
                     <div className="text-[9px] font-bold text-slate-600 uppercase tracking-widest group-hover:text-blue-500">Run Inference</div>
                   </button>
                 ))}
               </div>
            </section>
          </div>
        ) : (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-5 duration-700">
            <div className="flex justify-between items-center">
              <button
                onClick={handleReset}
                className="flex items-center space-x-2 px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors border border-slate-700"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Return to Dashboard</span>
              </button>
              
              <div className="px-4 py-1.5 bg-blue-600/10 border border-blue-600/20 rounded-full">
                 <span className="text-[10px] font-bold text-blue-500 uppercase tracking-widest">Inference Complete</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
              <div className="lg:col-span-3 space-y-6">
                <div className="flex items-center space-x-2 border-b border-slate-800 pb-4">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <h3 className="text-xl font-bold text-white uppercase italic tracking-tight">Imaging Results</h3>
                </div>
                
                {/* User requested snippet-based section */}
                <div className="space-y-4">
                  <SegmentationDisplay
                    originalImage={result?.original_base64 ?? null}
                    maskImage={result?.mask_base64 ?? null}
                    overlayImage={result?.overlay_base64 ?? null}
                    isLoading={loading}
                    hasSubregions={result?.has_subregions}
                  />
                </div>
              </div>

              <div className="lg:col-span-2 space-y-8">
                <div className="space-y-6">
                  <div className="flex items-center space-x-2 border-b border-slate-800 pb-4">
                    <ShieldAlert className="w-5 h-5 text-rose-500" />
                    <h3 className="text-xl font-bold text-white uppercase italic tracking-tight">Survival Metrics</h3>
                  </div>
                  <PredictionCard 
                    probability={prediction?.probability ?? null} 
                    days={prediction?.days ?? null} 
                    isLoading={false} 
                  />
                </div>

                {/* Technical Overview Section */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                  <h3 className="text-sm font-bold mb-4 flex items-center space-x-2 text-slate-400 uppercase tracking-widest">
                    <Cpu className="w-4 h-4" />
                    <span>System Insights</span>
                  </h3>
                  <div className="space-y-4 text-xs font-medium">
                    <div className="flex justify-between border-b border-slate-800/50 pb-2">
                      <span className="text-slate-500">Processing Engine</span>
                      <span className="text-slate-300">PyTorch U-Net</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800/50 pb-2">
                      <span className="text-slate-500">Dataset Origin</span>
                      <span className="text-slate-300">BraTS 2020</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800/50 pb-2">
                       <span className="text-slate-500">Resolution</span>
                       <span className="text-slate-300">128x128px (Grayscale)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && !loading && !isDatasetLoading && (
          <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-6 rounded-2xl text-center max-w-md mx-auto">
            <p className="font-bold mb-1">System Error Observed</p>
            <p className="text-sm opacity-80">{error}</p>
          </div>
        )}
      </main>

      {/* Footer Legal/Clinical Disclaimer */}
      <footer className="max-w-3xl mx-auto mt-32 text-center border-t border-slate-800 pt-12 px-4">
        <p className="text-xs text-slate-500 leading-relaxed max-w-2xl mx-auto font-medium">
          🔬 SHIELD AI: INSTITUTIONAL RESEARCH PROTOTYPE. <br className="mt-1" />
          This system is intended for demonstration purposes only. Not for clinical diagnosis or prescription. 
        </p>
      </footer>
    </div>
  );
}

export default App;
