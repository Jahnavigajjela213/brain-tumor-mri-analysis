import { Layers, Maximize2, ScanEye } from 'lucide-react';
import ReactCompareImage from 'react-compare-image';

interface SegmentationDisplayProps {
  originalImage: string | null;
  maskImage: string | null;
  overlayImage: string | null;
  isLoading: boolean;
  hasSubregions?: boolean;
}

const SegmentationDisplay: React.FC<SegmentationDisplayProps> = ({ 
  originalImage, 
  maskImage, 
  overlayImage,
  isLoading,
  hasSubregions
}) => {
  if (!originalImage && !isLoading) return null;

  return (
    <div className="w-full space-y-8">
      {/* Main Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Interactive Comparison Slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
              <ScanEye className="w-4 h-4 text-blue-400" />
              Interactive Clinical Review (Original vs Overlay)
            </h3>
            <span className="text-[10px] bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded-full border border-blue-500/20">
              SLIDE TO COMPARE
            </span>
          </div>
          
          <div className="relative rounded-2xl overflow-hidden border border-slate-700/50 bg-slate-950 aspect-square shadow-2xl">
            {originalImage && overlayImage ? (
              <ReactCompareImage
                leftImage={`data:image/png;base64,${originalImage}`}
                rightImage={`data:image/png;base64,${overlayImage}`}
                sliderLineColor="#3b82f6"
                handleSize={40}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-slate-600">
                {isLoading ? (
                  <div className="animate-pulse flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-full border-4 border-blue-500/20 border-t-blue-500 animate-spin" />
                    <span className="text-xs font-medium">Processing MRI...</span>
                  </div>
                ) : (
                  "Waiting for Scan Upload..."
                )}
              </div>
            )}
          </div>
        </div>

        {/* Technical Decomposition (Subregions) */}
        <div className="space-y-6">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
              <Layers className="w-4 h-4 text-pink-400" />
              Binary Segmentation Mask (WT/TC/ET)
            </h3>
            <div className="relative rounded-2xl overflow-hidden border border-slate-700/50 aspect-square bg-slate-900 group">
              {maskImage ? (
                <img
                  src={`data:image/png;base64,${maskImage}`}
                  alt="Tumor Mask"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-600 bg-slate-950/50">
                  Binary View Offline
                </div>
              )}
              <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="p-2 bg-black/60 rounded-lg backdrop-blur-md text-white border border-white/10 hover:bg-black/80">
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {hasSubregions && (
        <div className="bg-slate-900/40 rounded-2xl p-6 border border-slate-800/50 backdrop-blur-sm">
          <h4 className="text-[11px] font-black text-slate-500 uppercase tracking-[0.3em] mb-4">BraTS 2020 Legend</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3 bg-slate-800/30 p-3 rounded-xl border border-slate-700/30">
              <div className="w-4 h-4 rounded-md bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]"></div>
              <div>
                <p className="text-xs font-bold text-slate-200">Whole Tumor (WT)</p>
                <p className="text-[9px] text-slate-500 uppercase font-medium">Edema + Core</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 bg-slate-800/30 p-3 rounded-xl border border-slate-700/30">
              <div className="w-4 h-4 rounded-md bg-yellow-400 shadow-[0_0_10px_rgba(250,204,21,0.3)]"></div>
              <div>
                <p className="text-xs font-bold text-slate-200">Tumor Core (TC)</p>
                <p className="text-[9px] text-slate-500 uppercase font-medium">Non-Enhancing</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 bg-slate-800/30 p-3 rounded-xl border border-slate-700/30">
              <div className="w-4 h-4 rounded-md bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.3)]"></div>
              <div>
                <p className="text-xs font-bold text-slate-200">Enhancing (ET)</p>
                <p className="text-[9px] text-slate-500 uppercase font-medium">Active Region</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SegmentationDisplay;
