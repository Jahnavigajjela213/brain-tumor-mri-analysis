import React, { useRef, useState } from 'react';
import { Upload, X, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface FileUploadProps {
  onUpload: (file: File) => void;
  onClear: () => void;
  isLoading: boolean;
  uploadedFile: File | null;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUpload, onClear, isLoading, uploadedFile }) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div 
        className={`relative glass-card rounded-2xl p-8 border-2 border-dashed transition-all duration-300 ${
          dragActive ? 'border-primary/50 bg-primary/5' : 'border-slate-700/50 hover:border-slate-600'
        } ${uploadedFile ? 'bg-secondary/5 border-secondary/20' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept="image/*"
          onChange={handleChange}
          disabled={isLoading}
        />

        <AnimatePresence mode="wait">
          {!uploadedFile ? (
            <motion.div 
              key="upload"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center justify-center space-y-4"
            >
              <div className="p-4 bg-primary/10 rounded-full text-primary">
                {isLoading ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Upload className="w-8 h-8" />
                  </motion.div>
                ) : (
                  <Upload className="w-8 h-8" />
                )}
              </div>
              <div className="text-center">
                <p className="text-lg font-medium">Drag & Drop MRI Scan</p>
                <p className="text-sm text-slate-400 mt-1">Supports PNG, JPG, JPEG (T1, T2, FLAIR)</p>
              </div>
              <button
                onClick={onButtonClick}
                disabled={isLoading}
                className="px-6 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-medium transition-all transform active:scale-95 disabled:opacity-50 disabled:active:scale-100 glow-primary"
              >
                {isLoading ? 'Processing...' : 'Browse Files'}
              </button>
            </motion.div>
          ) : (
            <motion.div 
              key="complete"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center justify-between bg-slate-800/50 p-4 rounded-xl border border-secondary/20"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-secondary/10 rounded-lg text-secondary">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
                <div>
                  <p className="font-medium text-slate-200 truncate max-w-[200px]">{uploadedFile.name}</p>
                  <p className="text-xs text-slate-400">{(uploadedFile.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
              <button 
                onClick={onClear}
                className="p-2 hover:bg-slate-700/50 rounded-full transition-colors"
                title="Remove File"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default FileUpload;
