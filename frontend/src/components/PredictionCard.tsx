import React from 'react';
import { motion } from 'framer-motion';
import { Calendar, Activity, TrendingUp, Info } from 'lucide-react';

interface PredictionCardProps {
  probability: number | null;
  days: number | null;
  isLoading: boolean;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ probability, days, isLoading }) => {
  if (!isLoading && probability === null) return null;

  const getRiskLevel = (p: number) => {
    if (p > 0.7) return { text: 'High Risk', color: 'text-rose-500', bg: 'bg-rose-500/10' };
    if (p >= 0.4) return { text: 'Moderate Risk', color: 'text-amber-500', bg: 'bg-amber-500/10' };
    return { text: 'Low Risk', color: 'text-emerald-500', bg: 'bg-emerald-500/10' };
  };

  const risk = probability !== null ? getRiskLevel(probability) : null;

  return (
    <div className="w-full max-w-2xl mx-auto mt-8">
      <div className="glass-card rounded-2xl p-6 border-slate-700/50 relative overflow-hidden">
        {isLoading ? (
          <div className="flex flex-col items-center py-8 space-y-4">
            <Activity className="w-10 h-10 text-primary animate-pulse" />
            <p className="text-slate-400">Analyzing Survival Probability...</p>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="flex flex-col space-y-2">
              <div className="flex items-center space-x-2 text-slate-400 text-sm">
                <TrendingUp className="w-4 h-4" />
                <span>Survival Probability:</span>
              </div>
              <div className="flex items-end space-x-3">
                <span className="text-4xl font-bold text-white leading-none">
                  {(probability! * 100).toFixed(1)}%
                </span>
              </div>
              <div className="pt-4 pb-2">
                <p className="text-xs text-slate-400 font-medium mb-1 uppercase tracking-wider">Risk Category</p>
                {risk && (
                  <div className="flex flex-col">
                    <span className={`text-lg font-bold ${risk.color}`}>
                      {risk.text}
                    </span>
                    <span className="text-[10px] text-slate-500 font-medium italic">
                      (Based on predicted probability range)
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-col space-y-2">
              <div className="flex items-center space-x-2 text-slate-400 text-sm">
                <Calendar className="w-4 h-4" />
                <span>Estimated Survival:</span>
              </div>
              <div className="flex items-baseline space-x-1">
                <span className="text-4xl font-bold text-white leading-none">{days}</span>
                <span className="text-xl text-slate-400 font-medium">days</span>
              </div>
              <p className="text-[10px] text-slate-500 italic mt-auto font-medium">
                * Personalized estimate based on MRI features
              </p>
            </div>

            <div className="md:col-span-2 flex items-start space-x-3 p-3 bg-primary/5 rounded-xl border border-primary/10 mt-2">
              <Info className="w-5 h-5 text-primary shrink-0 mt-0.5" />
              <p className="text-xs text-slate-400 leading-relaxed font-medium">
                This lightweight AI model analyzes MRI features such as tumor patterns and spatial regions to generate segmentation and survival predictions.
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default PredictionCard;
