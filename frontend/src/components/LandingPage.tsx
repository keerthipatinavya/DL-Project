import React from 'react';
import { ArrowRight, Video, Target, ShieldCheck, Zap } from 'lucide-react';

interface LandingPageProps {
  setActiveTab: (tab: string) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ setActiveTab }) => {
  return (
    <div className="space-y-12 pb-12">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-b from-slate-900 via-slate-950 to-slate-900 border border-slate-800/80 p-8 md:p-12">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -mb-12 -ml-12 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-xs font-semibold text-emerald-400">
            <Zap className="w-3.5 h-3.5" />
            <span>Spatiotemporal Deep Learning Framework</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Deep Learning Based <br />
            <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
              Badminton Rally & Shot Outcome Prediction
            </span>
          </h1>

          <p className="text-lg text-slate-300 leading-relaxed">
            An end-to-end computer vision and PyTorch deep learning framework for sports video understanding.
            Analyzes player positioning, court homography, and high-velocity shuttle trajectories to forecast next-shot selection and rally winner probabilities.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={() => setActiveTab('video')}
              className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-600 text-slate-950 font-bold text-sm hover:opacity-95 transition shadow-lg shadow-emerald-500/25"
            >
              <Video className="w-4 h-4" />
              Upload Match Video & Analyze
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setActiveTab('prediction')}
              className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-200 font-semibold text-sm hover:bg-slate-800 transition"
            >
              <Target className="w-4 h-4 text-emerald-400" />
              Explore Prediction Dashboard
            </button>
          </div>
        </div>
      </section>

      {/* Spatiotemporal Deep Learning Architecture Flow */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">System Architecture & Pipeline</h2>
            <p className="text-sm text-slate-400">Pure Deep Learning Predictive Pipeline — No Traditional ML Classifiers</p>
          </div>
          <span className="text-xs px-3 py-1 bg-slate-900 border border-slate-800 rounded-lg text-slate-400">Academic Standard Compliant</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 font-bold text-sm">
              01
            </div>
            <h3 className="font-semibold text-white">Video Ingestion & Geometry</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Accepts MP4 broadcast video. Extracts frames, computes 3x3 Homography Matrix <strong className="text-slate-200">H</strong>, mapping pixel space to 2D court coordinates (13.4m × 6.1m).
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-bold text-sm">
              02
            </div>
            <h3 className="font-semibold text-white">Multi-Object Tracking</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              YOLOv8 + ByteTrack object detection for Player 1 (top court) and Player 2 (bottom court). Frame-differencing & parabolic curve fitting for shuttle trajectory tracking.
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-bold text-sm">
              03
            </div>
            <h3 className="font-semibold text-white">Spatiotemporal Encoder</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Feature projection layer + 2-layer Bidirectional LSTM + Multi-Head Self-Attention Transformer Encoder modeling temporal shot sequences.
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400 font-bold text-sm">
              04
            </div>
            <h3 className="font-semibold text-white">Multi-Task Prediction</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Dual prediction heads outputting next-shot probability distributions across 7 shot types and rally outcome winner forecasting with uncertainty scores.
            </p>
          </div>
        </div>
      </section>

      {/* Key Features Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Target className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-semibold text-white">Task A: Next-Shot Forecasting</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Predicts the probability of the immediate next stroke (Smash, Clear, Drop, Drive, Lift, Net Shot, Serve) based on observed player movements and shuttle state.
          </p>
        </div>

        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Zap className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-semibold text-white">Task B: Rally Winner Outcome</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Forecasts the live win probability for Player A vs Player B as the rally evolves in real-time, accounting for player positioning and shot pressure.
          </p>
        </div>

        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-semibold text-white">Observed vs Predicted Separation</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Maintains strict academic integrity: explicit visual distinction between empirical ground-truth observations (boxes/shuttle) and deep learning predictions.
          </p>
        </div>
      </section>

      {/* Tech Stack Banner */}
      <section className="glass-card p-6 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h4 className="text-sm font-semibold text-slate-200">Built with Modern Deep Learning & Full-Stack Tech:</h4>
          <p className="text-xs text-slate-400">PyTorch, BiLSTM, Transformer, OpenCV, YOLO, ByteTrack, FastAPI, React 19, TypeScript, Tailwind CSS</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-700 text-xs text-emerald-400 font-mono">PyTorch 2.14</span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-700 text-xs text-cyan-400 font-mono">FastAPI</span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-700 text-xs text-purple-400 font-mono">React TS</span>
        </div>
      </section>
    </div>
  );
};
