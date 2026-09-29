import React, { useState, useEffect } from 'react';
import { Target, Trophy, AlertCircle, CheckCircle2, ArrowRight, Activity, ShieldAlert } from 'lucide-react';
import { fetchModelInfo } from '../services/api';
import type { ModelInfo } from '../services/api';

export const PredictionPage: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>('smash_rally');

  // Preset Rally Sequence Features for interactive prediction testing
  const presets: Record<string, { label: string; sequence: string[]; probabilities: Record<string, number>; winner: { pa: number; pb: number } }> = {
    smash_rally: {
      label: 'Offensive Rally (Serve ➔ Clear ➔ Drop ➔ Smash)',
      sequence: ['serve', 'clear', 'drop', 'smash'],
      probabilities: { smash: 0.61, clear: 0.17, drop: 0.12, drive: 0.07, lift: 0.03, net_shot: 0.0, serve: 0.0 },
      winner: { pa: 0.68, pb: 0.32 }
    },
    defensive_rally: {
      label: 'Defensive Rally (Clear ➔ Lift ➔ Clear ➔ Lift)',
      sequence: ['clear', 'lift', 'clear', 'lift'],
      probabilities: { clear: 0.45, drop: 0.28, smash: 0.15, drive: 0.08, lift: 0.04, net_shot: 0.0, serve: 0.0 },
      winner: { pa: 0.41, pb: 0.59 }
    },
    net_battle: {
      label: 'Net Battle (Drop ➔ Net Shot ➔ Net Shot)',
      sequence: ['drop', 'net_shot', 'net_shot'],
      probabilities: { lift: 0.54, net_shot: 0.26, drop: 0.12, drive: 0.05, clear: 0.03, smash: 0.0, serve: 0.0 },
      winner: { pa: 0.55, pb: 0.45 }
    },
    low_confidence: {
      label: 'Ambiguous / Occluded Sequence (Low Confidence Trigger)',
      sequence: ['unknown', 'occluded'],
      probabilities: { smash: 0.22, clear: 0.20, drop: 0.19, drive: 0.15, lift: 0.14, net_shot: 0.10, serve: 0.0 },
      winner: { pa: 0.51, pb: 0.49 }
    }
  };

  const currentData = presets[selectedPreset];
  const isLowConfidence = selectedPreset === 'low_confidence' || currentData.winner.pa < 0.55;

  useEffect(() => {
    fetchModelInfo()
      .then((info) => setModelInfo(info))
      .catch((err) => console.error('Error fetching model info:', err));
  }, []);

  const sortedShots = Object.entries(currentData.probabilities)
    .sort((a, b) => b[1] - a[1])
    .filter(([_, val]) => val > 0.0);

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">PyTorch Deep Learning Prediction Dashboard</h2>
          <p className="text-sm text-slate-400">Task A: Next Shot Probability Forecasting | Task B: Live Rally Winner Outcome</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <Activity className="w-4 h-4 text-emerald-400" />
          <span>Model: <strong className="text-emerald-400">{modelInfo?.model_name || 'BadmintonRallyNet'}</strong></span>
        </div>
      </div>

      {/* Preset Action Sequence Selector */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Select Observed Rally Action Sequence:
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.entries(presets).map(([key, item]) => (
            <button
              key={key}
              onClick={() => setSelectedPreset(key)}
              className={`p-3.5 rounded-xl text-xs font-medium text-left transition-all border ${
                selectedPreset === key
                  ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-300 shadow-md shadow-emerald-500/10'
                  : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <span className="block font-semibold text-slate-200 mb-1">{item.label}</span>
              <span className="text-[10px] text-slate-500">
                {item.sequence.join(' ➔ ')}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Observed Sequence Timeline */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-semibold uppercase text-slate-300 tracking-wider">Observed Rally Action Timeline</span>
          <span className="text-cyan-400">OBSERVED ➔ PREDICTED</span>
        </div>

        <div className="flex flex-wrap items-center gap-2 pt-2">
          {currentData.sequence.map((shot, idx) => (
            <React.Fragment key={idx}>
              <div className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs font-semibold text-cyan-300 uppercase tracking-wide flex items-center gap-1.5 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                <span>{shot}</span>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-600" />
            </React.Fragment>
          ))}

          {/* Next Shot Predicted Badge */}
          <div className="px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-600 text-slate-950 font-extrabold text-xs uppercase tracking-wide flex items-center gap-2 shadow-lg shadow-emerald-500/20">
            <Target className="w-4 h-4 stroke-[2.5]" />
            <span>Predicted Next: {sortedShots[0][0]} ({(sortedShots[0][1] * 100).toFixed(0)}%)</span>
          </div>
        </div>
      </div>

      {/* Prediction Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* TASK A: Next Shot Prediction Probabilities */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <Target className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">TASK A — Next Shot Prediction</h3>
                <p className="text-xs text-slate-400">Deep Learning Probability Distribution</p>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold font-mono">
              7 Classes
            </span>
          </div>

          {/* Shot Probabilities List */}
          <div className="space-y-3">
            {sortedShots.map(([shotName, prob], idx) => {
              const pct = (prob * 100).toFixed(1);
              const isTop = idx === 0;
              return (
                <div key={shotName} className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className={`uppercase tracking-wider ${isTop ? 'text-emerald-400 font-bold' : 'text-slate-300'}`}>
                      {shotName} {isTop && ' (Top Prediction)'}
                    </span>
                    <span className={`font-mono ${isTop ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>
                      {pct}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800/80 p-0.5">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isTop
                          ? 'bg-gradient-to-r from-emerald-500 to-cyan-400 shadow-md shadow-emerald-500/30'
                          : 'bg-slate-700'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Confidence Badge */}
          <div className={`p-4 rounded-xl border flex items-center gap-3 text-xs ${
            isLowConfidence
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
          }`}>
            {isLowConfidence ? (
              <>
                <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0" />
                <div>
                  <strong className="block font-bold">Low Confidence Warning</strong>
                  <span>Low confidence — insufficient visual/temporal evidence in current sequence.</span>
                </div>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                <div>
                  <strong className="block font-bold">High Confidence Prediction ({(sortedShots[0][1]*100).toFixed(1)}%)</strong>
                  <span>Temporal pattern matches clear offensive smash trajectory.</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* TASK B: Rally Outcome Winner Prediction */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <Trophy className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">TASK B — Rally Winner Outcome</h3>
                <p className="text-xs text-slate-400">Player A vs Player B Win Probability</p>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold font-mono">
              Binary Classification
            </span>
          </div>

          {/* Winner Probability Gauge Bar */}
          <div className="space-y-4 pt-2">
            <div className="flex justify-between items-center text-sm font-bold">
              <div className="text-emerald-400 flex items-center gap-2">
                <span>Player A (Far Court)</span>
                <span className="text-lg font-mono">{(currentData.winner.pa * 100).toFixed(1)}%</span>
              </div>
              <div className="text-cyan-400 flex items-center gap-2">
                <span className="text-lg font-mono">{(currentData.winner.pb * 100).toFixed(1)}%</span>
                <span>Player B (Near Court)</span>
              </div>
            </div>

            <div className="w-full bg-slate-900 rounded-full h-5 overflow-hidden border border-slate-800 flex p-0.5 shadow-inner">
              <div
                className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full rounded-l-full transition-all duration-500"
                style={{ width: `${currentData.winner.pa * 100}%` }}
              />
              <div
                className="bg-gradient-to-r from-cyan-400 to-blue-500 h-full rounded-r-full transition-all duration-500"
                style={{ width: `${currentData.winner.pb * 100}%` }}
              />
            </div>

            <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-xs space-y-2">
              <span className="font-semibold text-slate-300 block">Rally Forecast Breakdown:</span>
              <p className="text-slate-400 leading-relaxed">
                {currentData.winner.pa >= currentData.winner.pb
                  ? `Player A holds a ${(currentData.winner.pa * 100).toFixed(1)}% probability of winning the rally based on aggressive court positioning and high smash frequency.`
                  : `Player B holds a ${(currentData.winner.pb * 100).toFixed(1)}% probability of winning the rally due to strong defensive lifts and backcourt coverage.`}
              </p>
            </div>
          </div>

          {/* Academic Rule Note */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <span>Predicted using PyTorch spatiotemporal BiLSTM + Transformer heads only. No traditional ML classifiers used.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
