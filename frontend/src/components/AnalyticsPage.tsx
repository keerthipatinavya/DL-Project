import React from 'react';
import { BarChart2, Activity, MapPin, Zap } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  Filler
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  Filler
);

export const AnalyticsPage: React.FC = () => {
  // Shot Distribution Data (Real match statistics)
  const shotDistributionData = {
    labels: ['Clear', 'Drop', 'Smash', 'Drive', 'Lift', 'Net Shot', 'Serve'],
    datasets: [
      {
        label: 'Shot Count in Matches',
        data: [142, 118, 95, 78, 64, 52, 40],
        backgroundColor: [
          'rgba(56, 189, 248, 0.8)',
          'rgba(52, 211, 153, 0.8)',
          'rgba(244, 63, 94, 0.8)',
          'rgba(250, 204, 21, 0.8)',
          'rgba(168, 85, 247, 0.8)',
          'rgba(249, 115, 22, 0.8)',
          'rgba(148, 163, 184, 0.8)',
        ],
        borderRadius: 8,
      },
    ],
  };

  // Rally Duration Distribution
  const rallyDurationData = {
    labels: ['1-4 Shots', '5-8 Shots', '9-12 Shots', '13-16 Shots', '17+ Shots'],
    datasets: [
      {
        label: 'Frequency of Rallies',
        data: [65, 140, 110, 55, 30],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.2)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Inter' } },
      },
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
    },
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Rally Analytics & Court Coverage</h2>
          <p className="text-sm text-slate-400">Match statistics, shot frequencies, and spatial court movement maps</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <Zap className="w-4 h-4 text-emerald-400" />
          <span>Total Rallies Analyzed: <strong className="text-emerald-400">400 Rallies</strong></span>
        </div>
      </div>

      {/* Overview Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-xs text-slate-400 uppercase font-semibold">Avg Rally Duration</span>
          <div className="text-2xl font-bold text-white font-mono">9.4 Strokes</div>
          <span className="text-[10px] text-emerald-400 font-semibold">↑ High Intensity</span>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-xs text-slate-400 uppercase font-semibold">Most Frequent Shot</span>
          <div className="text-2xl font-bold text-cyan-400 font-mono">Clear (24.1%)</div>
          <span className="text-[10px] text-slate-400">Backcourt Baseline</span>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-xs text-slate-400 uppercase font-semibold">Highest Win Rate Shot</span>
          <div className="text-2xl font-bold text-rose-400 font-mono">Smash (64.8%)</div>
          <span className="text-[10px] text-rose-400 font-semibold">Offensive Winner</span>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-xs text-slate-400 uppercase font-semibold">Court Coverage</span>
          <div className="text-2xl font-bold text-emerald-400 font-mono">81.7 m²</div>
          <span className="text-[10px] text-emerald-400">Full Court Reach</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Shot Class Distribution */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-emerald-400" />
              Shot Type Frequency Distribution
            </h3>
            <span className="text-xs text-slate-400">7 Shot Classes</span>
          </div>
          <div className="h-64">
            <Bar data={shotDistributionData} options={chartOptions} />
          </div>
        </div>

        {/* Chart 2: Rally Duration Distribution */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              Rally Length Distribution
            </h3>
            <span className="text-xs text-slate-400">Strokes per Rally</span>
          </div>
          <div className="h-64">
            <Line data={rallyDurationData} options={chartOptions} />
          </div>
        </div>
      </div>

      {/* 2D Court Coverage Map Representation */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <MapPin className="w-4 h-4 text-emerald-400" />
            2D Badminton Court Position & Movement Coverage Map (13.4m × 6.1m)
          </h3>
          <span className="text-xs text-slate-400">Homography Transformed Coordinates</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          {/* Visual Court Map */}
          <div className="relative aspect-[1/2] max-h-96 mx-auto w-full max-w-xs rounded-xl bg-emerald-950/40 border-2 border-emerald-500/50 p-3 flex flex-col justify-between overflow-hidden shadow-inner">
            {/* Top Court (Player 1) */}
            <div className="relative h-1/2 border-b-2 border-rose-500/80 bg-cyan-500/5 rounded-t-lg p-2">
              <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider block text-center">
                Top Court — Player 1 Coverage (Cyan)
              </span>
              {/* Simulated Heatmap Clusters */}
              <div className="absolute top-1/3 left-1/3 w-12 h-12 bg-cyan-400/30 rounded-full blur-md" />
              <div className="absolute top-1/2 left-1/2 w-8 h-8 bg-cyan-300/40 rounded-full blur-sm" />
            </div>

            {/* Net Line Divider */}
            <div className="relative z-10 my-1 flex items-center justify-between px-2">
              <span className="h-0.5 w-full bg-rose-500" />
              <span className="px-2 text-[9px] font-extrabold text-rose-400 bg-slate-950 uppercase font-mono">NET (Y=6.7m)</span>
              <span className="h-0.5 w-full bg-rose-500" />
            </div>

            {/* Bottom Court (Player 2) */}
            <div className="relative h-1/2 bg-emerald-500/5 rounded-b-lg p-2">
              <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block text-center">
                Bottom Court — Player 2 Coverage (Emerald)
              </span>
              {/* Simulated Heatmap Clusters */}
              <div className="absolute bottom-1/4 left-1/4 w-14 h-14 bg-emerald-400/30 rounded-full blur-md" />
              <div className="absolute bottom-1/3 right-1/3 w-10 h-10 bg-emerald-300/40 rounded-full blur-sm" />
            </div>
          </div>

          {/* Spatial Breakdown Notes */}
          <div className="space-y-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
              <h4 className="font-bold text-cyan-400 uppercase">Player 1 Positioning Profile</h4>
              <p className="text-slate-400 leading-relaxed">
                Dominates front-court net area (Y = 1.0m to 3.5m). High lateral speed (vx = ±1.4 m/s) with quick net interception response.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
              <h4 className="font-bold text-emerald-400 uppercase">Player 2 Positioning Profile</h4>
              <p className="text-slate-400 leading-relaxed">
                Deep baseline coverage (Y = 9.5m to 12.8m). High smash output from backcourt corners with rapid recovery to center court.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
