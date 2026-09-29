import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './components/LandingPage';
import { VideoAnalysisPage } from './components/VideoAnalysisPage';
import { PredictionPage } from './components/PredictionPage';
import { AnalyticsPage } from './components/AnalyticsPage';
import { ModelInfoPage } from './components/ModelInfoPage';
import { fetchHealth } from './services/api';
import type { HealthStatus } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('landing');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data))
      .catch((err) => console.log('Backend health check note:', err));
  }, []);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-slate-950">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 pt-8">
        {activeTab === 'landing' && <LandingPage setActiveTab={setActiveTab} />}
        {activeTab === 'video' && <VideoAnalysisPage setActiveTab={setActiveTab} />}
        {activeTab === 'prediction' && <PredictionPage />}
        {activeTab === 'analytics' && <AnalyticsPage />}
        {activeTab === 'model' && <ModelInfoPage />}
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t border-slate-800/80 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 Badminton AI Analytics — Student Deep Learning Framework</p>
          <div className="flex items-center gap-4">
            <span className="text-slate-400">PyTorch 2.14</span>
            <span className="text-slate-400">FastAPI</span>
            <span className="text-slate-400">React + TS</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
