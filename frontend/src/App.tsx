import React, { useState, useEffect } from 'react';
import { Home } from './pages/Home';
import { Results } from './pages/Results';
import { About } from './pages/About';
import { getSystemHealth, HealthResponse } from './services/api';
import { ScanFace, UploadCloud, Search, Info, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'home' | 'results' | 'about'>('home');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await getSystemHealth();
        setHealth(data);
        setIsBackendOnline(true);
      } catch (err) {
        setIsBackendOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="navbar-inner">
          <div className="nav-brand" onClick={() => setActiveTab('home')}>
            <ScanFace className="brand-icon" size={28} />
            <span>Deepfake Detection (ResNet18-MTCNN)</span>
          </div>

          <div className="nav-links">
            <button
              className={`nav-item ${activeTab === 'home' ? 'active' : ''}`}
              onClick={() => setActiveTab('home')}
            >
              <UploadCloud size={16} />
              <span>Detect</span>
            </button>

            <button
              className={`nav-item ${activeTab === 'results' ? 'active' : ''}`}
              onClick={() => setActiveTab('results')}
            >
              <Search size={16} />
              <span>Lookup Result</span>
            </button>

            <button
              className={`nav-item ${activeTab === 'about' ? 'active' : ''}`}
              onClick={() => setActiveTab('about')}
            >
              <Info size={16} />
              <span>About Architecture</span>
            </button>
          </div>

          {/* System Health Badge */}
          <div className="health-status-badge">
            {isBackendOnline === true ? (
              health?.model_file_present ? (
                <span className="badge badge-success" title="FastAPI, MongoDB, and Model detector.pt active">
                  <CheckCircle2 size={12} className="mr-1" /> Model Ready
                </span>
              ) : (
                <span className="badge badge-warning" title="FastAPI online. Model detector.pt needs to be trained and placed in backend/data/">
                  <AlertTriangle size={12} className="mr-1" /> Model Pending Training
                </span>
              )
            ) : isBackendOnline === false ? (
              <span className="badge badge-danger" title="Cannot connect to FastAPI at http://127.0.0.1:8000">
                <XCircle size={12} className="mr-1" /> Backend Offline
              </span>
            ) : null}
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'home' && <Home />}
        {activeTab === 'results' && <Results />}
        {activeTab === 'about' && <About />}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          Deepfake Detection in Image and Video Using ResNet18-MTCNN Pipeline • BTech Final Year Project
        </p>
        <p className="text-xs text-muted mt-1">
          Local Monolith: React + Vite + TypeScript + FastAPI + PyTorch + OpenCV + MongoDB
        </p>
      </footer>
    </div>
  );
};

export default App;
