import React, { useState } from 'react';
import { Uploader } from '../components/Uploader';
import { JobStatus } from '../components/JobStatus';
import { ResultCard } from '../components/ResultCard';
import { FrameGallery } from '../components/FrameGallery';
import { JobResultResponse } from '../services/api';
import { ScanFace, Video, ShieldCheck, Cpu } from 'lucide-react';

export const Home: React.FC = () => {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);

  const handleUploadSuccess = (jobId: string) => {
    setActiveJobId(jobId);
    setJobResult(null);
  };

  const handleJobCompleted = (result: JobResultResponse) => {
    setJobResult(result);
  };

  const handleReset = () => {
    setActiveJobId(null);
    setJobResult(null);
  };

  return (
    <div className="page-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">
            <ScanFace size={16} className="mr-1 text-primary" />
            BTech Final Year Project • Local Computer Vision System
          </div>
          <h1 className="hero-title">
            Deepfake Detection in Image and Video
          </h1>
          <p className="hero-subtitle">
            Detect potentially manipulated images and videos using an ImageNet-pretrained{' '}
            <strong>ResNet18</strong> backbone and <strong>MTCNN</strong> facial bounding box extraction.
          </p>
        </div>

        {/* Feature highlight pills */}
        <div className="feature-grid">
          <div className="feature-card">
            <ScanFace className="feature-icon" size={24} />
            <div>
              <h4>MTCNN Face Localization</h4>
              <p>Automatic 20% padded cropping around largest facial region</p>
            </div>
          </div>

          <div className="feature-card">
            <Cpu className="feature-icon" size={24} />
            <div>
              <h4>ResNet18 Transfer Learning</h4>
              <p>Fine-tuned binary classifier outputting single calibrated logit</p>
            </div>
          </div>

          <div className="feature-card">
            <Video className="feature-icon" size={24} />
            <div>
              <h4>Video Frame Sampling</h4>
              <p>Evenly samples up to 64 frames with mean probability aggregation</p>
            </div>
          </div>

          <div className="feature-card">
            <ShieldCheck className="feature-icon" size={24} />
            <div>
              <h4>Three-State Decision</h4>
              <p>Configurable thresholds: REAL (&lt;0.4), UNCERTAIN, FAKE (&gt;0.6)</p>
            </div>
          </div>
        </div>
      </section>

      {/* Main Interactive Workspace */}
      <section className="workspace-section">
        {!activeJobId && !jobResult && (
          <div className="uploader-wrapper">
            <div className="section-header text-center mb-6">
              <h2>Upload Media for Analysis</h2>
              <p className="text-muted">Choose an image or video file from your computer to run deepfake inference</p>
            </div>
            <Uploader onUploadSuccess={handleUploadSuccess} />
          </div>
        )}

        {activeJobId && !jobResult && (
          <div className="status-wrapper">
            <div className="section-header text-center mb-6">
              <h2>Analyzing Media Content</h2>
              <p className="text-muted">Please wait while our local PyTorch and MTCNN pipeline processes your file</p>
            </div>
            <JobStatus
              jobId={activeJobId}
              onCompleted={handleJobCompleted}
              onReset={handleReset}
            />
          </div>
        )}

        {jobResult && (
          <div className="results-wrapper">
            <div className="section-header text-center mb-6">
              <h2>Deepfake Analysis Report</h2>
              <p className="text-muted">Review the classification verdict, probability score, and frame evidence below</p>
            </div>
            <ResultCard result={jobResult} onReset={handleReset} />
            {jobResult.type === 'video' && <FrameGallery frames={jobResult.frames} />}
          </div>
        )}
      </section>
    </div>
  );
};
