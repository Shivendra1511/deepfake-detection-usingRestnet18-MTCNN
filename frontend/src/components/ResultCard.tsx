import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, FileText, Calendar, Film, Image as ImageIcon } from 'lucide-react';
import { JobResultResponse } from '../services/api';

interface ResultCardProps {
  result: JobResultResponse;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, onReset }) => {
  const score = result.score ?? 0;
  const scorePercent = (score * 100).toFixed(1);
  const label = result.label || 'UNCERTAIN';

  const getVerdictTheme = () => {
    switch (label) {
      case 'REAL':
        return {
          badgeClass: 'verdict-real',
          icon: <ShieldCheck size={36} className="text-success" />,
          title: 'Authentic / Real',
          description: 'No significant indicators of synthetic or deepfake manipulation were detected in the facial features.',
        };
      case 'FAKE':
        return {
          badgeClass: 'verdict-fake',
          icon: <ShieldAlert size={36} className="text-danger" />,
          title: 'Deepfake / Manipulated',
          description: 'The neural network detected strong facial manipulation artifacts consistent with deepfake generation techniques.',
        };
      case 'UNCERTAIN':
      default:
        return {
          badgeClass: 'verdict-uncertain',
          icon: <AlertTriangle size={36} className="text-warning" />,
          title: 'Uncertain / Inconclusive',
          description: 'The prediction score falls within the borderline threshold (0.40 - 0.60). Manual verification is recommended.',
        };
    }
  };

  const theme = getVerdictTheme();

  return (
    <div className="card result-card">
      <div className={`verdict-header ${theme.badgeClass}`}>
        <div className="verdict-icon-container">{theme.icon}</div>
        <div className="verdict-text-container">
          <span className="verdict-tag">{label}</span>
          <h2 className="verdict-title">{theme.title}</h2>
          <p className="verdict-desc">{theme.description}</p>
        </div>
      </div>

      <div className="result-body">
        {/* Metric gauge */}
        <div className="metric-box">
          <div className="metric-header">
            <span className="metric-title">Deepfake Manipulation Probability</span>
            <span className="metric-value font-mono">{scorePercent}%</span>
          </div>

          <div className="probability-meter-track">
            <div
              className="probability-fill"
              style={{
                width: `${Math.min(100, Math.max(0, score * 100))}%`,
                backgroundColor:
                  label === 'REAL' ? 'var(--color-success)' : label === 'FAKE' ? 'var(--color-danger)' : 'var(--color-warning)',
              }}
            />
            {/* Threshold marker lines */}
            <div className="threshold-marker marker-lower" title="Lower Threshold (0.40)" />
            <div className="threshold-marker marker-upper" title="Upper Threshold (0.60)" />
          </div>

          <div className="meter-labels">
            <span>0% (Authentic)</span>
            <span className="threshold-label">0.40 (Real)</span>
            <span className="threshold-label">0.60 (Fake)</span>
            <span>100% (Manipulated)</span>
          </div>

          <p className="metric-disclaimer">
            <strong>Important distinction:</strong> This value represents the <em>predicted fake probability</em>{' '}
            output by the sigmoid activation function for this specific file, NOT overall model test accuracy.
          </p>
        </div>

        {/* Media details grid */}
        <div className="details-grid">
          <div className="detail-item">
            <span className="detail-label">
              <FileText size={14} className="mr-1 inline" /> File Name
            </span>
            <span className="detail-value text-truncate" title={result.filename}>
              {result.filename}
            </span>
          </div>

          <div className="detail-item">
            <span className="detail-label">
              {result.type === 'video' ? <Film size={14} className="mr-1 inline" /> : <ImageIcon size={14} className="mr-1 inline" />}
              Media Type
            </span>
            <span className="detail-value capitalize">{result.type}</span>
          </div>

          {result.type === 'video' && (
            <>
              <div className="detail-item">
                <span className="detail-label">Sampled Frames</span>
                <span className="detail-value">{result.processed_frame_count} frames</span>
              </div>

              <div className="detail-item">
                <span className="detail-label">Faces Detected</span>
                <span className="detail-value">{result.detected_face_count} frames</span>
              </div>
            </>
          )}

          <div className="detail-item">
            <span className="detail-label">
              <Calendar size={14} className="mr-1 inline" /> Analyzed On
            </span>
            <span className="detail-value">
              {new Date(result.updated_at).toLocaleString()}
            </span>
          </div>
        </div>

        <div className="result-action-bar">
          <button className="btn btn-secondary" onClick={onReset}>
            Analyze Another Media File
          </button>
        </div>
      </div>
    </div>
  );
};
