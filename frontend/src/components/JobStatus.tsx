import React, { useEffect, useRef, useState } from 'react';
import { Loader2, CheckCircle, AlertOctagon, Clock, Copy, Check } from 'lucide-react';
import { getJobStatus, getJobResult, JobStatusResponse, JobResultResponse } from '../services/api';

interface JobStatusProps {
  jobId: string;
  onCompleted: (result: JobResultResponse) => void;
  onReset: () => void;
}

export const JobStatus: React.FC<JobStatusProps> = ({ jobId, onCompleted, onReset }) => {
  const [statusData, setStatusData] = useState<JobStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  // Keep the latest callback in a ref so a new inline function from the parent
  // does not restart polling on every parent re-render.
  const onCompletedRef = useRef(onCompleted);
  useEffect(() => {
    onCompletedRef.current = onCompleted;
  }, [onCompleted]);

  useEffect(() => {
    let isMounted = true;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    let consecutiveErrors = 0;
    const POLL_INTERVAL_MS = 2000;
    const MAX_CONSECUTIVE_ERRORS = 5;

    const poll = async () => {
      try {
        const data = await getJobStatus(jobId);
        if (!isMounted) return;

        consecutiveErrors = 0;
        setErrorMessage(null);
        setStatusData(data);

        if (data.status === 'completed') {
          // Fetch complete payload including frame breakdown
          const fullResult = await getJobResult(jobId);
          if (isMounted) {
            onCompletedRef.current(fullResult);
          }
          return;
        }

        if (data.status === 'failed') {
          setErrorMessage(data.error_message || 'Inference processing failed.');
          return;
        }

        // Continue polling every 2 seconds while queued or processing
        timerId = setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err: any) {
        if (!isMounted) return;
        const msg = err.response?.data?.detail || err.message || 'Error checking job status.';
        consecutiveErrors += 1;

        // 404 means the job does not exist; anything else may be a temporary
        // network/server hiccup, so retry a few times before giving up.
        if (err.response?.status === 404 || consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          setErrorMessage(msg);
          return;
        }
        timerId = setTimeout(poll, POLL_INTERVAL_MS * consecutiveErrors);
      }
    };

    poll();

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [jobId]);

  const copyJobId = () => {
    navigator.clipboard.writeText(jobId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'queued':
        return <span className="status-badge status-queued"><Clock size={14} className="mr-1" /> Queued</span>;
      case 'processing':
        return <span className="status-badge status-processing"><Loader2 size={14} className="spinner mr-1" /> Processing</span>;
      case 'completed':
        return <span className="status-badge status-completed"><CheckCircle size={14} className="mr-1" /> Completed</span>;
      case 'failed':
        return <span className="status-badge status-failed"><AlertOctagon size={14} className="mr-1" /> Failed</span>;
      default:
        return <span className="status-badge">{status}</span>;
    }
  };

  return (
    <div className="card job-status-card">
      <div className="job-status-header">
        <div>
          <span className="label-subtle">Job ID</span>
          <div className="job-id-pill" onClick={copyJobId} title="Click to copy Job ID">
            <span className="mono">{jobId}</span>
            <button className="copy-btn" aria-label="Copy Job ID">
              {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
            </button>
          </div>
        </div>

        <div className="status-badge-container">
          <span className="label-subtle">Current Status</span>
          <div>{statusData ? getStatusBadge(statusData.status) : <span className="status-badge">Connecting...</span>}</div>
        </div>
      </div>

      {statusData?.status !== 'failed' && (
        <div className="progress-steps-row">
          <div className={`step-item ${statusData?.status ? 'active' : ''}`}>
            <div className="step-circle">1</div>
            <span className="step-label">Upload Saved</span>
          </div>
          <div className="step-connector" />
          <div className={`step-item ${statusData?.status === 'processing' || statusData?.status === 'completed' ? 'active' : ''}`}>
            <div className="step-circle">2</div>
            <span className="step-label">Face Detection (MTCNN)</span>
          </div>
          <div className="step-connector" />
          <div className={`step-item ${statusData?.status === 'completed' ? 'active' : ''}`}>
            <div className="step-circle">3</div>
            <span className="step-label">ResNet18 Inference</span>
          </div>
        </div>
      )}

      {statusData?.status === 'processing' && (
        <div className="processing-indicator">
          <Loader2 size={32} className="spinner text-primary mb-2" />
          <p className="font-medium">Deep learning pipeline is executing...</p>
          <p className="text-muted text-sm">
            {statusData.type === 'video'
              ? 'Extracting up to 64 frames, running MTCNN face detection on each frame, and computing probability scores.'
              : 'Detecting largest face bounding box, applying 20% padding, and generating fake probability.'}
          </p>
        </div>
      )}

      {errorMessage && (
        <div className="alert alert-error mt-4">
          <AlertOctagon size={24} className="alert-icon" />
          <div>
            <h4 className="font-semibold text-error mb-1">Processing Error</h4>
            <p className="text-sm">{errorMessage}</p>
            {errorMessage.includes('Trained model not found') && (
              <p className="text-xs text-muted mt-2">
                Tip: Train the model by running <code>python training/train.py</code> and verify that <code>backend/data/detector.pt</code> exists.
              </p>
            )}
            <button className="btn btn-secondary btn-sm mt-3" onClick={onReset}>
              Try Another File
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
