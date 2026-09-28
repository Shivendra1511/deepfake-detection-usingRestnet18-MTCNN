import React, { useState } from 'react';
import { Search, Loader2, AlertCircle } from 'lucide-react';
import { getJobResult, JobResultResponse } from '../services/api';
import { ResultCard } from '../components/ResultCard';
import { FrameGallery } from '../components/FrameGallery';
import { JobStatus } from '../components/JobStatus';

export const Results: React.FC = () => {
  const [searchJobId, setSearchJobId] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<JobResultResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanId = searchJobId.trim();
    if (!cleanId) return;

    setIsLoading(true);
    setErrorMessage(null);
    setResult(null);

    try {
      const data = await getJobResult(cleanId);
      setResult(data);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Job ID not found in database.';
      setErrorMessage(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="section-header text-center mb-6">
        <h1>Lookup Analysis Results</h1>
        <p className="text-muted">
          Enter a previously generated Job ID to retrieve saved analysis results from MongoDB
        </p>
      </div>

      <div className="card max-w-2xl mx-auto mb-8">
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-wrapper">
            <Search size={20} className="search-icon" />
            <input
              type="text"
              className="search-input mono"
              placeholder="Paste Job ID (e.g. 9b1deb4d...)"
              value={searchJobId}
              onChange={(e) => setSearchJobId(e.target.value)}
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={isLoading || !searchJobId.trim()}>
            {isLoading ? <Loader2 size={16} className="spinner mr-1" /> : 'Fetch Result'}
          </button>
        </form>

        {errorMessage && (
          <div className="alert alert-error mt-4">
            <AlertCircle size={20} className="alert-icon" />
            <div className="alert-text">{errorMessage}</div>
          </div>
        )}
      </div>

      {result && (
        <div className="results-wrapper">
          {(result.status === 'queued' || result.status === 'processing') && (
            <JobStatus
              jobId={result.job_id}
              onCompleted={(completed) => setResult(completed)}
              onReset={() => setResult(null)}
            />
          )}

          {result.status === 'failed' && (
            <div className="alert alert-error">
              <AlertCircle size={20} className="alert-icon" />
              <div className="alert-text">
                Job failed{result.error_message ? `: ${result.error_message}` : '.'}
              </div>
            </div>
          )}

          {result.status === 'completed' && (
            <>
              <ResultCard result={result} onReset={() => setResult(null)} />
              {result.type === 'video' && <FrameGallery frames={result.frames} />}
            </>
          )}
        </div>
      )}
    </div>
  );
};
