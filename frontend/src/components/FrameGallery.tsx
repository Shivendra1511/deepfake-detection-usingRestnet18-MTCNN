import React, { useState } from 'react';
import { FrameInfo, getMediaUrl } from '../services/api';
import { Film, UserCheck, UserX, X } from 'lucide-react';

interface FrameGalleryProps {
  frames: FrameInfo[];
}

export const FrameGallery: React.FC<FrameGalleryProps> = ({ frames }) => {
  const [selectedFrame, setSelectedFrame] = useState<FrameInfo | null>(null);
  const [filterFaceOnly, setFilterFaceOnly] = useState<boolean>(false);

  if (!frames || frames.length === 0) {
    return null;
  }

  const displayedFrames = filterFaceOnly ? frames.filter((f) => f.face_detected) : frames;

  return (
    <div className="card frame-gallery-card mt-6">
      <div className="gallery-header">
        <div>
          <h3 className="gallery-title flex items-center">
            <Film size={20} className="mr-2 text-primary" />
            Video Frame Analysis Gallery
          </h3>
          <p className="text-muted text-sm">
            Showing {displayedFrames.length} of {frames.length} extracted video frames
          </p>
        </div>

        <div className="gallery-filter">
          <label className="filter-checkbox-label">
            <input
              type="checkbox"
              checked={filterFaceOnly}
              onChange={(e) => setFilterFaceOnly(e.target.checked)}
            />
            <span>Faces detected only</span>
          </label>
        </div>
      </div>

      <div className="frame-grid">
        {displayedFrames.map((frame) => {
          const scorePercent = frame.score !== null && frame.score !== undefined ? `${(frame.score * 100).toFixed(1)}%` : 'N/A';
          const label = frame.label || 'N/A';

          return (
            <div
              key={frame.frame_index}
              className={`frame-card ${frame.face_detected ? 'has-face' : 'no-face'}`}
              onClick={() => setSelectedFrame(frame)}
            >
              <div className="frame-image-wrapper">
                <img
                  src={getMediaUrl(frame.thumbnail_url || frame.frame_url)}
                  alt={`Frame ${frame.frame_index}`}
                  loading="lazy"
                  className="frame-thumbnail"
                />
                <span className="frame-timestamp-badge">#{frame.frame_index} • {frame.timestamp_sec}s</span>
              </div>

              <div className="frame-card-footer">
                <div className="frame-face-status">
                  {frame.face_detected ? (
                    <span className="status-tag status-tag-face" title="Face detected and processed">
                      <UserCheck size={12} className="mr-1" /> Face
                    </span>
                  ) : (
                    <span className="status-tag status-tag-noface" title="No face found on this frame">
                      <UserX size={12} className="mr-1" /> No Face
                    </span>
                  )}
                </div>

                {frame.face_detected && (
                  <div className="frame-score-box">
                    <span className="frame-score-val mono">{scorePercent}</span>
                    <span className={`frame-label-tag label-${label.toLowerCase()}`}>{label}</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Frame Preview Modal */}
      {selectedFrame && (
        <div className="modal-backdrop" onClick={() => setSelectedFrame(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h4>Frame #{selectedFrame.frame_index} Details ({selectedFrame.timestamp_sec}s)</h4>
              <button className="close-btn" onClick={() => setSelectedFrame(null)}>
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              <img
                src={getMediaUrl(selectedFrame.frame_url)}
                alt={`Full Frame ${selectedFrame.frame_index}`}
                className="modal-preview-img"
              />
              <div className="modal-meta-row mt-4">
                <div>
                  <strong>Face Detected:</strong> {selectedFrame.face_detected ? 'Yes (MTCNN)' : 'No'}
                </div>
                {selectedFrame.face_detected && (
                  <>
                    <div>
                      <strong>Fake Probability:</strong> {((selectedFrame.score ?? 0) * 100).toFixed(2)}%
                    </div>
                    <div>
                      <strong>Frame Label:</strong>{' '}
                      <span className={`frame-label-tag label-${selectedFrame.label?.toLowerCase()}`}>
                        {selectedFrame.label}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
