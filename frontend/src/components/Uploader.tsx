import React, { useState, useRef } from 'react';
import { Upload, FileImage, Video, AlertCircle, ArrowRight } from 'lucide-react';
import { uploadMedia } from '../services/api';

interface UploaderProps {
  onUploadSuccess: (jobId: string) => void;
}

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.mp4', '.mov', '.avi'];

export const Uploader: React.FC<UploaderProps> = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMessage(
        `Unsupported format (${ext}). Supported formats: Images (.jpg, .jpeg, .png) or Videos (.mp4, .mov, .avi)`
      );
      setSelectedFile(null);
      return false;
    }
    setErrorMessage(null);
    return true;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMessage(null);

    try {
      const response = await uploadMedia(selectedFile);
      onUploadSuccess(response.job_id);
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Upload failed. Please ensure the backend server is running.';
      setErrorMessage(detail);
    } finally {
      setIsUploading(false);
    }
  };

  const isVideo = selectedFile && ['.mp4', '.mov', '.avi'].some(ext => selectedFile.name.toLowerCase().endsWith(ext));

  return (
    <div className="card uploader-container">
      <div
        className={`drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.mp4,.mov,.avi"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        <div className="drop-zone-content">
          <div className="icon-bubble">
            {selectedFile ? (
              isVideo ? <Video size={36} className="text-primary" /> : <FileImage size={36} className="text-primary" />
            ) : (
              <Upload size={36} className="text-muted" />
            )}
          </div>

          {selectedFile ? (
            <div className="file-info-preview">
              <p className="selected-filename">{selectedFile.name}</p>
              <div className="file-badge-row">
                <span className="badge badge-subtle">
                  {isVideo ? 'Video' : 'Image'}
                </span>
                <span className="badge badge-subtle">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </span>
              </div>
              <p className="click-to-change">Click or drag a new file to replace</p>
            </div>
          ) : (
            <>
              <h3 className="drop-title">Drag & drop your media here</h3>
              <p className="drop-subtitle">or click to browse your files from your computer</p>
              <div className="supported-formats-pills">
                <span>.JPG</span>
                <span>.PNG</span>
                <span>.MP4</span>
                <span>.MOV</span>
                <span>.AVI</span>
              </div>
            </>
          )}
        </div>
      </div>

      {errorMessage && (
        <div className="alert alert-error">
          <AlertCircle size={20} className="alert-icon" />
          <div className="alert-text">{errorMessage}</div>
        </div>
      )}

      {selectedFile && (
        <div className="uploader-action-bar">
          <button
            className="btn btn-primary btn-large"
            disabled={isUploading}
            onClick={(e) => {
              e.stopPropagation();
              handleUpload();
            }}
          >
            {isUploading ? (
              <>
                <span className="spinner-border mr-2" />
                Uploading & Scheduling Job...
              </>
            ) : (
              <>
                Analyze Media
                <ArrowRight size={18} className="ml-2" />
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
