import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export type JobStatusType = 'queued' | 'processing' | 'completed' | 'failed';
export type MediaType = 'image' | 'video';
export type PredictionLabel = 'REAL' | 'UNCERTAIN' | 'FAKE';

export interface FrameInfo {
  frame_index: number;
  timestamp_sec: number;
  frame_url: string;
  thumbnail_url?: string | null;
  face_detected: boolean;
  score?: number | null;
  label?: PredictionLabel | null;
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  type: MediaType;
  status: string;
}

export interface JobStatusResponse {
  job_id: string;
  filename: string;
  type: MediaType;
  status: JobStatusType;
  score?: number | null;
  label?: PredictionLabel | null;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobResultResponse {
  job_id: string;
  filename: string;
  type: MediaType;
  status: JobStatusType;
  score?: number | null;
  label?: PredictionLabel | null;
  frames: FrameInfo[];
  frame_scores: number[];
  processed_frame_count: number;
  detected_face_count: number;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface HealthResponse {
  status: string;
  mongodb_connected: boolean;
  model_file_present: boolean;
  model_path: string;
}

/**
 * Upload an image or video file to start background inference job
 */
export async function uploadMedia(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    // No timeout for uploads: large videos can take longer than the default 30s to send
    timeout: 0,
  });
  return response.data;
}

/**
 * Poll job lifecycle status
 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await apiClient.get<JobStatusResponse>(`/jobs/${jobId}`);
  return response.data;
}

/**
 * Fetch full completed result payload for a job
 */
export async function getJobResult(jobId: string): Promise<JobResultResponse> {
  const response = await apiClient.get<JobResultResponse>(`/jobs/results/${jobId}`);
  return response.data;
}

/**
 * Check backend system health and model availability
 */
export async function getSystemHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
}

/**
 * Parse a backend timestamp. Older jobs were stored as naive UTC ISO strings
 * (no "Z"/offset), which browsers would otherwise treat as local time.
 */
export function parseServerDate(value: string): Date {
  const hasTimezone = /([zZ]|[+-]\d{2}:?\d{2})$/.test(value);
  return new Date(hasTimezone ? value : `${value}Z`);
}

/**
 * Build absolute URL for frame previews and uploaded files
 */
export function getMediaUrl(relativeUrl: string): string {
  if (!relativeUrl) return '';
  if (relativeUrl.startsWith('http://') || relativeUrl.startsWith('https://')) {
    return relativeUrl;
  }
  const cleanPath = relativeUrl.startsWith('/') ? relativeUrl : `/${relativeUrl}`;
  return `${API_BASE_URL}${cleanPath}`;
}
