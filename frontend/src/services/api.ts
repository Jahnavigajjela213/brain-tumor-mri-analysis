import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export interface UploadResponse {
  upload_id: string;
  filename: string;
}

export interface SegmentResponse {
  mask_base64: string;
  original_base64: string;
  overlay_base64: string;
  mask_size: [number, number];
  has_subregions?: boolean;
}

export interface DatasetResponse {
  dataset_image: string;
  original_base64: string;
  mask_base64: string;
  overlay_base64: string;
  survival_probability: number;
  estimated_survival_days: number;
}

export interface PredictResponse {
  survival_probability: number;
  estimated_survival_days: number;
}

export const uploadMRI = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<UploadResponse>('/upload', formData);
  return response.data;
};

export const segmentMRI = async (uploadId: string): Promise<SegmentResponse> => {
  const formData = new FormData();
  formData.append('upload_id', uploadId);

  const response = await apiClient.post<SegmentResponse>('/segment', formData);
  return response.data;
};

export const predictSurvival = async (uploadId: string): Promise<PredictResponse> => {
  const formData = new FormData();
  formData.append('upload_id', uploadId);

  const response = await apiClient.post<PredictResponse>('/predict', formData);
  return response.data;
};

export const testDataset = async (index: number): Promise<DatasetResponse> => {
  const formData = new FormData();
  formData.append('index', index.toString());

  const response = await apiClient.post<DatasetResponse>('/test-dataset', formData);
  return response.data;
};
