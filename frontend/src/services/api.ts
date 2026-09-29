import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export interface HealthStatus {
  status: string;
  version: string;
  device: string;
  model_loaded: boolean;
}

export interface VideoUploadResult {
  video_id: string;
  filename: string;
  duration_sec: number;
  fps: number;
  width: number;
  height: number;
  file_path: string;
  message: string;
}

export interface NextShotPrediction {
  top_prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  is_low_confidence: boolean;
  confidence_message: string;
}

export interface RallyOutcomePrediction {
  predicted_winner: string;
  player_a_prob: number;
  player_b_prob: number;
  confidence: number;
  is_low_confidence: boolean;
}

export interface PredictionResponse {
  model_status: string;
  next_shot: NextShotPrediction;
  rally_outcome: RallyOutcomePrediction;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  task_a: string;
  task_b: string;
  hardware: string;
  num_classes: number;
  classes: string[];
  model_size_mb?: number;
  inference_latency_ms?: number;
  test_metrics?: any;
}

export const fetchHealth = async (): Promise<HealthStatus> => {
  const res = await axios.get(`${API_BASE_URL}/health`);
  return res.data;
};

export const uploadVideo = async (file: File): Promise<VideoUploadResult> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await axios.post(`${API_BASE_URL}/video/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const startVideoAnalysis = async (videoId: string, sampleRate = 2) => {
  const res = await axios.post(`${API_BASE_URL}/video/analyze`, {
    video_id: videoId,
    sample_rate: sampleRate,
  });
  return res.data;
};

export const getAnalysisStatus = async (videoId: string) => {
  const res = await axios.get(`${API_BASE_URL}/video/${videoId}/status`);
  return res.data;
};

export const getAnalysisResults = async (videoId: string) => {
  const res = await axios.get(`${API_BASE_URL}/video/${videoId}/results`);
  return res.data;
};

export const predictShot = async (featureSequence: number[][]) => {
  const res = await axios.post(`${API_BASE_URL}/predict/shot`, {
    feature_sequence: featureSequence,
  });
  return res.data;
};

export const predictRally = async (featureSequence: number[][]) => {
  const res = await axios.post(`${API_BASE_URL}/predict/rally`, {
    feature_sequence: featureSequence,
  });
  return res.data;
};

export const fetchModelInfo = async (): Promise<ModelInfo> => {
  const res = await axios.get(`${API_BASE_URL}/model/info`);
  return res.data;
};
