from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    model_loaded: bool

class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    duration_sec: float
    fps: float
    width: int
    height: int
    file_path: str
    message: str

class VideoAnalysisRequest(BaseModel):
    video_id: str
    sample_rate: Optional[int] = 2

class PredictionRequest(BaseModel):
    feature_sequence: List[List[float]] = Field(
        ...,
        description="Sequence of 16-dimensional spatiotemporal feature vectors per timestep"
    )

class ShotPredictionResponse(BaseModel):
    model_status: str
    top_prediction: str
    confidence: float
    probabilities: Dict[str, float]
    is_low_confidence: bool
    confidence_message: str

class RallyOutcomePredictionResponse(BaseModel):
    model_status: str
    predicted_winner: str
    player_a_prob: float
    player_b_prob: float
    confidence: float
    is_low_confidence: bool

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    task_a: str
    task_b: str
    hardware: str
    num_classes: int
    classes: List[str]
    model_size_mb: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    test_metrics: Optional[Dict[str, Any]] = None
