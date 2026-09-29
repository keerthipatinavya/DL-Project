import os
import uuid
import json
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any

from backend.app.schemas.schemas import (
    HealthResponse, VideoUploadResponse, VideoAnalysisRequest,
    PredictionRequest, ShotPredictionResponse, RallyOutcomePredictionResponse,
    ModelInfoResponse
)
from ml.config import BASE_DIR, MODELS_DIR, DEVICE, METRICS_FILE_PATH, MODEL_CHECKPOINT_PATH
from ml.preprocessing.video_processor import BadmintonVideoProcessor
from ml.inference.predict import BadmintonPredictor

router = APIRouter()

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(UPLOADS_DIR, "processed")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Global in-memory task status dictionary for async video processing
analysis_jobs: Dict[str, Dict[str, Any]] = {}
predictor = BadmintonPredictor()
video_processor = BadmintonVideoProcessor()

@router.get("/health", response_model=HealthResponse)
def health_check():
    metrics_exist = os.path.exists(METRICS_FILE_PATH)
    return {
        "status": "online",
        "version": "1.0.0",
        "device": DEVICE.upper(),
        "model_loaded": predictor.is_loaded
    }

@router.post("/video/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format. Please upload an MP4, MOV, or AVI file.")
        
    video_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    save_filename = f"{video_id}{ext}"
    save_path = os.path.join(UPLOADS_DIR, save_filename)
    
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    try:
        meta = video_processor.inspect_video(save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted video file: {str(e)}")
        
    return {
        "video_id": video_id,
        "filename": file.filename,
        "duration_sec": meta["duration_sec"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "file_path": f"/uploads/{save_filename}",
        "message": "Video uploaded successfully."
    }

def run_bg_analysis(video_id: str, video_path: str, sample_rate: int):
    try:
        analysis_jobs[video_id] = {"status": "processing", "progress": 10}
        out_filename = f"annotated_{video_id}.mp4"
        out_path = os.path.join(PROCESSED_DIR, out_filename)
        
        res = video_processor.process_video(video_path, output_video_path=out_path)
        res["annotated_video_url"] = f"/uploads/processed/{out_filename}"
        
        analysis_jobs[video_id] = {
            "status": "completed",
            "progress": 100,
            "results": res
        }
    except Exception as e:
        analysis_jobs[video_id] = {
            "status": "failed",
            "error": str(e)
        }

@router.post("/video/analyze")
def analyze_video(req: VideoAnalysisRequest, bg_tasks: BackgroundTasks):
    video_id = req.video_id
    matching_files = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(video_id)]
    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Video with ID '{video_id}' not found.")
        
    video_path = os.path.join(UPLOADS_DIR, matching_files[0])
    
    analysis_jobs[video_id] = {"status": "processing", "progress": 5}
    bg_tasks.add_task(run_bg_analysis, video_id, video_path, req.sample_rate)
    
    return {"job_id": video_id, "status": "processing", "message": "Video analysis task started in background."}

@router.get("/video/{video_id}/status")
def get_analysis_status(video_id: str):
    if video_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return analysis_jobs[video_id]

@router.get("/video/{video_id}/results")
def get_analysis_results(video_id: str):
    if video_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    job = analysis_jobs[video_id]
    if job["status"] != "completed":
        return {"status": job["status"], "message": "Results not ready yet."}
    return job["results"]

@router.post("/predict/shot", response_model=ShotPredictionResponse)
def predict_next_shot(req: PredictionRequest):
    seq_arr = np.array(req.feature_sequence, dtype=np.float32)
    res = predictor.predict_sequence(seq_arr)
    ns = res["next_shot"]
    return {
        "model_status": res["model_status"],
        "top_prediction": ns["top_prediction"],
        "confidence": ns["confidence"],
        "probabilities": ns["probabilities"],
        "is_low_confidence": ns["is_low_confidence"],
        "confidence_message": ns["confidence_message"]
    }

@router.post("/predict/rally", response_model=RallyOutcomePredictionResponse)
def predict_rally_outcome(req: PredictionRequest):
    seq_arr = np.array(req.feature_sequence, dtype=np.float32)
    res = predictor.predict_sequence(seq_arr)
    ro = res["rally_outcome"]
    return {
        "model_status": res["model_status"],
        "predicted_winner": ro["predicted_winner"],
        "player_a_prob": ro["player_a_prob"],
        "player_b_prob": ro["player_b_prob"],
        "confidence": ro["confidence"],
        "is_low_confidence": ro["is_low_confidence"]
    }

@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    test_metrics = None
    if os.path.exists(METRICS_FILE_PATH):
        try:
            with open(METRICS_FILE_PATH, "r") as f:
                test_metrics = json.load(f)
        except Exception:
            pass
            
    mb_size = None
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        mb_size = round(os.path.getsize(MODEL_CHECKPOINT_PATH) / (1024 * 1024), 2)
        
    return {
        "model_name": "BadmintonRallyNet",
        "architecture": "Spatiotemporal ResNet/CNN Spatial Projection + 2-Layer BiLSTM + Transformer Self-Attention + Multi-Task Prediction Heads",
        "task_a": "Task A — Next Shot Probability Distribution Forecasting (7 Classes)",
        "task_b": "Task B — Rally Winner Outcome Prediction (Player 1 vs Player 2)",
        "hardware": DEVICE.upper(),
        "num_classes": 7,
        "classes": ["serve", "clear", "drop", "smash", "drive", "lift", "net_shot"],
        "model_size_mb": mb_size,
        "inference_latency_ms": test_metrics.get("inference_latency_ms") if test_metrics else None,
        "test_metrics": test_metrics
    }
