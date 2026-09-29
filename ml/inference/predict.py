import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.config import (
    DEVICE, SHOT_CLASSES, NUM_SHOT_CLASSES, FEATURE_DIM, SEQUENCE_LENGTH,
    MODEL_CHECKPOINT_PATH, METRICS_FILE_PATH, IDX_TO_SHOT
)
from ml.models.badminton_dl import BadmintonRallyNet

class BadmintonPredictor:
    """
    Inference predictor wrapper for PyTorch BadmintonRallyNet Deep Learning model.
    Predicts:
    1. Next Shot Probability Distribution (Task A)
    2. Rally Winner Outcome Probabilities (Task B)
    Includes confidence scoring and uncertainty detection.
    """
    def __init__(self, model_path: str = MODEL_CHECKPOINT_PATH):
        self.device = DEVICE
        self.model = None
        self.is_loaded = False
        
        if os.path.exists(model_path):
            try:
                self.model = BadmintonRallyNet(
                    input_dim=FEATURE_DIM,
                    hidden_dim=128,
                    num_shot_classes=NUM_SHOT_CLASSES,
                    num_lstm_layers=2,
                    num_heads=4
                ).to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.is_loaded = True
                print(f"[Predictor] Loaded trained model checkpoint from {model_path}")
            except Exception as e:
                print(f"[Predictor Error] Failed loading model checkpoint: {e}")
                self.is_loaded = False
        else:
            print(f"[Predictor Warning] Checkpoint file {model_path} not found. Running in interface/fallback mode.")

    def predict_sequence(self, feature_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Runs deep learning inference on sequence of spatiotemporal feature vectors.
        Args:
            feature_sequence: array of shape (sequence_length, 16) or (1, sequence_length, 16)
        """
        if feature_sequence.ndim == 2:
            feature_sequence = np.expand_dims(feature_sequence, axis=0)
            
        seq_len = feature_sequence.shape[1]
        if seq_len < SEQUENCE_LENGTH:
            # Pad with leading zeros
            pad = np.zeros((1, SEQUENCE_LENGTH - seq_len, FEATURE_DIM), dtype=np.float32)
            feature_sequence = np.concatenate([pad, feature_sequence], axis=1)
        elif seq_len > SEQUENCE_LENGTH:
            feature_sequence = feature_sequence[:, -SEQUENCE_LENGTH:, :]
            
        x_tensor = torch.tensor(feature_sequence, dtype=torch.float32).to(self.device)
        
        if not self.is_loaded or self.model is None:
            # Baseline probability fallback if model checkpoint not trained yet
            shot_probs_dict = {cls_name: round(1.0 / NUM_SHOT_CLASSES, 4) for cls_name in SHOT_CLASSES}
            shot_probs_dict["smash"] = 0.35
            shot_probs_dict["clear"] = 0.25
            shot_probs_dict["drop"] = 0.20
            shot_probs_dict["drive"] = 0.10
            shot_probs_dict["lift"] = 0.06
            shot_probs_dict["net_shot"] = 0.04
            
            return {
                "model_status": "Model not trained — demo/untrained checkpoint",
                "next_shot": {
                    "top_prediction": "smash",
                    "confidence": 0.35,
                    "probabilities": shot_probs_dict,
                    "is_low_confidence": True,
                    "confidence_message": "Low confidence — model checkpoint not trained."
                },
                "rally_outcome": {
                    "predicted_winner": "Player A",
                    "player_a_prob": 0.52,
                    "player_b_prob": 0.48,
                    "confidence": 0.52,
                    "is_low_confidence": True
                }
            }

        with torch.no_grad():
            outputs = self.model(x_tensor)
            shot_probs = outputs["shot_probs"][0].cpu().numpy()
            rally_probs = outputs["rally_probs"][0].cpu().numpy()
            
        # Top shot prediction
        top_idx = int(np.argmax(shot_probs))
        top_shot = IDX_TO_SHOT[top_idx]
        conf_shot = float(shot_probs[top_idx])
        
        shot_probs_dict = {IDX_TO_SHOT[i]: round(float(p), 4) for i, p in enumerate(shot_probs)}
        
        # Low confidence thresholding
        is_low_conf_shot = conf_shot < 0.38
        conf_msg = "Sufficient temporal evidence." if not is_low_conf_shot else "Low confidence — insufficient visual/temporal evidence."
        
        # Rally winner prediction
        pa_prob = float(rally_probs[0])
        pb_prob = float(rally_probs[1])
        predicted_winner = "Player A" if pa_prob >= pb_prob else "Player B"
        conf_rally = max(pa_prob, pb_prob)
        is_low_conf_rally = conf_rally < 0.55
        
        return {
            "model_status": "Loaded trained PyTorch Deep Learning Model",
            "next_shot": {
                "top_prediction": top_shot,
                "confidence": round(conf_shot, 4),
                "probabilities": shot_probs_dict,
                "is_low_confidence": is_low_conf_shot,
                "confidence_message": conf_msg
            },
            "rally_outcome": {
                "predicted_winner": predicted_winner,
                "player_a_prob": round(pa_prob, 4),
                "player_b_prob": round(pb_prob, 4),
                "confidence": round(conf_rally, 4),
                "is_low_confidence": is_low_conf_rally
            }
        }
