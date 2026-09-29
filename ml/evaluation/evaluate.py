import os
import sys
import json
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.config import (
    DATA_DIR, MODELS_DIR, DEVICE, SHOT_CLASSES, NUM_SHOT_CLASSES,
    FEATURE_DIM, SEQUENCE_LENGTH, BATCH_SIZE, MODEL_CHECKPOINT_PATH, METRICS_FILE_PATH
)
from ml.models.badminton_dl import BadmintonRallyNet
from ml.datasets.dataset import create_rally_dataloaders

def evaluate_model():
    print("=== Evaluating BadmintonRallyNet Deep Learning Model on Test Set ===")
    
    dataset_path = os.path.join(DATA_DIR, "processed", "badminton_rallies.json")
    if not os.path.exists(dataset_path):
        print(f"[Evaluate Error] Dataset not found at {dataset_path}.")
        return None
        
    _, _, test_loader, _, _, n_test = create_rally_dataloaders(
        dataset_path,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=42
    )
    
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"[Evaluate Error] Saved checkpoint not found at {MODEL_CHECKPOINT_PATH}. Model not trained — metrics unavailable.")
        metrics = {"status": "Model not trained — metrics unavailable."}
        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    model = BadmintonRallyNet(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_shot_classes=NUM_SHOT_CLASSES,
        num_lstm_layers=2,
        num_heads=4
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_true_shot = []
    all_pred_shot = []
    all_prob_shot = []
    
    all_true_rally = []
    all_pred_rally = []
    all_prob_rally = []
    
    latencies = []
    
    with torch.no_grad():
        for features, target_shot, target_winner in test_loader:
            features = features.to(DEVICE)
            
            t0 = time.perf_counter()
            outputs = model(features)
            t1 = time.perf_counter()
            
            batch_latency_ms = ((t1 - t0) * 1000.0) / features.size(0)
            latencies.append(batch_latency_ms)
            
            pred_shot = torch.argmax(outputs["shot_probs"], dim=-1).cpu().numpy()
            pred_rally = torch.argmax(outputs["rally_probs"], dim=-1).cpu().numpy()
            
            all_true_shot.extend(target_shot.numpy())
            all_pred_shot.extend(pred_shot)
            all_prob_shot.extend(outputs["shot_probs"].cpu().numpy())
            
            all_true_rally.extend(target_winner.numpy())
            all_pred_rally.extend(pred_rally)
            all_prob_rally.extend(outputs["rally_probs"][:, 1].cpu().numpy())  # prob of Player B win
            
    # Calculate real Task A metrics (Next Shot Classification)
    shot_acc = float(accuracy_score(all_true_shot, all_pred_shot))
    shot_prec = float(precision_score(all_true_shot, all_pred_shot, average="macro", zero_division=0))
    shot_rec = float(recall_score(all_true_shot, all_pred_shot, average="macro", zero_division=0))
    shot_f1_macro = float(f1_score(all_true_shot, all_pred_shot, average="macro", zero_division=0))
    shot_f1_per_class = f1_score(all_true_shot, all_pred_shot, average=None, zero_division=0).tolist()
    cm_shot = confusion_matrix(all_true_shot, all_pred_shot).tolist()
    
    # Calculate real Task B metrics (Rally Outcome Prediction)
    rally_acc = float(accuracy_score(all_true_rally, all_pred_rally))
    rally_prec = float(precision_score(all_true_rally, all_pred_rally, zero_division=0))
    rally_rec = float(recall_score(all_true_rally, all_pred_rally, zero_division=0))
    rally_f1 = float(f1_score(all_true_rally, all_pred_rally, zero_division=0))
    try:
        rally_roc_auc = float(roc_auc_score(all_true_rally, all_prob_rally))
    except Exception:
        rally_roc_auc = 0.50
        
    avg_latency_ms = float(np.mean(latencies))
    avg_fps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0
    
    checkpoint_size_mb = float(os.path.getsize(MODEL_CHECKPOINT_PATH) / (1024 * 1024))
    
    metrics = {
        "status": "Trained & Evaluated",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": DEVICE.upper(),
        "num_test_samples": n_test,
        "num_shot_classes": NUM_SHOT_CLASSES,
        "shot_classes": SHOT_CLASSES,
        "model_file_size_mb": round(checkpoint_size_mb, 2),
        "inference_latency_ms": round(avg_latency_ms, 2),
        "inference_fps": round(avg_fps, 1),
        "task_a_next_shot": {
            "accuracy": round(shot_acc, 4),
            "precision_macro": round(shot_prec, 4),
            "recall_macro": round(shot_rec, 4),
            "f1_macro": round(shot_f1_macro, 4),
            "per_class_f1": {cls_name: round(f1_val, 4) for cls_name, f1_val in zip(SHOT_CLASSES, shot_f1_per_class)},
            "confusion_matrix": cm_shot
        },
        "task_b_rally_outcome": {
            "accuracy": round(rally_acc, 4),
            "precision": round(rally_prec, 4),
            "recall": round(rally_rec, 4),
            "f1_score": round(rally_f1, 4),
            "roc_auc": round(rally_roc_auc, 4)
        }
    }
    
    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[Evaluate] Successfully computed real metrics:")
    print(f"  Task A (Next Shot)  -> Accuracy: {shot_acc*100:.2f}%, Macro F1: {shot_f1_macro:.4f}")
    print(f"  Task B (Rally Winner) -> Accuracy: {rally_acc*100:.2f}%, ROC-AUC: {rally_roc_auc:.4f}")
    print(f"  Inference Latency   -> {avg_latency_ms:.2f} ms per sample ({avg_fps:.1f} FPS)")
    print(f"  Metrics saved to    -> {METRICS_FILE_PATH}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model()
