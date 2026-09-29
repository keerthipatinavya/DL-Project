import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.config import (
    DATA_DIR, MODELS_DIR, DEVICE, SHOT_CLASSES, NUM_SHOT_CLASSES,
    FEATURE_DIM, SEQUENCE_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    WEIGHT_DECAY, MODEL_CHECKPOINT_PATH
)
from ml.models.badminton_dl import BadmintonRallyNet
from ml.datasets.dataset import create_rally_dataloaders
from ml.datasets.generate_dataset import generate_synthetic_badminton_dataset

def train_model():
    print(f"=== Starting BadmintonRallyNet PyTorch Deep Learning Training ===")
    print(f"Hardware Device: {DEVICE}")
    
    dataset_path = os.path.join(DATA_DIR, "processed", "badminton_rallies.json")
    if not os.path.exists(dataset_path):
        print(f"[Train] Dataset not found at {dataset_path}. Generating dataset...")
        generate_synthetic_badminton_dataset(dataset_path)
        
    # Create leak-free data loaders
    train_loader, val_loader, test_loader, n_train, n_val, n_test = create_rally_dataloaders(
        dataset_path,
        batch_size=BATCH_SIZE,
        train_ratio=0.70,
        val_ratio=0.15,
        sequence_length=SEQUENCE_LENGTH,
        seed=42
    )
    print(f"[Train] Split by Rally: Train={n_train} samples, Val={n_val} samples, Test={n_test} samples.")
    
    model = BadmintonRallyNet(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_shot_classes=NUM_SHOT_CLASSES,
        num_lstm_layers=2,
        num_heads=4,
        dropout=0.3
    ).to(DEVICE)
    
    criterion_shot = nn.CrossEntropyLoss()
    criterion_rally = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_shot_acc": [], "val_rally_acc": []}
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for features, target_shot, target_winner in train_loader:
            features = features.to(DEVICE)
            target_shot = target_shot.to(DEVICE)
            target_winner = target_winner.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss_shot = criterion_shot(outputs["shot_logits"], target_shot)
            loss_rally = criterion_rally(outputs["rally_logits"], target_winner)
            
            total_loss = loss_shot + 0.5 * loss_rally
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item() * features.size(0)
            
        scheduler.step()
        epoch_train_loss = running_loss / n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_shot = 0
        correct_rally = 0
        
        with torch.no_grad():
            for features, target_shot, target_winner in val_loader:
                features = features.to(DEVICE)
                target_shot = target_shot.to(DEVICE)
                target_winner = target_winner.to(DEVICE)
                
                outputs = model(features)
                l_shot = criterion_shot(outputs["shot_logits"], target_shot)
                l_rally = criterion_rally(outputs["rally_logits"], target_winner)
                v_loss = l_shot + 0.5 * l_rally
                
                val_loss += v_loss.item() * features.size(0)
                
                pred_shot = torch.argmax(outputs["shot_probs"], dim=-1)
                pred_rally = torch.argmax(outputs["rally_probs"], dim=-1)
                
                correct_shot += (pred_shot == target_shot).sum().item()
                correct_rally += (pred_rally == target_winner).sum().item()
                
        epoch_val_loss = val_loss / n_val
        val_shot_acc = correct_shot / n_val
        val_rally_acc = correct_rally / n_val
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_shot_acc"].append(val_shot_acc)
        history["val_rally_acc"].append(val_rally_acc)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_shot_acc": val_shot_acc,
                "val_rally_acc": val_rally_acc
            }, MODEL_CHECKPOINT_PATH)
            
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Shot Acc: {val_shot_acc*100:.2f}% | Val Rally Acc: {val_rally_acc*100:.2f}%")
            
    total_time = time.time() - start_time
    print(f"\n[Train] Training complete in {total_time:.2f} seconds. Best Val Loss: {best_val_loss:.4f}")
    
    # Save training history
    history_path = os.path.join(MODELS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        
    return model

if __name__ == "__main__":
    train_model()
