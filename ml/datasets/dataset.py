import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple

class BadmintonRallyDataset(Dataset):
    """
    PyTorch Dataset for Badminton Rally Action Sequences.
    Loads rally sequences and targets for Next Shot Prediction (Task A)
    and Rally Outcome Prediction (Task B).
    """
    def __init__(self, data_path: str, sequence_length: int = 10):
        super().__init__()
        self.sequence_length = sequence_length
        self.samples = []
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
        with open(data_path, "r") as f:
            rallies = json.load(f)
            
        # Process rallies into fixed-length sliding windows
        for rally in rallies:
            match_id = rally.get("match_id", "m1")
            rally_id = rally.get("rally_id", "r1")
            winner_id = rally.get("winner_id", 0)  # 0 = Player A, 1 = Player B
            actions = rally.get("actions", [])
            
            if len(actions) < 2:
                continue
                
            # Build feature sequence
            seq_features = []
            shot_labels = []
            
            for action in actions:
                # 16-dim spatiotemporal feature vector
                p1_x, p1_y = action.get("p1_pos", [3.05, 1.5])
                p1_vx, p1_vy = action.get("p1_vel", [0.0, 0.0])
                p2_x, p2_y = action.get("p2_pos", [3.05, 11.5])
                p2_vx, p2_vy = action.get("p2_vel", [0.0, 0.0])
                s_x, s_y = action.get("shuttle_pos", [3.05, 6.7])
                s_vx, s_vy = action.get("shuttle_vel", [0.0, 10.0])
                p_dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
                hitter_id = action.get("hitter_id", 0)
                shuttle_alt = action.get("shuttle_alt", 2.5)
                court_depth = abs(s_y - 6.7)
                
                feat = [
                    p1_x, p1_y, p1_vx, p1_vy,
                    p2_x, p2_y, p2_vx, p2_vy,
                    s_x, s_y, s_vx, s_vy,
                    p_dist, float(hitter_id), shuttle_alt, court_depth
                ]
                seq_features.append(feat)
                shot_labels.append(action.get("shot_idx", 0))
                
            # Create sliding window sequence samples
            for i in range(1, len(seq_features)):
                start_idx = max(0, i - self.sequence_length)
                sub_seq = seq_features[start_idx:i]
                
                # Pad to sequence_length if needed
                if len(sub_seq) < self.sequence_length:
                    pad_len = self.sequence_length - len(sub_seq)
                    pad_feat = sub_seq[0] if len(sub_seq) > 0 else [0.0] * 16
                    sub_seq = [pad_feat] * pad_len + sub_seq
                    
                target_shot = shot_labels[i]
                
                self.samples.append({
                    "match_id": match_id,
                    "rally_id": rally_id,
                    "features": np.array(sub_seq, dtype=np.float32),
                    "target_shot": int(target_shot),
                    "target_winner": int(winner_id)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        features = torch.tensor(sample["features"], dtype=torch.float32)
        target_shot = torch.tensor(sample["target_shot"], dtype=torch.long)
        target_winner = torch.tensor(sample["target_winner"], dtype=torch.long)
        return features, target_shot, target_winner


def create_rally_dataloaders(
    data_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sequence_length: int = 10,
    seed: int = 42
):
    """
    Split data BY RALLY/MATCH to prevent temporal data leakage.
    """
    dataset = BadmintonRallyDataset(data_path, sequence_length=sequence_length)
    
    # Group samples by match_id & rally_id
    rally_groups = {}
    for idx, sample in enumerate(dataset.samples):
        group_key = f"{sample['match_id']}_{sample['rally_id']}"
        if group_key not in rally_groups:
            rally_groups[group_key] = []
        rally_groups[group_key].append(idx)
        
    group_keys = list(rally_groups.keys())
    np.random.seed(seed)
    np.random.shuffle(group_keys)
    
    n_groups = len(group_keys)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    
    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    test_keys = set(group_keys[n_train + n_val:])
    
    train_indices = [i for k in train_keys for i in rally_groups[k]]
    val_indices = [i for k in val_keys for i in rally_groups[k]]
    test_indices = [i for k in test_keys for i in rally_groups[k]]
    
    train_sub = torch.utils.data.Subset(dataset, train_indices)
    val_sub = torch.utils.data.Subset(dataset, val_indices)
    test_sub = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_indices), len(val_indices), len(test_indices)
