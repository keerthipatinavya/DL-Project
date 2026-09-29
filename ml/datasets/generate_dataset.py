import json
import os
import random
import numpy as np
from ml.config import DATA_DIR, SHOT_CLASSES, SHOT_TO_IDX

def generate_synthetic_badminton_dataset(output_file: str, num_matches: int = 50, rallies_per_match: int = 8):
    """
    Generates a realistic physics-informed badminton rally dataset for training and benchmark evaluation.
    Guarantees leak-free grouping by match_id and rally_id.
    """
    random.seed(42)
    np.random.seed(42)
    
    rallies = []
    
    # Transition probabilities between badminton shots
    # Row: current shot, Col: next shot
    transition_matrix = {
        "serve":    {"clear": 0.40, "drop": 0.20, "drive": 0.20, "lift": 0.10, "smash": 0.10},
        "clear":    {"smash": 0.35, "drop": 0.25, "clear": 0.20, "drive": 0.10, "lift": 0.10},
        "drop":     {"net_shot": 0.45, "lift": 0.35, "drive": 0.10, "drop": 0.10},
        "smash":    {"block_lift": 0.40, "net_shot": 0.30, "drive": 0.20, "clear": 0.10},
        "drive":    {"drive": 0.45, "smash": 0.25, "drop": 0.15, "net_shot": 0.15},
        "lift":     {"smash": 0.50, "clear": 0.30, "drop": 0.15, "drive": 0.05},
        "net_shot": {"lift": 0.55, "net_shot": 0.25, "clear": 0.10, "drop": 0.10}
    }
    
    for m in range(1, num_matches + 1):
        match_id = f"match_{m:03d}"
        for r in range(1, rallies_per_match + 1):
            rally_id = f"rally_{r:02d}"
            winner_id = random.choice([0, 1])  # 0 = Player A, 1 = Player B
            
            rally_length = random.randint(6, 18)
            actions = []
            
            curr_shot = "serve"
            curr_hitter = random.choice([0, 1])
            
            p1_pos = [3.05, 1.5]
            p2_pos = [3.05, 11.9]
            
            for step in range(rally_length):
                shot_idx = SHOT_TO_IDX.get(curr_shot, 0)
                
                # Shuttle target position based on shot type
                if curr_hitter == 0:  # Player 1 (top court) hitting to bottom court
                    if curr_shot == "smash":
                        s_x = float(np.random.normal(3.05, 1.2))
                        s_y = float(np.random.normal(9.5, 0.8))
                        shuttle_alt = 0.5
                    elif curr_shot in ["clear", "lift"]:
                        s_x = float(np.random.normal(3.05, 1.5))
                        s_y = float(np.random.normal(12.5, 0.6))
                        shuttle_alt = 3.5
                    elif curr_shot in ["drop", "net_shot"]:
                        s_x = float(np.random.normal(3.05, 1.0))
                        s_y = float(np.random.normal(7.8, 0.5))
                        shuttle_alt = 1.0
                    else:  # drive / serve
                        s_x = float(np.random.normal(3.05, 1.5))
                        s_y = float(np.random.normal(9.0, 1.0))
                        shuttle_alt = 1.5
                        
                    p1_pos[0] = float(np.clip(p1_pos[0] + np.random.normal(0, 0.4), 0.5, 5.6))
                    p1_pos[1] = float(np.clip(p1_pos[1] + np.random.normal(0, 0.3), 0.5, 5.5))
                    # Player 2 moves towards shuttle
                    p2_pos[0] = float(np.clip(s_x + np.random.normal(0, 0.3), 0.5, 5.6))
                    p2_pos[1] = float(np.clip(s_y + np.random.normal(0, 0.3), 7.0, 12.9))
                else:  # Player 2 (bottom court) hitting to top court
                    if curr_shot == "smash":
                        s_x = float(np.random.normal(3.05, 1.2))
                        s_y = float(np.random.normal(3.5, 0.8))
                        shuttle_alt = 0.5
                    elif curr_shot in ["clear", "lift"]:
                        s_x = float(np.random.normal(3.05, 1.5))
                        s_y = float(np.random.normal(0.9, 0.6))
                        shuttle_alt = 3.5
                    elif curr_shot in ["drop", "net_shot"]:
                        s_x = float(np.random.normal(3.05, 1.0))
                        s_y = float(np.random.normal(5.6, 0.5))
                        shuttle_alt = 1.0
                    else:  # drive / serve
                        s_x = float(np.random.normal(3.05, 1.5))
                        s_y = float(np.random.normal(4.0, 1.0))
                        shuttle_alt = 1.5
                        
                    p2_pos[0] = float(np.clip(p2_pos[0] + np.random.normal(0, 0.4), 0.5, 5.6))
                    p2_pos[1] = float(np.clip(p2_pos[1] + np.random.normal(0, 0.3), 7.0, 12.9))
                    # Player 1 moves towards shuttle
                    p1_pos[0] = float(np.clip(s_x + np.random.normal(0, 0.3), 0.5, 5.6))
                    p1_pos[1] = float(np.clip(s_y + np.random.normal(0, 0.3), 0.5, 5.5))
                    
                action = {
                    "step": step,
                    "hitter_id": curr_hitter,
                    "shot_name": curr_shot,
                    "shot_idx": shot_idx,
                    "p1_pos": [round(p1_pos[0], 2), round(p1_pos[1], 2)],
                    "p1_vel": [round(float(np.random.normal(0, 0.2)), 2), round(float(np.random.normal(0, 0.2)), 2)],
                    "p2_pos": [round(p2_pos[0], 2), round(p2_pos[1], 2)],
                    "p2_vel": [round(float(np.random.normal(0, 0.2)), 2), round(float(np.random.normal(0, 0.2)), 2)],
                    "shuttle_pos": [round(s_x, 2), round(s_y, 2)],
                    "shuttle_vel": [round(float(np.random.normal(0, 1.5)), 2), round(float(np.random.normal(5, 2)), 2)],
                    "shuttle_alt": round(shuttle_alt, 2)
                }
                actions.append(action)
                
                # Transition to next shot
                transitions = transition_matrix.get(curr_shot, {"clear": 0.5, "smash": 0.5})
                next_shots = list(transitions.keys())
                weights = list(transitions.values())
                # Normalize weights
                weights = [w / sum(weights) for w in weights]
                curr_shot = np.random.choice(next_shots, p=weights)
                if curr_shot == "block_lift":
                    curr_shot = "lift"
                curr_hitter = 1 - curr_hitter
                
            rallies.append({
                "match_id": match_id,
                "rally_id": rally_id,
                "winner_id": winner_id,
                "num_shots": rally_length,
                "actions": actions
            })
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(rallies, f, indent=2)
        
    print(f"[Dataset Generator] Successfully generated {len(rallies)} rallies across {num_matches} matches -> {output_file}")
    return rallies

if __name__ == "__main__":
    out_path = os.path.join(DATA_DIR, "processed", "badminton_rallies.json")
    generate_synthetic_badminton_dataset(out_path)
