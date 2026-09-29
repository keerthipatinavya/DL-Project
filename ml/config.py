import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Badminton Court Dimensions (in meters)
COURT_LENGTH_M = 13.40  # 13.4 meters total length
COURT_WIDTH_M = 6.10    # 6.1 meters width for doubles / 5.18 for singles
NET_Y_M = 6.70          # Net is at center (Y = 6.7m)

# Supported Shot Classes (7 classes)
SHOT_CLASSES = ["serve", "clear", "drop", "smash", "drive", "lift", "net_shot"]
NUM_SHOT_CLASSES = len(SHOT_CLASSES)
SHOT_TO_IDX = {cls_name: i for i, cls_name in enumerate(SHOT_CLASSES)}
IDX_TO_SHOT = {i: cls_name for i, cls_name in enumerate(SHOT_CLASSES)}

# Feature Vector Dimension
# Spatiotemporal features per timestep:
# 0-1: Player 1 (x, y) court pos in meters
# 2-3: Player 1 velocity (vx, vy)
# 4-5: Player 2 (x, y) court pos in meters
# 6-7: Player 2 velocity (vx, vy)
# 8-9: Shuttle position (x, y) in meters
# 10-11: Shuttle velocity (vx, vy)
# 12: Player distance
# 13: Hitter ID (0 or 1)
# 14: Shuttle altitude / arc proxy
# 15: Court depth indicator
FEATURE_DIM = 16

# Temporal Sequence Length (number of observed actions / frame states in rally sequence)
SEQUENCE_LENGTH = 10

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
WEIGHT_DECAY = 1e-4

# Model Weights File Path
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "badminton_rallynet_best.pt")
METRICS_FILE_PATH = os.path.join(MODELS_DIR, "metrics.json")
