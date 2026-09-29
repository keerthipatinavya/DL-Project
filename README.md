# Deep Learning Based Badminton Rally & Shot Outcome Prediction Using Sports Video

> **Academic Prototype & Full-Stack AI Sports Analytics Platform**  
> *An end-to-end spatiotemporal deep learning framework for badminton match video analysis, high-speed shuttle tracking, court homography transformation, next-stroke forecasting, and live rally winner prediction.*

---

## 1. Problem Statement

Badminton is one of the world's fastest racquet sports, characterized by shuttlecock velocities exceeding 400 km/h and rapid strategic exchanges. Traditional sports analytics relies on manual post-match statistics or static event logging. However, video-based automated rally understanding requires capturing subtle spatiotemporal dynamics—such as player court positioning, foot movement velocity, shuttle trajectory curvature, and hit sequence patterns.

This project addresses the challenge of automated end-to-end video understanding in badminton by extracting 2D court geometry via computer vision and performing sequence forecasting exclusively using **PyTorch Deep Learning Neural Networks**.

---

## 2. Motivation

Predicting the outcome of a badminton rally or forecasting the next shot choice (e.g., Smash vs. Drop vs. Clear) is fundamentally a dynamic spatiotemporal problem. Unlike static computer vision tasks, stroke selection depends heavily on:
1. **Temporal Context**: The preceding sequence of strokes in the rally (e.g., a high defensive clear often invites an offensive smash).
2. **Spatial Geometry**: Real-world court coordinates of both players and the shuttle relative to the net.
3. **Inter-Player Dynamics**: Distance, movement velocity vectors, and court coverage.

Deep learning architectures—specifically combined Spatial Projections, Bidirectional LSTMs, and Multi-Head Transformer Self-Attention—are uniquely suited to learn these complex temporal dependencies from video streams.

---

## 3. Core Objectives

1. **Video Ingestion & Validation**: Process MP4/MOV match videos with FPS extraction and configurable sampling.
2. **Court Geometry & Homography**: Detect court boundaries and compute a 3x3 Homography Matrix $H$ mapping pixel coordinates $(u, v)$ to real-world court meters $(x_m, y_m)$ ($13.40\text{ m} \times 6.10\text{ m}$).
3. **Player Tracking**: Multi-object tracking (YOLOv8 + ByteTrack) for Player 1 (top court) and Player 2 (bottom court).
4. **Shuttlecock Trajectory Tracking**: Dedicated shuttlecock detection interface utilizing frame differencing and parabolic trajectory smoothing.
5. **Strict Deep Learning Prediction**:
   - **Task A (Next Shot Forecasting)**: Multi-class prediction over supported stroke types (`serve`, `clear`, `drop`, `smash`, `drive`, `lift`, `net_shot`).
   - **Task B (Rally Outcome Forecasting)**: Binary probability estimation for Player A vs. Player B winning the current rally.
6. **Academic Compliance**: **100% Deep Learning models only**. No traditional ML classifiers (XGBoost, SVM, Random Forest, Logistic Regression, etc.) are used for final predictions.
7. **Full-Stack Web Dashboard**: Professional React + TypeScript + Tailwind CSS UI integrated with a FastAPI backend.

---

## 4. Academic Novelty

Positioned as an integrated spatiotemporal research framework, the novelty of this system stems from:
- **Unified Multi-Task Deep Neural Architecture (`BadmintonRallyNet`)**: Jointly optimizes Next-Shot Classification and Rally Winner Prediction through shared temporal representations.
- **Leak-Free Match/Rally Data Partitioning**: Data is split strictly by **Match/Rally ID** rather than random frame splitting to prevent temporal data leakage.
- **Explicit Observed vs. Predicted Separation**: Visual overlays clearly demarcate ground-truth tracking observations from neural predictions.
- **Uncertainty & Low-Confidence Detection**: Identifies low-confidence predictions when visual evidence or sequence history is ambiguous.

---

## 5. System Architecture

```
                                  BADMINTON MATCH VIDEO (.MP4)
                                                │
                                                ▼
                                    [ Video Ingestion Module ]
                                                │
                ┌───────────────────────────────┼───────────────────────────────┐
                ▼                               ▼                               ▼
       [ Court Detector ]             [ Player Tracker ]               [ Shuttle Tracker ]
     Line & Corner Detection          YOLOv8 + ByteTrack             Frame Diff & Parabolic Fit
                │                               │                               │
                └───────────────────────────────┼───────────────────────────────┘
                                                ▼
                                  [ Homography Transformation H ]
                                (Pixel (u,v) ➔ Court Meters (x,y))
                                                │
                                                ▼
                             [ 16-Dim Spatiotemporal Feature Vector ]
                                                │
                                                ▼
                             [ PyTorch BadmintonRallyNet Architecture ]
                                ├── Spatial Dense Projection + GELU
                                ├── 2-Layer Bidirectional LSTM
                                └── Multi-Head Transformer Encoder
                                                │
                                ┌───────────────┴───────────────┐
                                ▼                               ▼
                    [ Task A: Next Shot Head ]      [ Task B: Rally Outcome Head ]
                    (Softmax over 7 Shot Classes)   (Player A vs Player B Win Prob)
```

---

## 6. Deep Learning Architecture (`BadmintonRallyNet`)

### Why Deep Learning is Necessary
Static match statistics fail to capture momentum, spatial court pressure, and temporal shot setups. Traditional ML models treat individual frames as independent samples, ignoring temporal sequence history. `BadmintonRallyNet` uses:
- **Spatial Feature Encoder**: Projecting 16-dimensional coordinate, velocity, and distance features into a 128-dimensional latent space with Layer Normalization and GELU activations.
- **Positional Encoding**: Injecting temporal order into sequence vectors.
- **Bidirectional LSTM**: Learning bidirectional sequence dependencies across rally timelines.
- **Multi-Head Transformer Self-Attention**: Capturing long-range temporal correlations across strokes.
- **Multi-Task Prediction Heads**: Simultaneous classification of stroke type and rally winner.

---

## 7. Dataset Strategy & Leak-Free Splits

- **Dataset Specs**: 400 rallies across 50 simulated matches generated via physics-informed stroke dynamics and player positioning.
- **Leak-Free Splitting**: Samples are grouped strictly by `match_id` and `rally_id` prior to partitioning (Train: 70%, Validation: 15%, Test: 15%). Random frame-based splitting is strictly prohibited.
- **Public Datasets Supported**: Compatible with **ShuttleSet** and **TrackNet** benchmark dataset schemas.

---

## 8. Empirical Evaluation Results (Real Metrics)

Evaluating `BadmintonRallyNet` on the unseen test set (681 test sequence samples) produced the following empirical results:

| Metric | Task A: Next Shot Classification | Task B: Rally Outcome Prediction |
| :--- | :--- | :--- |
| **Accuracy** | **39.06%** | **50.22%** |
| **Macro F1 Score** | **0.2714** | **0.6686** |
| **Precision (Macro)**| **0.2507** | **0.5022** |
| **Recall (Macro)** | **0.3486** | **1.0000** |
| **ROC-AUC** | N/A | **0.4822** |

- **Hardware Used**: Apple Silicon MPS (GPU Accelerated)
- **Model Checkpoint Size**: 5.94 MB
- **Inference Latency**: 3.62 ms / sample (**276.3 FPS**)

*Note: In compliance with project guidelines, all reported metrics are real empirical evaluation outputs from `ml/evaluation/evaluate.py`.*

---

## 9. Project Structure

```
project/
├── frontend/                     # React 19 + TypeScript + Vite + Tailwind CSS UI
│   ├── src/
│   │   ├── components/           # Landing, Video Analysis, Prediction, Analytics, ModelInfo
│   │   ├── services/             # Axios API Client
│   │   ├── App.tsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
├── backend/                      # FastAPI Python Backend
│   └── app/
│       ├── main.py               # Application entry point & CORS
│       ├── api/endpoints.py      # REST API Endpoints
│       └── schemas/schemas.py    # Pydantic Schemas
├── ml/                           # PyTorch Machine Learning Engine
│   ├── config.py                 # Central Hyperparameters & Dimensions
│   ├── models/badminton_dl.py    # PyTorch BadmintonRallyNet Deep Neural Network
│   ├── preprocessing/            # CourtDetector, PlayerTracker, ShuttleTracker, FeatureExtractor
│   ├── datasets/                 # PyTorch Dataset, Generator & Public Dataset Scripts
│   ├── training/train.py         # PyTorch Model Trainer
│   ├── evaluation/evaluate.py    # Empirical Evaluator
│   └── inference/predict.py      # Predictor Wrapper
├── data/                         # Processed JSON Rallies & Metadata
├── models_saved/                 # Trained Checkpoints (.pt) & metrics.json
├── tests/                        # Pytest API & Model Unit Tests
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 10. Installation & Local Execution

### Prerequisites
- Python 3.10+
- Node.js v18+ and npm

### Step 1: Install Python Dependencies & Train Model
```bash
# Install Python packages
pip install -r requirements.txt

# Generate dataset, train PyTorch model & evaluate test set
PYTHONPATH=. python3 ml/datasets/generate_dataset.py
PYTHONPATH=. python3 ml/training/train.py
PYTHONPATH=. python3 ml/evaluation/evaluate.py
```

### Step 2: Run FastAPI Backend
```bash
# Start FastAPI server on http://localhost:8000
uvicorn backend.app.main:app --reload --port 8000
```

### Step 3: Run React Frontend
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

---

## 11. API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System status, device info, PyTorch model loading state |
| `POST` | `/api/video/upload` | Upload MP4/MOV match video and return video metadata |
| `POST` | `/api/video/analyze` | Launch asynchronous spatiotemporal video analysis job |
| `GET` | `/api/video/{id}/status` | Check background video processing job status |
| `GET` | `/api/video/{id}/results` | Fetch complete tracking & deep learning analysis results |
| `POST` | `/api/predict/shot` | Predict Task A next-shot probability distribution |
| `POST` | `/api/predict/rally` | Forecast Task B live rally winner probability |
| `GET` | `/api/model/info` | Fetch model architecture specs and real evaluation metrics |

---

## 12. Research References

1. **TrackNet**: Sun et al., *"TrackNet: A Deep Learning Network for Tracking High-speed Small Objects in Badminton Videos"*, IEEE Access, 2020.
2. **ShuttleSet**: Wang et al., *"ShuttleSet: A High-Resolution Stroke-Level Badminton Dataset"*, ACM Multimedia / CVPR Workshops, 2023.
3. **Spatiotemporal Action Recognition**: Carreira & Zisserman, *"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"*, CVPR, 2017.
4. **Transformer Sequence Modeling**: Vaswani et al., *"Attention Is All You Need"*, NIPS, 2017.
5. **ByteTrack**: Zhang et al., *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box"*, ECCV, 2022.

---

## 13. Limitations & Future Work

- **Camera Angle Dependency**: Best results are obtained from standard broadcast camera angles centered behind the baseline. Multi-camera fusion is planned for future iterations.
- **High Shuttle Occlusion**: Shuttlecock trajectory estimation currently relies on parabolic interpolation during extreme motion blur or player overlaps.
- **Model Scaling**: Future extensions will integrate Vision Transformers (ViT / TimeSformer) directly on raw video frames when higher GPU computational budgets are available.
