import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "device" in data

def test_model_info_endpoint():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "BadmintonRallyNet"
    assert data["num_classes"] == 7
    assert "test_metrics" in data

def test_invalid_video_upload():
    # Attempt uploading non-video text file
    response = client.post(
        "/api/video/upload",
        files={"file": ("test.txt", b"invalid content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]

def test_predict_shot_endpoint():
    dummy_seq = [[3.05, 1.5, 0.0, 0.0, 3.05, 11.5, 0.0, 0.0, 3.05, 6.7, 0.0, 5.0, 10.0, 0.0, 2.5, 0.0]] * 10
    response = client.post(
        "/api/predict/shot",
        json={"feature_sequence": dummy_seq}
    )
    assert response.status_code == 200
    data = response.json()
    assert "top_prediction" in data
    assert "confidence" in data
    assert "probabilities" in data

def test_predict_rally_endpoint():
    dummy_seq = [[3.05, 1.5, 0.0, 0.0, 3.05, 11.5, 0.0, 0.0, 3.05, 6.7, 0.0, 5.0, 10.0, 0.0, 2.5, 0.0]] * 10
    response = client.post(
        "/api/predict/rally",
        json={"feature_sequence": dummy_seq}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_winner" in data
    assert "player_a_prob" in data
    assert "player_b_prob" in data
