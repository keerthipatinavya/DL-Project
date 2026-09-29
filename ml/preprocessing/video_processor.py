import cv2
import numpy as np
import os
import time
from typing import Dict, List, Any, Optional, Tuple

from ml.preprocessing.court_detector import CourtDetector
from ml.preprocessing.player_tracker import PlayerTracker
from ml.preprocessing.shuttle_tracker import ShuttleTracker
from ml.preprocessing.feature_extractor import SpatiotemporalFeatureExtractor
from ml.inference.predict import BadmintonPredictor

class BadmintonVideoProcessor:
    """
    End-to-end Computer Vision & Deep Learning Video Processing Pipeline for Badminton Matches.
    """
    def __init__(self, sample_rate: int = 2):
        self.sample_rate = sample_rate
        self.court_detector = CourtDetector()
        self.player_tracker = PlayerTracker()
        self.shuttle_tracker = ShuttleTracker()
        self.feature_extractor = SpatiotemporalFeatureExtractor()
        self.predictor = BadmintonPredictor()

    def inspect_video(self, video_path: str) -> Dict[str, Any]:
        """Validates video format, duration, FPS, and resolution."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        cap.release()
        
        return {
            "video_path": video_path,
            "filename": os.path.basename(video_path),
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration_sec, 2),
            "is_valid": True
        }

    def process_video(self, input_video_path: str, output_video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs complete spatiotemporal video analysis pipeline.
        Overlays OBSERVED tracking features and PREDICTED deep learning outputs.
        """
        meta = self.inspect_video(input_video_path)
        cap = cv2.VideoCapture(input_video_path)
        
        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, meta["fps"], (meta["width"], meta["height"]))
            
        frame_idx = 0
        court_corners = None
        
        timeline_events = []
        last_prediction = None
        
        p1_positions = []
        p2_positions = []
        
        t0 = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % self.sample_rate != 0 and frame_idx > 1:
                if writer and 'annotated_frame' in locals():
                    writer.write(annotated_frame)
                continue
                
            # 1. Court detection & homography
            if court_corners is None or frame_idx % 60 == 1:
                court_corners = self.court_detector.detect_court_corners(frame)
                
            # 2. Player detection & tracking
            tracking_info = self.player_tracker.detect_and_track(frame, self.court_detector)
            p1_positions.append(tracking_info["p1_pos_m"])
            p2_positions.append(tracking_info["p2_pos_m"])
            
            # 3. Shuttlecock tracking
            shuttle_info = self.shuttle_tracker.detect_and_track(frame, frame_idx, self.court_detector)
            
            # 4. Spatiotemporal feature extraction
            feat = self.feature_extractor.extract_frame_feature(tracking_info, shuttle_info)
            seq = self.feature_extractor.update_and_get_sequence(feat)
            
            # 5. Deep Learning Model Prediction
            prediction = self.predictor.predict_sequence(seq)
            last_prediction = prediction
            
            # 6. Video Overlays
            annotated_frame = self.court_detector.draw_court_overlay(frame, court_corners)
            annotated_frame = self.player_tracker.draw_player_overlay(annotated_frame, tracking_info)
            annotated_frame = self.shuttle_tracker.draw_shuttle_overlay(annotated_frame, shuttle_info)
            
            # Draw Prediction Badge
            top_shot = prediction["next_shot"]["top_prediction"].upper()
            conf_shot = prediction["next_shot"]["confidence"] * 100
            winner = prediction["rally_outcome"]["predicted_winner"]
            w_prob = prediction["rally_outcome"]["confidence"] * 100
            
            # Banner on top
            cv2.rectangle(annotated_frame, (10, 10), (520, 105), (15, 23, 42), -1)
            cv2.rectangle(annotated_frame, (10, 10), (520, 105), (59, 130, 246), 2)
            
            cv2.putText(annotated_frame, "DEEP LEARNING MODEL PREDICTIONS:", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"PREDICTED NEXT SHOT: {top_shot} ({conf_shot:.1f}%)", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 211, 238), 2)
            cv2.putText(annotated_frame, f"PREDICTED RALLY WINNER: {winner} ({w_prob:.1f}%)", (20, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52, 211, 153), 2)
                        
            if writer:
                writer.write(annotated_frame)
                
            # Log periodic timeline events
            if frame_idx % (self.sample_rate * 15) == 0:
                timestamp_sec = frame_idx / meta["fps"] if meta["fps"] > 0 else 0.0
                timeline_events.append({
                    "timestamp": round(timestamp_sec, 2),
                    "frame_idx": frame_idx,
                    "hitter": "Player 1" if feat[13] == 0 else "Player 2",
                    "observed_action": prediction["next_shot"]["top_prediction"],
                    "confidence": prediction["next_shot"]["confidence"],
                    "shuttle_pos": shuttle_info["shuttle_pos_m"]
                })
                
        cap.release()
        if writer:
            writer.release()
            
        proc_time = time.time() - t0
        
        return {
            "meta": meta,
            "processing_time_sec": round(proc_time, 2),
            "processed_frames": frame_idx,
            "annotated_video_path": output_video_path,
            "last_prediction": last_prediction,
            "timeline_events": timeline_events,
            "p1_positions": p1_positions[::5],  # sampled for heatmap
            "p2_positions": p2_positions[::5]
        }
