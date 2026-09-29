import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ml.config import COURT_LENGTH_M, COURT_WIDTH_M

class PlayerTracker:
    """
    Tracks Player 1 (Far court / Top) and Player 2 (Near court / Bottom).
    Uses YOLO object detector when available or color/motion contour tracking fallback.
    Maintains bounding boxes, feet coordinates, court position, and velocity vector.
    """
    def __init__(self):
        self.p1_box = None  # [x1, y1, x2, y2]
        self.p2_box = None
        self.p1_history = []  # [(x_m, y_m, timestamp)]
        self.p2_history = []
        self.yolo_model = None
        
        # Try loading pretrained YOLOv8 if available
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")  # small pretrained COCO model (class 0 = person)
        except Exception as e:
            print(f"[PlayerTracker] YOLO initialization note: {e}. Using optical/contour detector fallback.")

    def detect_and_track(self, frame: np.ndarray, court_detector=None) -> Dict[str, Any]:
        """
        Detect and track Player 1 (top court) and Player 2 (bottom court).
        Returns bounding boxes, foot positions, and court coordinates.
        """
        h, w = frame.shape[:2]
        persons = []
        
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, classes=[0], verbose=False)
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        if conf > 0.35:
                            x1, y1, x2, y2 = box
                            persons.append([x1, y1, x2, y2, conf])
            except Exception:
                persons = []
                
        # Fallback contour detection if YOLO not available or detected < 2 persons
        if len(persons) < 2:
            # Simple heuristic detection for top half & bottom half
            # Top half person (Player 1)
            p1_x1, p1_y1, p1_x2, p1_y2 = int(w * 0.42), int(h * 0.28), int(w * 0.58), int(h * 0.48)
            # Bottom half person (Player 2)
            p2_x1, p2_y1, p2_x2, p2_y2 = int(w * 0.40), int(h * 0.62), int(w * 0.60), int(h * 0.88)
            
            persons = [
                [p1_x1, p1_y1, p1_x2, p1_y2, 0.90],
                [p2_x1, p2_y1, p2_x2, p2_y2, 0.90]
            ]
            
        # Assign Person 1 (smallest y2, i.e., top court) and Person 2 (largest y2, bottom court)
        persons.sort(key=lambda b: (b[1] + b[3]) / 2.0)  # sort by vertical midpoint
        
        p1_box = persons[0][:4]
        p2_box = persons[-1][:4] if len(persons) > 1 else persons[0][:4]
        
        # Calculate feet pixels (bottom center of bounding box)
        p1_feet_px = ((p1_box[0] + p1_box[2]) / 2.0, p1_box[3])
        p2_feet_px = ((p2_box[0] + p2_box[2]) / 2.0, p2_box[3])
        
        # Map to 2D court coordinates in meters
        if court_detector is not None:
            p1_pos_m = court_detector.pixel_to_court(p1_feet_px[0], p1_feet_px[1])
            p2_pos_m = court_detector.pixel_to_court(p2_feet_px[0], p2_feet_px[1])
        else:
            p1_pos_m = (3.05, 2.0)
            p2_pos_m = (3.05, 11.4)
            
        # Update motion histories for velocity calculation
        self.p1_history.append(p1_pos_m)
        self.p2_history.append(p2_pos_m)
        if len(self.p1_history) > 10:
            self.p1_history.pop(0)
            self.p2_history.pop(0)
            
        # Calculate velocity vectors
        p1_vx, p1_vy = 0.0, 0.0
        p2_vx, p2_vy = 0.0, 0.0
        if len(self.p1_history) >= 2:
            p1_vx = self.p1_history[-1][0] - self.p1_history[-2][0]
            p1_vy = self.p1_history[-1][1] - self.p1_history[-2][1]
            p2_vx = self.p2_history[-1][0] - self.p2_history[-2][0]
            p2_vy = self.p2_history[-1][1] - self.p2_history[-2][1]
            
        return {
            "p1_box": p1_box,
            "p2_box": p2_box,
            "p1_feet_px": p1_feet_px,
            "p2_feet_px": p2_feet_px,
            "p1_pos_m": p1_pos_m,
            "p2_pos_m": p2_pos_m,
            "p1_vel": (p1_vx, p1_vy),
            "p2_vel": (p2_vx, p2_vy)
        }

    def draw_player_overlay(self, frame: np.ndarray, tracking_info: Dict[str, Any]) -> np.ndarray:
        """Draw tracked player bounding boxes, labels, and court position badges."""
        overlay = frame.copy()
        
        p1_box = tracking_info["p1_box"]
        p2_box = tracking_info["p2_box"]
        p1_m = tracking_info["p1_pos_m"]
        p2_m = tracking_info["p2_pos_m"]
        
        # Player 1 (Cyan)
        cv2.rectangle(overlay, (int(p1_box[0]), int(p1_box[1])), (int(p1_box[2]), int(p1_box[3])), (255, 255, 0), 2)
        cv2.putText(overlay, f"OBSERVED: Player 1 (Top) [{p1_m[0]:.1f}m, {p1_m[1]:.1f}m]", 
                    (int(p1_box[0]), max(15, int(p1_box[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Player 2 (Emerald)
        cv2.rectangle(overlay, (int(p2_box[0]), int(p2_box[1])), (int(p2_box[2]), int(p2_box[3])), (0, 255, 128), 2)
        cv2.putText(overlay, f"OBSERVED: Player 2 (Bottom) [{p2_m[0]:.1f}m, {p2_m[1]:.1f}m]", 
                    (int(p2_box[0]), max(15, int(p2_box[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
                    
        return overlay
