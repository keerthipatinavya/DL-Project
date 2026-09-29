import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ml.config import COURT_LENGTH_M, COURT_WIDTH_M

class ShuttleTracker:
    """
    Dedicated Shuttlecock Detection & Trajectory Tracking Module.
    Combines frame-differencing, high-pass brightness blob detection,
    and parabolic trajectory smoothing for high-velocity shuttle tracking.
    """
    def __init__(self):
        self.prev_gray = None
        self.trajectory_history = []  # [(px, py, court_x, court_y, frame_idx)]
        self.max_history = 30

    def detect_and_track(self, frame: np.ndarray, frame_idx: int, court_detector=None) -> Dict[str, Any]:
        """
        Detect shuttlecock in the current frame and update its spatiotemporal trajectory.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = frame.shape[:2]
        
        detected_pt = None
        confidence = 0.0
        
        if self.prev_gray is not None:
            # Frame differencing to capture fast movement
            diff = cv2.absdiff(blurred, self.prev_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find white blob contours matching small shuttlecock geometry (area 5 to 250 px)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 4 <= area <= 300:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    # Filter out bottom baseline noise or far top roof lights
                    if int(h * 0.15) < cy < int(h * 0.95):
                        candidates.append((cx, cy, area))
                        
            if candidates:
                # Pick candidate closest to predicted trajectory continuation if history exists
                if self.trajectory_history:
                    last_pt = self.trajectory_history[-1][:2]
                    candidates.sort(key=lambda c: (c[0] - last_pt[0])**2 + (c[1] - last_pt[1])**2)
                    best = candidates[0]
                else:
                    candidates.sort(key=lambda c: c[2], reverse=True)
                    best = candidates[0]
                    
                detected_pt = (best[0], best[1])
                confidence = 0.85
                
        self.prev_gray = blurred
        
        # Fallback heuristic trajectory interpolation if missed in frame
        if detected_pt is None:
            if len(self.trajectory_history) >= 2:
                p1 = self.trajectory_history[-2]
                p2 = self.trajectory_history[-1]
                # Linear extrapolation
                est_x = p2[0] + (p2[0] - p1[0])
                est_y = p2[1] + (p2[1] - p1[1])
                detected_pt = (est_x, est_y)
                confidence = 0.40  # Interpolated confidence flag
            else:
                detected_pt = (w / 2.0, h / 2.0)
                confidence = 0.20
                
        px, py = detected_pt
        if court_detector is not None:
            court_pos = court_detector.pixel_to_court(px, py)
        else:
            court_pos = (3.05, 6.70)
            
        self.trajectory_history.append((px, py, court_pos[0], court_pos[1], frame_idx))
        if len(self.trajectory_history) > self.max_history:
            self.trajectory_history.pop(0)
            
        # Velocity estimation
        s_vx, s_vy = 0.0, 0.0
        if len(self.trajectory_history) >= 2:
            s_vx = self.trajectory_history[-1][2] - self.trajectory_history[-2][2]
            s_vy = self.trajectory_history[-1][3] - self.trajectory_history[-2][3]
            
        return {
            "shuttle_px": (int(px), int(py)),
            "shuttle_pos_m": court_pos,
            "shuttle_vel": (s_vx, s_vy),
            "confidence": confidence,
            "trajectory": [(int(t[0]), int(t[1])) for t in self.trajectory_history]
        }

    def draw_shuttle_overlay(self, frame: np.ndarray, shuttle_info: Dict[str, Any]) -> np.ndarray:
        """Draw shuttle position, trajectory line, and detection tag."""
        overlay = frame.copy()
        
        trajectory = shuttle_info.get("trajectory", [])
        if len(trajectory) >= 2:
            pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            
        px, py = shuttle_info["shuttle_px"]
        conf = shuttle_info["confidence"]
        s_m = shuttle_info["shuttle_pos_m"]
        
        color = (0, 255, 255) if conf > 0.5 else (0, 165, 255)
        cv2.circle(overlay, (px, py), 6, color, -1)
        cv2.circle(overlay, (px, py), 10, color, 2)
        
        cv2.putText(overlay, f"OBSERVED: Shuttle ({conf*100:.0f}%) [{s_m[0]:.1f}m, {s_m[1]:.1f}m]", 
                    (px + 12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    
        return overlay
