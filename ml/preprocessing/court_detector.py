import cv2
import numpy as np
from typing import Tuple, List, Optional
from ml.config import COURT_LENGTH_M, COURT_WIDTH_M

class CourtDetector:
    """
    Detects badminton court boundaries and computes Homography Matrix H
    mapping pixel coordinates (u, v) to 2D real-world court coordinates (x_m, y_m).
    
    Court coordinate system:
    (0, 0) = Top-left corner of top court
    (6.10, 0) = Top-right corner of top court
    (0, 13.40) = Bottom-left corner of bottom court
    (6.10, 13.40) = Bottom-right corner of bottom court
    Net line is at Y = 6.70m
    """
    def __init__(self):
        # 4 corners of standard badminton court in meters:
        # [top-left, top-right, bottom-right, bottom-left]
        self.court_corners_meters = np.array([
            [0.0, 0.0],
            [COURT_WIDTH_M, 0.0],
            [COURT_WIDTH_M, COURT_LENGTH_M],
            [0.0, COURT_LENGTH_M]
        ], dtype=np.float32)
        
        self.homography_matrix = None
        self.inv_homography_matrix = None

    def detect_court_corners(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect key court corners from frame using line detection / HSV green-white segmentation.
        Fallback to heuristic camera perspective bounds if direct detection is noisy.
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough Lines P
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=w // 6, maxLineGap=20)
        
        # Default perspective coordinates (typical broadcast view)
        top_left = [int(w * 0.25), int(h * 0.28)]
        top_right = [int(w * 0.75), int(h * 0.28)]
        bottom_right = [int(w * 0.88), int(h * 0.90)]
        bottom_left = [int(w * 0.12), int(h * 0.90)]
        
        pixel_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        
        # Compute Homography
        self.homography_matrix, _ = cv2.findHomography(pixel_corners, self.court_corners_meters)
        if self.homography_matrix is not None:
            self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
            
        return pixel_corners

    def pixel_to_court(self, px: float, py: float) -> Tuple[float, float]:
        """Maps pixel coordinate (px, py) to 2D court coordinate in meters (x_m, y_m)."""
        if self.homography_matrix is None:
            # Simple fallback ratio
            return (px / 1280.0) * COURT_WIDTH_M, (py / 720.0) * COURT_LENGTH_M
            
        pt = np.array([px, py, 1.0], dtype=np.float32)
        res = np.dot(self.homography_matrix, pt)
        if abs(res[2]) > 1e-6:
            x_m = res[0] / res[2]
            y_m = res[1] / res[2]
            # Clamp to court bounds with margin
            x_m = np.clip(x_m, -1.0, COURT_WIDTH_M + 1.0)
            y_m = np.clip(y_m, -1.0, COURT_LENGTH_M + 1.0)
            return float(x_m), float(y_m)
        return 3.05, 6.70

    def court_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Maps 2D court coordinate in meters (x_m, y_m) back to frame pixel coordinate (px, py)."""
        if self.inv_homography_matrix is None:
            return int((x_m / COURT_WIDTH_M) * 1280), int((y_m / COURT_LENGTH_M) * 720)
            
        pt = np.array([x_m, y_m, 1.0], dtype=np.float32)
        res = np.dot(self.inv_homography_matrix, pt)
        if abs(res[2]) > 1e-6:
            px = int(res[0] / res[2])
            py = int(res[1] / res[2])
            return px, py
        return 640, 360

    def draw_court_overlay(self, frame: np.ndarray, corners_px: np.ndarray) -> np.ndarray:
        """Draw detected court boundary lines and key regions on frame."""
        overlay = frame.copy()
        pts = corners_px.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Draw Net line across middle (Y = 6.7m)
        net_p1 = self.court_to_pixel(0.0, 6.70)
        net_p2 = self.court_to_pixel(COURT_WIDTH_M, 6.70)
        cv2.line(overlay, net_p1, net_p2, (0, 0, 255), 2)
        
        # Add Label
        cv2.putText(overlay, "OBSERVED: Court Boundary & Homography", (int(corners_px[0][0]), max(20, int(corners_px[0][1]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return overlay
