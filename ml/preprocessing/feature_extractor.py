import numpy as np
from typing import Dict, List, Any
from ml.config import FEATURE_DIM, SEQUENCE_LENGTH

class SpatiotemporalFeatureExtractor:
    """
    Transforms player tracking, court homography, and shuttle trajectory into 
    a 16-dimensional spatiotemporal feature vector per timestep.
    """
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.feature_buffer = []

    def extract_frame_feature(self, tracking_info: Dict[str, Any], shuttle_info: Dict[str, Any]) -> List[float]:
        p1_x, p1_y = tracking_info["p1_pos_m"]
        p1_vx, p1_vy = tracking_info["p1_vel"]
        p2_x, p2_y = tracking_info["p2_pos_m"]
        p2_vx, p2_vy = tracking_info["p2_vel"]
        
        s_x, s_y = shuttle_info["shuttle_pos_m"]
        s_vx, s_vy = shuttle_info["shuttle_vel"]
        
        p_dist = float(np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2))
        
        # Hitter heuristic: player closest to shuttlecock
        d1 = np.sqrt((p1_x - s_x)**2 + (p1_y - s_y)**2)
        d2 = np.sqrt((p2_x - s_x)**2 + (p2_y - s_y)**2)
        hitter_id = 0.0 if d1 <= d2 else 1.0
        
        # Shuttle altitude proxy
        shuttle_alt = 2.5 if s_y < 4.0 or s_y > 9.4 else 1.2
        court_depth = abs(s_y - 6.7)
        
        feat = [
            float(p1_x), float(p1_y), float(p1_vx), float(p1_vy),
            float(p2_x), float(p2_y), float(p2_vx), float(p2_vy),
            float(s_x), float(s_y), float(s_vx), float(s_vy),
            float(p_dist), float(hitter_id), float(shuttle_alt), float(court_depth)
        ]
        return feat

    def update_and_get_sequence(self, feat: List[float]) -> np.ndarray:
        self.feature_buffer.append(feat)
        if len(self.feature_buffer) > self.sequence_length:
            self.feature_buffer.pop(0)
            
        seq = list(self.feature_buffer)
        while len(seq) < self.sequence_length:
            seq.insert(0, seq[0] if seq else [0.0] * FEATURE_DIM)
            
        return np.array(seq, dtype=np.float32)
