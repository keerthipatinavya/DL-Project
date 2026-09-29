import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal sequence modeling."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class BadmintonRallyNet(nn.Module):
    """
    Spatiotemporal Deep Learning Model for Badminton Next-Shot & Rally Outcome Prediction.
    
    Architecture:
    1. Feature Projection & Spatial Encoder (Dense + LayerNorm + GELU)
    2. Temporal Encoder (Bidirectional LSTM + Multi-Head Self-Attention Transformer Encoder)
    3. Multi-Task Heads:
       - Head A: Next Shot Classification (Softmax over 7 badminton shot types)
       - Head B: Rally Outcome Prediction (Softmax over Player 1 / Player 2 win probability)
    """
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        num_shot_classes: int = 7,
        num_lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Spatial Feature Projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Positional Encoding for Temporal Sequences
        self.pos_encoder = TemporalPositionalEncoding(d_model=hidden_dim)
        
        # Temporal Encoder Stage 1: Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        # Temporal Encoder Stage 2: Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        
        # Temporal Pooling Attention
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Task A Head: Next Shot Prediction
        self.shot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_shot_classes)
        )
        
        # Task B Head: Rally Outcome Prediction (Player 1 win vs Player 2 win)
        self.rally_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim]
        Returns:
            dict containing:
                'shot_logits': [batch_size, num_shot_classes]
                'rally_logits': [batch_size, 2]
                'shot_probs': [batch_size, num_shot_classes]
                'rally_probs': [batch_size, 2]
                'confidence_shot': [batch_size]
                'confidence_rally': [batch_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Spatial feature projection
        proj = self.spatial_proj(x)  # [batch_size, seq_len, hidden_dim]
        proj = self.pos_encoder(proj)
        
        # 2. BiLSTM temporal encoding
        lstm_out, _ = self.bilstm(proj)  # [batch_size, seq_len, hidden_dim]
        
        # 3. Transformer temporal encoder
        trans_out = self.transformer(lstm_out)  # [batch_size, seq_len, hidden_dim]
        
        # 4. Attention-weighted pooling across sequence
        attn_scores = self.attention_weights(trans_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(trans_out * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        # 5. Prediction Heads
        shot_logits = self.shot_head(context_vector)     # [batch_size, num_shot_classes]
        rally_logits = self.rally_head(context_vector)   # [batch_size, 2]
        
        shot_probs = F.softmax(shot_logits, dim=-1)
        rally_probs = F.softmax(rally_logits, dim=-1)
        
        # Max probability as baseline confidence score
        confidence_shot, _ = torch.max(shot_probs, dim=-1)
        confidence_rally, _ = torch.max(rally_probs, dim=-1)
        
        return {
            "shot_logits": shot_logits,
            "rally_logits": rally_logits,
            "shot_probs": shot_probs,
            "rally_probs": rally_probs,
            "confidence_shot": confidence_shot,
            "confidence_rally": confidence_rally
        }
