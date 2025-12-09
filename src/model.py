import torch
import torch.nn as nn
import math
from typing import List
from src.config import ROOT_TO_INT, QUALITY_TO_INT

class CRNN(nn.Module):
    """
    Enhanced CRNN for real-time chord recognition with a multi-head output
    for root and quality.
    """
    def __init__(self, n_mels: int, n_roots: int, n_qualities: int,
                 conv_channels: List[int] = [32, 64, 128],
                 rnn_hidden: int = 128,
                 rnn_layers: int = 2,
                 dropout: float = 0.3,
                 use_attention: bool = False,
                 quality_tower_dim: int = None):
        super().__init__()
        
        convs = []
        f = n_mels
        in_ch = 1
        
        for i, out_ch in enumerate(conv_channels):
            k_freq = 5 if i == 0 else 3
            k_time = 3
            
            convs.append(nn.Conv2d(in_ch, out_ch, 
                                   kernel_size=(k_freq, k_time), 
                                   padding=(k_freq//2, k_time//2)))
            convs.append(nn.BatchNorm2d(out_ch))
            convs.append(nn.ReLU(inplace=True))
            
            if i < 2:
                convs.append(nn.MaxPool2d(kernel_size=(2, 1)))
                f = f // 2
            
            if dropout > 0 and i < len(conv_channels) - 1:
                convs.append(nn.Dropout2d(dropout * 0.5))
            
            in_ch = out_ch
        
        self.conv = nn.Sequential(*convs)
        conv_out_dim = conv_channels[-1] * f
        
        self.rnn = nn.GRU(input_size=conv_out_dim,
                          hidden_size=rnn_hidden,
                          num_layers=rnn_layers,
                          bidirectional=True,
                          batch_first=False,
                          dropout=0.0 if rnn_layers == 1 else dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_hidden * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=False
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # --- Multi-head Classifier ---
        self.root_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_roots)
        )
        
        self.quality_tower = None
        if quality_tower_dim:
            self.quality_tower = nn.Sequential(
                nn.Linear(rnn_hidden * 2, quality_tower_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            quality_head_input_dim = quality_tower_dim
        else:
            quality_head_input_dim = rnn_hidden * 2

        self.quality_head = nn.Sequential(
            nn.Linear(quality_head_input_dim, rnn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden // 2, n_qualities)
        )
    
    def forward(self, x):  # x: (B,T,F)
        B, T, F = x.shape
        
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        
        x = self.conv(x)
        
        T2, B2, C, Fp = x.permute(3, 0, 1, 2).shape
        x = x.permute(3, 0, 1, 2).reshape(T2, B2, C * Fp)
        
        x, _ = self.rnn(x)
        
        if self.use_attention:
            x, _ = self.attention(x, x, x)
        
        x = self.dropout(x)
        
        root_logits = self.root_head(x)
        
        features_for_quality = x
        if self.quality_tower:
            features_for_quality = self.quality_tower(features_for_quality)
        
        quality_logits = self.quality_head(features_for_quality)
        
        root_logits = root_logits.permute(1, 0, 2)
        quality_logits = quality_logits.permute(1, 0, 2)

        return root_logits, quality_logits


class ChordRecognitionWithSmoothing(nn.Module):
    """
    Wrapper that adds temporal smoothing for more stable predictions on multi-head outputs.
    """
    def __init__(self, base_model: nn.Module, smoothing_window: int = 5):
        super().__init__()
        self.model = base_model
        self.smoothing_window = smoothing_window
    
    def _smooth_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if self.smoothing_window <= 1:
            return probs
            
        B, T, C = probs.shape
        kernel = torch.ones(C, 1, 1, self.smoothing_window, 
                           device=probs.device) / self.smoothing_window
        probs_t = probs.permute(0, 2, 1).unsqueeze(2)
        
        pad = self.smoothing_window // 2
        probs_smooth = torch.nn.functional.conv2d(
            probs_t, kernel, padding=(0, pad), groups=C
        )
        return probs_smooth.squeeze(2).permute(0, 2, 1)

    def forward(self, x, apply_smoothing=True):
        root_logits, quality_logits = self.model(x)
        
        if not apply_smoothing or not self.training:
            root_probs = torch.softmax(root_logits, dim=-1)
            quality_probs = torch.softmax(quality_logits, dim=-1)
            
            if apply_smoothing:
                root_probs = self._smooth_probs(root_probs)
                quality_probs = self._smooth_probs(quality_probs)
            
            return root_probs, quality_probs
        
        return root_logits, quality_logits