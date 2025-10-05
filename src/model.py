import torch
import torch.nn as nn
import math
from typing import List

class CRNN(nn.Module):
    """
    Enhanced CRNN for real-time chord recognition
    
    Key improvements:
    - Better kernel sizes for harmonic capture
    - Smoother frequency pooling
    - Optional attention mechanism
    - Temporal smoothing via output pooling
    """
    def __init__(self, n_mels: int, n_classes: int,
                 conv_channels: List[int] = [32, 64, 128],
                 rnn_hidden: int = 128,
                 rnn_layers: int = 2,
                 dropout: float = 0.3,
                 use_attention: bool = False):
        super().__init__()
        
        # Build convolutional blocks with better kernel sizes
        convs = []
        f = n_mels
        in_ch = 1
        
        for i, out_ch in enumerate(conv_channels):
            # Larger kernels to capture harmonic relationships
            k_freq = 5 if i == 0 else 3
            k_time = 3
            
            convs.append(nn.Conv2d(in_ch, out_ch, 
                                   kernel_size=(k_freq, k_time), 
                                   padding=(k_freq//2, k_time//2)))
            convs.append(nn.BatchNorm2d(out_ch))
            convs.append(nn.ReLU(inplace=True))
            
            # Less aggressive pooling: only first two layers
            if i < 2:
                convs.append(nn.MaxPool2d(kernel_size=(2, 1)))
                f = f // 2
            
            # Optional: light dropout after each block
            if dropout > 0 and i < len(conv_channels) - 1:
                convs.append(nn.Dropout2d(dropout * 0.5))
            
            in_ch = out_ch
        
        self.conv = nn.Sequential(*convs)
        conv_out_dim = conv_channels[-1] * f
        
        # Bidirectional GRU
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
        
        # Classifier with optional intermediate layer
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_classes)
        )
    
    def forward(self, x):  # x: (B,T,F)
        B, T, F = x.shape
        
        # Reshape for conv: (B,1,F,T)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        
        # Conv blocks
        x = self.conv(x)  # (B,C,F',T)
        
        # Prepare for RNN: (T,B,C*F')
        x = x.permute(3, 0, 1, 2)
        T2, B2, C, Fp = x.shape
        x = x.reshape(T2, B2, C * Fp)
        
        # RNN
        x, _ = self.rnn(x)  # (T,B,2H)
        
        # Optional attention
        if self.use_attention:
            x, _ = self.attention(x, x, x)
        
        x = self.dropout(x)
        x = self.classifier(x)  # (T,B,n_classes)
        
        # Return (B,T,n_classes)
        x = x.permute(1, 0, 2)
        return x


class ChordRecognitionWithSmoothing(nn.Module):
    """
    Wrapper that adds temporal smoothing for more stable predictions
    """
    def __init__(self, base_model: nn.Module, smoothing_window: int = 5):
        super().__init__()
        self.model = base_model
        self.smoothing_window = smoothing_window
    
    def forward(self, x, apply_smoothing=True):
        logits = self.model(x)  # (B,T,C)
        
        if not apply_smoothing or not self.training:
            probs = torch.softmax(logits, dim=-1)
            
            # Apply temporal smoothing via convolution
            if apply_smoothing and self.smoothing_window > 1:
                B, T, C = probs.shape
                # Smooth each class independently
                kernel = torch.ones(C, 1, 1, self.smoothing_window, 
                                  device=probs.device) / self.smoothing_window
                probs_t = probs.permute(0, 2, 1).unsqueeze(2)  # (B,C,1,T)
                
                pad = self.smoothing_window // 2
                probs_smooth = torch.nn.functional.conv2d(
                    probs_t, kernel, padding=(0, pad), groups=C
                )
                probs = probs_smooth.squeeze(2).permute(0, 2, 1)  # (B,T,C)
            
            return probs
        
        return logits  # Return raw logits during training