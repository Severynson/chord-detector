import torch
import torch.nn as nn
import math
from typing import List

class CRNN(nn.Module):
    """
    Input:  (B, T, F)
    -> reshape to (B, 1, F, T)
    Conv blocks pool over F (freq) only, keep T aligned
    -> (B, C, F', T)
    Collapse freq: permute to (T, B, C*F')
    BiGRU -> (T, B, H*2)
    Linear per frame -> (T, B, C)
    Return (B, T, C)
    """
    def __init__(self, n_mels: int, n_classes: int,
                 conv_channels: List[int] = [32, 64, 128],
                 rnn_hidden: int = 128,
                 rnn_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        chs = [1] + conv_channels
        convs = []
        f = n_mels
        for i in range(len(conv_channels)):
            convs += [
                nn.Conv2d(chs[i], chs[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(chs[i+1]),
                nn.ReLU(inplace=True),
                # pool along freq only; keep time resolution
                nn.MaxPool2d(kernel_size=(2,1))
            ]
            f = math.ceil(f / 2)
        self.conv = nn.Sequential(*convs)
        conv_out_dim = conv_channels[-1] * f  # channels * reduced freq

        self.rnn = nn.GRU(input_size=conv_out_dim,
                          hidden_size=rnn_hidden,
                          num_layers=rnn_layers,
                          bidirectional=True,
                          batch_first=False,
                          dropout=0.0 if rnn_layers == 1 else dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, x):  # x: (B,T,F)
        B, T, F = x.shape
        x = x.unsqueeze(1)          # (B,1,T,F)? we want (B,1,F,T)
        x = x.permute(0,1,3,2)      # (B,1,F,T)
        x = self.conv(x)            # (B,C,F',T)
        x = x.permute(3,0,1,2)      # (T,B,C,F')
        T2, B2, Cc, Fp = x.shape
        x = x.reshape(T2, B2, Cc*Fp) # (T,B,conv_out_dim)
        x, _ = self.rnn(x)           # (T,B,2H)
        x = self.dropout(x)
        x = self.classifier(x)       # (T,B,C)
        x = x.permute(1,0,2)         # (B,T,C)
        return x
