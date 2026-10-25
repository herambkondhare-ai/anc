import torch
import torch.nn as nn

class AudioDenoiserCNN(nn.Module):
    def __init__(self):
        super(AudioDenoiserCNN, self).__init__()
        
        # Encoder: Extract multi-scale temporal/spectral features
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Decoder: Reconstruct clean audio waveform
        self.dec1 = nn.Sequential(
            nn.Conv1d(64 + 64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(32 + 32, 1, kernel_size=15, stride=1, padding=7),
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def forward(self, x):
        # x shape: [Batch_Size, 1, Segment_Length]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Skip connections for precise feature reconstruction
        d1 = self.dec1(torch.cat([e3, e2], dim=1))
        out = self.dec2(torch.cat([d1, e1], dim=1))
        return out