"""All model architectures.

Three models are exposed:
    - CNN2          : the AMR-Benchmark-style baseline (we will reproduce this)
    - CBAM1D        : the attention block (channel + spatial)
    - CNN2_CBAM     : CNN2 with CBAM inserted after each conv block

Input shape convention (PyTorch Conv1d): (batch, 2, 128)
    channel 0 -> I component
    channel 1 -> Q component
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ===========================================================================
# Baseline: CNN2 (mirrors AMR-Benchmark "CNN2")
# ===========================================================================
class CNN2(nn.Module):
    """Two-conv-layer CNN baseline used in the AMR-Benchmark study."""

    def __init__(self, num_classes: int = C.NUM_CLASSES, dropout: float = C.DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=256, out_channels=80, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ===========================================================================
# CBAM (1D version)
# ===========================================================================
class ChannelAttention1D(nn.Module):
    """Channel-attention sub-module: squeeze across time, attend across channels."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        # x: (B, C, L)
        avg = self.mlp(self.avg_pool(x).squeeze(-1))   # (B, C)
        mx = self.mlp(self.max_pool(x).squeeze(-1))    # (B, C)
        att = torch.sigmoid(avg + mx).unsqueeze(-1)    # (B, C, 1)
        return x * att


class SpatialAttention1D(nn.Module):
    """Spatial-attention sub-module: aggregate channels, attend across time."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        # x: (B, C, L)
        avg = x.mean(dim=1, keepdim=True)               # (B, 1, L)
        mx, _ = x.max(dim=1, keepdim=True)              # (B, 1, L)
        feat = torch.cat([avg, mx], dim=1)              # (B, 2, L)
        att = torch.sigmoid(self.conv(feat))            # (B, 1, L)
        return x * att


class CBAM1D(nn.Module):
    """Convolutional Block Attention Module for 1D signals."""

    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ===========================================================================
# Extended: CNN2 + CBAM
# ===========================================================================
class CNN2_CBAM(nn.Module):
    """CNN2 with a CBAM block after each conv layer."""

    def __init__(self, num_classes: int = C.NUM_CLASSES,
                 dropout: float = C.DROPOUT, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 256, kernel_size=3, padding=1)
        self.cbam1 = CBAM1D(256, reduction=reduction)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(256, 80, kernel_size=3, padding=1)
        self.cbam2 = CBAM1D(80, reduction=reduction)
        self.drop2 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.cbam1(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.cbam2(x)
        x = self.drop2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "cnn2":
        return CNN2(**kwargs)
    if name in {"cnn2_cbam", "cbam"}:
        return CNN2_CBAM(**kwargs)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    # quick sanity check
    for name in ("cnn2", "cnn2_cbam"):
        m = build_model(name)
        x = torch.randn(4, 2, 128)
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name:12s}  out={tuple(y.shape)}  params={n_params:,}")
