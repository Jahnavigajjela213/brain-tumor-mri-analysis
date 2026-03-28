from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # shapes should match for 128x128; add minimal guard for odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 spatial dimensions (H, W)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Enhanced U-Net with Attention Gates for Multi-modal Brain Tumor Segmentation.
    Input: 4 channels (T1, T1ce, T2, FLAIR)
    Output: 3 channels (WT, TC, ET)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 16):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.bottleneck = Down(base * 4, base * 8)

        self.up1 = Up(base * 8 + base * 4, base * 4)
        self.att1 = AttentionGate(F_g=base * 8, F_l=base * 4, F_int=base * 2)
        
        self.up2 = Up(base * 4 + base * 2, base * 2)
        self.att2 = AttentionGate(F_g=base * 4, F_l=base * 2, F_int=base)
        
        self.up3 = Up(base * 2 + base, base)
        self.att3 = AttentionGate(F_g=base * 2, F_l=base, F_int=base // 2)
        
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bottleneck(x3)

        # Decode with Attention
        a1 = self.att1(g=x4, x=x3)
        x = self.up1(x4, a1)
        
        a2 = self.att2(g=x, x=x2)
        x = self.up2(x, a2)
        
        a3 = self.att3(g=x, x=x1)
        x = self.up3(x, a3)
        
        logits = self.outc(x)
        return logits


# Original alias for backward compatibility or replacement
UNetLight = AttentionUNet

