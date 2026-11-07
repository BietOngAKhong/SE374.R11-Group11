import torch
import torch.nn as nn

class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.drop = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        features: tuple[int, ...] = (16, 32, 64, 128),
        bottleneck: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        f1, f2, f3, f4 = features

        self.stem = nn.Conv3d(in_channels, f1, kernel_size=3, padding=1, bias=False)
        self.enc1 = ResBlock3D(f1, f1, dropout)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ResBlock3D(f1, f2, dropout)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResBlock3D(f2, f3, dropout)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ResBlock3D(f3, f4, dropout)
        self.down4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ResBlock3D(f4, bottleneck, dropout)

        self.up4 = nn.ConvTranspose3d(bottleneck, f4, kernel_size=2, stride=2)
        self.dec4 = ResBlock3D(f4 + f4, f4, dropout)

        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(f3 + f3, f3, dropout)

        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(f2 + f2, f2, dropout)

        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(f1 + f1, f1, dropout)

        self.head = nn.Conv3d(f1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

