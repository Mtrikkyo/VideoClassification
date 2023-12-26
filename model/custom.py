import torch
import torch.nn as nn


class CTYSeparableConv3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(1, kernel_size, kernel_size),  # 空間方向の畳み込み
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(kernel_size, 1, 1),  # 時間方向の畳み込み
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),  # チャネル方向の畳み込み
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                groups=1,
                bias=False,
            ),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.separable_conv = CTYSeparableConv3D(in_channels, out_channels, kernel_size, stride, padding)
        self.residual_conv = nn.Conv3d(in_channels, out_channels, 1, stride, 0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.separable_conv(x)
        residual = self.residual_conv(residual)
        x += residual
        x = self.relu(x)

        return x


class VideoXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv3d(3, 3, 1, 1, 0, 1)
        self.resblock1 = ResidualBlock(3, 64)
        self.resblock2 = ResidualBlock(64, 128)
        self.resblock3 = ResidualBlock(128, 256)
        self.resblock4 = ResidualBlock(256, 512)

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2)) # 時間方向は行わない。
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 16)

    def forward(self, x):
        x = self.input(x)
        x = self.resblock1(x)
        x = self.maxpool(x)
        x = self.resblock2(x)
        x = self.maxpool(x)
        x = self.resblock3(x)
        x = self.maxpool(x)
        x = self.resblock4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = VideoXception().to("cuda")

    summary(model, (3, 10, 112, 112))
