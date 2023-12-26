import torch
import torch.nn as nn


class CTSeparableConv3D(nn.Sequential):
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

        self.separable_conv = CTSeparableConv3D(in_channels, out_channels, kernel_size, stride, padding)
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
    def __init__(self, xception_type):
        super().__init__()
        self.xception_type = xception_type

        self.first_norm = nn.BatchNorm3d(3)

        self.input = nn.Conv3d(3, 3, 1, 1, 0, 1)
        self.resblock1 = ResidualBlock(3, 64)
        self.resblock2 = ResidualBlock(64, 128)
        self.resblock3 = ResidualBlock(128, 256)
        self.resblock4 = ResidualBlock(256, 512)
        self.resblock5 = ResidualBlock(512, 512, kernel_size=7, padding=3) # type-B用
        self.resblock6 = ResidualBlock(512, 1024, kernel_size=7, padding=3) # type-B用
        self.resblock7 = ResidualBlock(1024, 1024, kernel_size=7, padding=3) # type-B用

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 時間方向は行わない。
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_typeA = nn.Linear(512, 16)
        self.fc_typeB = nn.Linear(1024, 16)

    def forward_typeA(self, x):
        x = self.first_norm(x)
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
        x = self.fc_typeA(x)

        return x

    def forward_typeB(self, x):
        x = self.first_norm(x)
        x = self.input(x)
        x = self.resblock1(x)
        x = self.maxpool(x)
        x = self.resblock2(x)
        x = self.maxpool(x)
        x = self.resblock3(x)
        x = self.maxpool(x)
        x = self.resblock4(x)
        x = self.maxpool(x)
        x = self.resblock5(x)
        x = self.maxpool(x)
        x = self.resblock6(x)
        x = self.maxpool(x)
        x = self.resblock7(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc_typeB(x)

        return x

    def forward(self, x):
        if self.xception_type == "A":
            x = self.forward_typeA(x)
        elif self.xception_type == "B":
            x = self.forward_typeB(x)

        return x


if __name__ == "__main__":
    # summary
    from torchinfo import summary

    model = VideoXception("A").to("cuda")

    summary(model,(64,3,10,128,128))


