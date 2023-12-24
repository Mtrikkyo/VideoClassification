from ipaddress import summarize_address_range
import torch
import torch.nn as nn


class ConvolutionBlock(nn.Sequential):
    """Channel-TemporalSeparableConvolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        middle_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                dilation=(1, dilation, dilation),
                groups=in_channels,
                bias=bias,
            ),  # depth-tenporal wise
            nn.Conv3d(
                in_channels,
                middle_channels,
                (1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),  # point wise(depth convolution)
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                middle_channels,
                out_channels,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(1, 0, 0),
                dilation=(dilation, 1, 1),
                bias=bias,
            ),  # (temporal convolution)
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        middle_channels=512,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ) -> None:
        super().__init__()
        self.conv_layer = ConvolutionBlock(
            in_channels,
            out_channels,
            middle_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.residual_layer = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, False)

    def forward(self, x):
        residual = x
        x = self.conv_layer(x)
        residual = self.residual_layer(residual)
        x += residual
        return x


class TemporalXception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv3d(3, 3, 1, 1, 0, 1)
        self.resblock1 = ResidualBlock(3, 64)
        self.resblock2 = ResidualBlock(64, 128)
        self.resblock3 = ResidualBlock(128, 256)
        self.resblock4 = ResidualBlock(256, 512)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 16)

    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = TemporalXception()

    from torchsummary import summary

    summary(model, (3, 10, 112, 112))
