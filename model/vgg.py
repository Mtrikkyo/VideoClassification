import torch
import torch.nn as nn

import torchsummary


class Conv2Plus1D(nn.Sequential):
    """https://pytorch.org/vision/master/_modules/torchvision/models/video/resnet.html"""

    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> tuple[int, int, int]:
        return stride, stride, stride


class Conv3DSimple(nn.Conv3d):
    """https://pytorch.org/vision/master/_modules/torchvision/models/video/resnet.html"""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> tuple[int, int, int]:
        return stride, stride, stride


class VGG3DBrock(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, temporal_pooling: int = 1) -> None:
        super().__init__(
            Conv3DSimple(in_planes=in_planes, out_planes=out_planes),
            nn.ReLU(),
            Conv3DSimple(in_planes=out_planes, out_planes=out_planes),
            nn.ReLU(),
            nn.MaxPool3d((temporal_pooling, 2, 2)),
        )


class VGG3D13(nn.Sequential):
    def __init__(self):
        super().__init__(
            VGG3DBrock(3, 64),
            VGG3DBrock(64, 128, temporal_pooling=2),
            VGG3DBrock(128, 256),
            VGG3DBrock(256, 512, temporal_pooling=2),
            VGG3DBrock(512, 512),
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 16),
        )


if __name__ == "__main__":
    model = VGG3D13().to("cuda")

    torchsummary.summary(model, (3, 10, 112, 112))
