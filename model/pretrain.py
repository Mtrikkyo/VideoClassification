import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights


class ResNetR3D(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.base_model = r3d_18(R3D_18_Weights.KINETICS400_V1)
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classify_layer = nn.Linear(in_features=400, out_features=16)

    def forward(self, x: torch.Tensor):
        x = self.base_model(x)
        x = self.classify_layer(x)
        return x


class ResNetR2Plus1D(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.base_model = r3d_18(R3D_18_Weights.KINETICS400_V1)
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classify_layer = nn.Linear(in_features=400, out_features=16)

    def forward(self, x: torch.Tensor):
        x = self.base_model(x)
        x = self.classify_layer(x)
        return x


if __name__ == "__main__":
    import torchsummary

    model = ResNetR3D().to("cuda")

    torchsummary.summary(model, (3, 10, 112, 112))

    pass
