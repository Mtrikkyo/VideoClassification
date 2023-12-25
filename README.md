# videoclassification

## 参考にしたモデルアーキテクチャ

1. ### VideoResNet(参考文献：[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248))

   画像タスクに対して堅実な ResNet を画像分類タスクに転用したモデル．参考文献では時間方向への畳み込みと空間方向の畳み込みを分離する構造，Conv(2+1)D を用いることで，各フレーム画像に対する画像分類に落とし込むことを可能としている．

   ConV(2+1)D の実装は以下の通りである。

   ```python
   class Conv2Plus1D(nn.Sequential):
    """ https://pytorch.org/vision/stable/_modules/torchvision/models/video/resnet.html より引用・加筆修正"""
       def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
           super().__init__(
               nn.Conv3d(
                   in_planes,
                   midplanes,
                   kernel_size=(1, 3, 3),　#　空間方向の畳み込み
                   stride=(1, stride, stride),
                   padding=(0, padding, padding),
                   bias=False
               ),
               nn.BatchNorm3d(midplanes),
               nn.ReLU(inplace=True), # ReLUを追加することで，Conv3Dよりも複雑な表現が得られる．
               nn.Conv3d(
                   midplanes,
                   out_planes,
                   kernel_size=(3, 1, 1),　#　時間方向の畳み込み
                   stride=(stride, 1, 1),
                   padding=(padding, 0, 0),
                   bias=False
               ),
           )
   ```

2. ### Xception(参考文献：[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357))

   Xception は，入力のチャネル同士に相関関係がないことを利用して，チャネル毎に異なるフィルターを作用させる畳み込み，SeparableConvolution を用いることで省パラメータを実現したモデルである．

   SeparableConvolution の実装は以下の通りである．

   ```python
   class SeparableConv2d(nn.Sequential):
   " https://github.com/pprp/timm/blob/master/timm/models/xception.py より引用・加筆修正"
   def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1) -> None:
      super().__init__(
        nn.Conv2d(
              in_channels,
              in_channels,
              kernel_size,
              stride,
              padding,
              dilation,
              groups=in_channels, # 空間方向の畳み込み
              bias=False
          ),
        nn.Conv2d(
              in_channels,
              out_channels,
              1, 1, 0, 1, 1,　# チャネル方向の畳み込み
              bias=False
          )
      )
   ```

## 実装したモデル

今回実装したモデルでは上記 2 種の Separable Convolution を組み合わせて行うことで、省メモリ且つ複雑な表現を可能とする畳み込み層を実装し、これの畳み込みを利用した残差接続ありの CNN モデルを作成した。

```python
class CTSeparableCon3D(nn.Sequential):

def __init__(self,) -> None:
    super().__init__(
        self.spacewise = nn.Conv3d(
            in_channels,
            in_channels,

        )
    )

```
