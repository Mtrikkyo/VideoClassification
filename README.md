# videoclassification
本リポジトリは、立教大学大学院人工知能科学研究科の必修科目，深層学習(2023)の課題として出題された動画分類コンペティションに提出したモデルをまとめたものである.

今回作成したモデルは検証データにおける正答率が　となっており，コンペティションでの成績は　であった．

## コンペに用いるデータセット

 人間の行動を101クラスに分けたデータセットである[UCF-101](https://arxiv.org/abs/1212.0402)のうち，
 ```text
     "BandMarching","BenchPress","Bowling","BoxingPunchingBag","CricketShot","Drumming","HorseRiding","IceDancing","PlayingCello","PlayingDaf","PlayingDhol","PlayingGuitar","PlayingSitar","Punch","ShavingBeard","TennisSwing"
 ```
 の16クラスのみを用いた．UCF-101では、時間方向のサイズがバラバラであるが、今回のデータセットは、画像のサイズを一律で，
 $$(C,T,H,W)=(3,10,128,128)$$
に加工されている.
## 参考にしたモデルアーキテクチャ
モデルを作成するにあたって，以下の二つを参考にした.

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
# 畳み込み層
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
```

残差ブロックの中で，入力は CTSeparableConv3D を 1 層通過したのちに残差と加算される．

このブロックを複数回行う Xception 系のモデルを構築した(VideoXception)．

ブロックの数や，そのカーネルサイズによって複数のタイプを作成した．

<details><summary>type-A</summary><div>

```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VideoXception                            [64, 16]                  --
├─BatchNorm3d: 1-1                       [64, 3, 10, 128, 128]     6
├─Conv3d: 1-2                            [64, 3, 10, 128, 128]     12
├─ResidualBlock: 1-3                     [64, 64, 10, 128, 128]    --
│    └─CTSeparableConv3D: 2-1            [64, 64, 10, 128, 128]    --
│    │    └─Conv3d: 3-1                  [64, 3, 10, 128, 128]     27
│    │    └─BatchNorm3d: 3-2             [64, 3, 10, 128, 128]     6
│    │    └─ReLU: 3-3                    [64, 3, 10, 128, 128]     --
│    │    └─Conv3d: 3-4                  [64, 3, 10, 128, 128]     9
│    │    └─BatchNorm3d: 3-5             [64, 3, 10, 128, 128]     6
│    │    └─ReLU: 3-6                    [64, 3, 10, 128, 128]     --
│    │    └─Conv3d: 3-7                  [64, 64, 10, 128, 128]    192
│    └─Conv3d: 2-2                       [64, 64, 10, 128, 128]    256
│    └─ReLU: 2-3                         [64, 64, 10, 128, 128]    --
├─MaxPool3d: 1-4                         [64, 64, 10, 64, 64]      --
├─ResidualBlock: 1-5                     [64, 128, 10, 64, 64]     --
│    └─CTSeparableConv3D: 2-4            [64, 128, 10, 64, 64]     --
│    │    └─Conv3d: 3-8                  [64, 64, 10, 64, 64]      576
│    │    └─BatchNorm3d: 3-9             [64, 64, 10, 64, 64]      128
│    │    └─ReLU: 3-10                   [64, 64, 10, 64, 64]      --
│    │    └─Conv3d: 3-11                 [64, 64, 10, 64, 64]      192
│    │    └─BatchNorm3d: 3-12            [64, 64, 10, 64, 64]      128
│    │    └─ReLU: 3-13                   [64, 64, 10, 64, 64]      --
│    │    └─Conv3d: 3-14                 [64, 128, 10, 64, 64]     8,192
│    └─Conv3d: 2-5                       [64, 128, 10, 64, 64]     8,320
│    └─ReLU: 2-6                         [64, 128, 10, 64, 64]     --
├─MaxPool3d: 1-6                         [64, 128, 10, 32, 32]     --
├─ResidualBlock: 1-7                     [64, 256, 10, 32, 32]     --
│    └─CTSeparableConv3D: 2-7            [64, 256, 10, 32, 32]     --
│    │    └─Conv3d: 3-15                 [64, 128, 10, 32, 32]     1,152
│    │    └─BatchNorm3d: 3-16            [64, 128, 10, 32, 32]     256
│    │    └─ReLU: 3-17                   [64, 128, 10, 32, 32]     --
│    │    └─Conv3d: 3-18                 [64, 128, 10, 32, 32]     384
│    │    └─BatchNorm3d: 3-19            [64, 128, 10, 32, 32]     256
│    │    └─ReLU: 3-20                   [64, 128, 10, 32, 32]     --
│    │    └─Conv3d: 3-21                 [64, 256, 10, 32, 32]     32,768
│    └─Conv3d: 2-8                       [64, 256, 10, 32, 32]     33,024
│    └─ReLU: 2-9                         [64, 256, 10, 32, 32]     --
├─MaxPool3d: 1-8                         [64, 256, 10, 16, 16]     --
├─ResidualBlock: 1-9                     [64, 512, 10, 16, 16]     --
│    └─CTSeparableConv3D: 2-10           [64, 512, 10, 16, 16]     --
│    │    └─Conv3d: 3-22                 [64, 256, 10, 16, 16]     2,304
│    │    └─BatchNorm3d: 3-23            [64, 256, 10, 16, 16]     512
│    │    └─ReLU: 3-24                   [64, 256, 10, 16, 16]     --
│    │    └─Conv3d: 3-25                 [64, 256, 10, 16, 16]     768
│    │    └─BatchNorm3d: 3-26            [64, 256, 10, 16, 16]     512
│    │    └─ReLU: 3-27                   [64, 256, 10, 16, 16]     --
│    │    └─Conv3d: 3-28                 [64, 512, 10, 16, 16]     131,072
│    └─Conv3d: 2-11                      [64, 512, 10, 16, 16]     131,584
│    └─ReLU: 2-12                        [64, 512, 10, 16, 16]     --
├─AdaptiveAvgPool3d: 1-10                [64, 512, 1, 1, 1]        --
├─Linear: 1-11                           [64, 16]                  8,208
==========================================================================================
Total params: 360,850
Trainable params: 360,850
Non-trainable params: 0
Total mult-adds (G): 138.16
==========================================================================================
Input size (MB): 125.83
Forward/backward pass size (MB): 31037.86
Params size (MB): 1.44
Estimated Total Size (MB): 31165.13
==========================================================================================
```

</div></details>

<details><summary>type-B</summary><div>

```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VideoXception                            [64, 16]                  8,208
├─BatchNorm3d: 1-1                       [64, 3, 10, 128, 128]     6
├─Conv3d: 1-2                            [64, 3, 10, 128, 128]     12
├─ResidualBlock: 1-3                     [64, 64, 10, 128, 128]    --
│    └─CTSeparableConv3D: 2-1            [64, 64, 10, 128, 128]    --
│    │    └─Conv3d: 3-1                  [64, 3, 10, 128, 128]     27
│    │    └─BatchNorm3d: 3-2             [64, 3, 10, 128, 128]     6
│    │    └─ReLU: 3-3                    [64, 3, 10, 128, 128]     --
│    │    └─Conv3d: 3-4                  [64, 3, 10, 128, 128]     9
│    │    └─BatchNorm3d: 3-5             [64, 3, 10, 128, 128]     6
│    │    └─ReLU: 3-6                    [64, 3, 10, 128, 128]     --
│    │    └─Conv3d: 3-7                  [64, 64, 10, 128, 128]    192
│    └─Conv3d: 2-2                       [64, 64, 10, 128, 128]    256
│    └─ReLU: 2-3                         [64, 64, 10, 128, 128]    --
├─MaxPool3d: 1-4                         [64, 64, 10, 64, 64]      --
├─ResidualBlock: 1-5                     [64, 128, 10, 64, 64]     --
│    └─CTSeparableConv3D: 2-4            [64, 128, 10, 64, 64]     --
│    │    └─Conv3d: 3-8                  [64, 64, 10, 64, 64]      576
│    │    └─BatchNorm3d: 3-9             [64, 64, 10, 64, 64]      128
│    │    └─ReLU: 3-10                   [64, 64, 10, 64, 64]      --
│    │    └─Conv3d: 3-11                 [64, 64, 10, 64, 64]      192
│    │    └─BatchNorm3d: 3-12            [64, 64, 10, 64, 64]      128
│    │    └─ReLU: 3-13                   [64, 64, 10, 64, 64]      --
│    │    └─Conv3d: 3-14                 [64, 128, 10, 64, 64]     8,192
│    └─Conv3d: 2-5                       [64, 128, 10, 64, 64]     8,320
│    └─ReLU: 2-6                         [64, 128, 10, 64, 64]     --
├─MaxPool3d: 1-6                         [64, 128, 10, 32, 32]     --
├─ResidualBlock: 1-7                     [64, 256, 10, 32, 32]     --
│    └─CTSeparableConv3D: 2-7            [64, 256, 10, 32, 32]     --
│    │    └─Conv3d: 3-15                 [64, 128, 10, 32, 32]     1,152
│    │    └─BatchNorm3d: 3-16            [64, 128, 10, 32, 32]     256
│    │    └─ReLU: 3-17                   [64, 128, 10, 32, 32]     --
│    │    └─Conv3d: 3-18                 [64, 128, 10, 32, 32]     384
│    │    └─BatchNorm3d: 3-19            [64, 128, 10, 32, 32]     256
│    │    └─ReLU: 3-20                   [64, 128, 10, 32, 32]     --
│    │    └─Conv3d: 3-21                 [64, 256, 10, 32, 32]     32,768
│    └─Conv3d: 2-8                       [64, 256, 10, 32, 32]     33,024
│    └─ReLU: 2-9                         [64, 256, 10, 32, 32]     --
├─MaxPool3d: 1-8                         [64, 256, 10, 16, 16]     --
├─ResidualBlock: 1-9                     [64, 512, 10, 16, 16]     --
│    └─CTSeparableConv3D: 2-10           [64, 512, 10, 16, 16]     --
│    │    └─Conv3d: 3-22                 [64, 256, 10, 16, 16]     2,304
│    │    └─BatchNorm3d: 3-23            [64, 256, 10, 16, 16]     512
│    │    └─ReLU: 3-24                   [64, 256, 10, 16, 16]     --
│    │    └─Conv3d: 3-25                 [64, 256, 10, 16, 16]     768
│    │    └─BatchNorm3d: 3-26            [64, 256, 10, 16, 16]     512
│    │    └─ReLU: 3-27                   [64, 256, 10, 16, 16]     --
│    │    └─Conv3d: 3-28                 [64, 512, 10, 16, 16]     131,072
│    └─Conv3d: 2-11                      [64, 512, 10, 16, 16]     131,584
│    └─ReLU: 2-12                        [64, 512, 10, 16, 16]     --
├─MaxPool3d: 1-10                        [64, 512, 10, 8, 8]       --
├─ResidualBlock: 1-11                    [64, 512, 10, 8, 8]       --
│    └─CTSeparableConv3D: 2-13           [64, 512, 10, 8, 8]       --
│    │    └─Conv3d: 3-29                 [64, 512, 10, 8, 8]       25,088
│    │    └─BatchNorm3d: 3-30            [64, 512, 10, 8, 8]       1,024
│    │    └─ReLU: 3-31                   [64, 512, 10, 8, 8]       --
│    │    └─Conv3d: 3-32                 [64, 512, 10, 8, 8]       3,584
│    │    └─BatchNorm3d: 3-33            [64, 512, 10, 8, 8]       1,024
│    │    └─ReLU: 3-34                   [64, 512, 10, 8, 8]       --
│    │    └─Conv3d: 3-35                 [64, 512, 10, 8, 8]       262,144
│    └─Conv3d: 2-14                      [64, 512, 10, 8, 8]       262,656
│    └─ReLU: 2-15                        [64, 512, 10, 8, 8]       --
├─MaxPool3d: 1-12                        [64, 512, 10, 4, 4]       --
├─ResidualBlock: 1-13                    [64, 1024, 10, 4, 4]      --
│    └─CTSeparableConv3D: 2-16           [64, 1024, 10, 4, 4]      --
│    │    └─Conv3d: 3-36                 [64, 512, 10, 4, 4]       25,088
│    │    └─BatchNorm3d: 3-37            [64, 512, 10, 4, 4]       1,024
│    │    └─ReLU: 3-38                   [64, 512, 10, 4, 4]       --
│    │    └─Conv3d: 3-39                 [64, 512, 10, 4, 4]       3,584
│    │    └─BatchNorm3d: 3-40            [64, 512, 10, 4, 4]       1,024
│    │    └─ReLU: 3-41                   [64, 512, 10, 4, 4]       --
│    │    └─Conv3d: 3-42                 [64, 1024, 10, 4, 4]      524,288
│    └─Conv3d: 2-17                      [64, 1024, 10, 4, 4]      525,312
│    └─ReLU: 2-18                        [64, 1024, 10, 4, 4]      --
├─MaxPool3d: 1-14                        [64, 1024, 10, 2, 2]      --
├─ResidualBlock: 1-15                    [64, 1024, 10, 2, 2]      --
│    └─CTSeparableConv3D: 2-19           [64, 1024, 10, 2, 2]      --
│    │    └─Conv3d: 3-43                 [64, 1024, 10, 2, 2]      50,176
│    │    └─BatchNorm3d: 3-44            [64, 1024, 10, 2, 2]      2,048
│    │    └─ReLU: 3-45                   [64, 1024, 10, 2, 2]      --
│    │    └─Conv3d: 3-46                 [64, 1024, 10, 2, 2]      7,168
│    │    └─BatchNorm3d: 3-47            [64, 1024, 10, 2, 2]      2,048
│    │    └─ReLU: 3-48                   [64, 1024, 10, 2, 2]      --
│    │    └─Conv3d: 3-49                 [64, 1024, 10, 2, 2]      1,048,576
│    └─Conv3d: 2-20                      [64, 1024, 10, 2, 2]      1,049,600
│    └─ReLU: 2-21                        [64, 1024, 10, 2, 2]      --
├─AdaptiveAvgPool3d: 1-16                [64, 1024, 1, 1, 1]       --
├─Linear: 1-17                           [64, 16]                  16,400
==========================================================================================
Total params: 4,172,706
Trainable params: 4,172,706
Non-trainable params: 0
Total mult-adds (G): 177.39
==========================================================================================
Input size (MB): 125.83
Forward/backward pass size (MB): 32505.86
Params size (MB): 16.66
Estimated Total Size (MB): 32648.35
==========================================================================================
```

</div></details>

## Optimizer & Scheduler

### AdamW



## DataAugmentation

## 検証データの精度
検証データは，訓練データの3割を使用している．