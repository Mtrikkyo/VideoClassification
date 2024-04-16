# videoclassification
本リポジトリは、立教大学大学院人工知能科学研究科の必修科目，深層学習(2023)の課題として出題された動画分類コンペティションに提出したモデルをまとめたものである.

今回作成したモデルは検証データにおける正答率が59.5%となっており，コンペティションでの成績は1位であった．

## コンペ概要
### コンペに用いるデータセット

 人間の行動を101クラスに分けたデータセットである[UCF-101](https://arxiv.org/abs/1212.0402)のうち，
 ```text
     "BandMarching","BenchPress","Bowling","BoxingPunchingBag","CricketShot","Drumming","HorseRiding","IceDancing","PlayingCello","PlayingDaf","PlayingDhol","PlayingGuitar","PlayingSitar","Punch","ShavingBeard","TennisSwing"
 ```
 の16クラスのみのミニデータセットを用いた．また，UCF-101では時間方向のサイズがバラバラであるが，今回のデータセットは画像のサイズを一律で
 $$(C,T,H,W)=(3,10,128,128)$$
に加工されている.

訓練データは1842本，テストデータは624本の動画で構成されている.
### コンペルール
- 身で実装したモデルやコードを使って取り組む
- 他人の実装したモデル・コードを利用することは不可
- 外部データ・事前学習済みモデルを使うことは不可
- テストデータに自分でラベルをつけることも禁止

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



<details><summary>構造</summary><div>

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


## Optimizer & Scheduler

### AdamW
### CosineLRScheduler

## DataAugmentation
なし

## 検証データの精度
検証データは，訓練データの3割を使用している．

## リソース
NVIDIA RTX A5000(24564MiB)
