import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
import numpy as np
from sklearn.model_selection import train_test_split

import argparse


class TrainDataset(Dataset):
    """訓練用のデータセットクラス
    """

    def __init__(self, X, y, transform):
        self.train_X = X
        self.train_y = y
        self.transform = transform

    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, index):
        X = self.train_X[index]
        y = self.train_y[index]

        X = tv_tensors.Video(X)

        # Normalize and apply any other transformations
        X = X / 255.0
        X = self.transform(X)

        y = torch.LongTensor(y)

        return X, y


class ValidDataset(Dataset):
    def __init__(self, X, y):
        self.valid_X = X
        self.valid_y = y
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.permute(3, 0, 1, 2))
            ]
        )

    def __len__(self):
        return len(self.valid_X)

    def __getitem__(self, index):
        X = self.valid_X[index]
        y = self.valid_y[index]

        X = tv_tensors.Video(X)

        # Normalize and apply any other transformations
        X = X / 255.0
        X = self.transform(X)

        y = torch.LongTensor(y)

        return X, y


class TestDataset(Dataset):
    def __init__(self, X):
        self.test_X = X
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.permute(3, 0, 1, 2))
            ]
        )

    def __len__(self):
        return len(self.test_X)

    def __getitem__(self, index):
        X = self.test_X[index]
        X = tv_tensors.Video(X)

        # Normalize and apply any other transformations
        X = X / 255.0
        X = self.transform(X)

        return X


def train_dataloader(X, y, transform):
    train_dataset = TrainDataset(X, y, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader


def valid_dataloader(X, y):
    valid_dataset = ValidDataset(X, y)

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False)

    return valid_loader


def test_dataloader(X):
    test_dataset = TestDataset(X)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/data",
    )
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--rotation", type=int, default=None)

    args = parser.parse_args()

    train_X = np.load(f"{args.data_dir}/x_train_report.npy")
    train_y = np.load(f"{args.data_dir}/y_train_report.npy")
    test_X = np.load(f"{args.data_dir}/x_test_report.npy")

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, shuffle=True,
                                                          random_state=args.random_state,
                                                          train_size=args.train_size)

    train_loader = train_dataloader(X=train_X, y=train_y, transform=v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x.permute(3, 0, 1, 2))
        ]
    ))
    valid_loader = valid_dataloader(X=valid_X, y=valid_y)
    test_loader = test_dataloader(X=test_X)

    print(f"{len(train_loader.dataset)=}")
    print(f"{len(valid_loader.dataset)=}")
    print(f"{len(test_loader.dataset)=}")
