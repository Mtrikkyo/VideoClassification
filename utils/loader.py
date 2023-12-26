import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
import numpy as np
from sklearn.model_selection import train_test_split

import argparse


class TrainValidDataset(Dataset):
    def __init__(self, args):
        self.train_X = np.load(f"{args.data_dir}/x_train_report.npy")
        self.train_y = np.load(f"{args.data_dir}/y_train_report.npy")
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.permute(3, 0, 1, 2)),
            ]
        )

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


class TestDataset(Dataset):
    def __init__(self, args):
        self.test_X = np.load(f"{args.data_dir}/x_test_report.npy")
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.permute(3, 0, 1, 2)),
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


def train_valid_dataloader(args: argparse.Namespace):
    # load

    # dataset
    train_dataset = TrainValidDataset(args)

    train_dataset, valid_dataset = train_test_split(
        train_dataset,
        shuffle=True,
        random_state=args.random_state,
        train_size=args.train_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    return (train_loader, valid_loader)


def test_dataloader(args=argparse.Namespace):
    test_dataset = TestDataset(args)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/tsubasa/Competition/Rikkyo/VideoClassification/data",
    )
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    train_loader, valid_loader = train_valid_dataloader(args)

    test_loader = test_dataloader(args)

    print(test_loader.dataset[0].shape)
