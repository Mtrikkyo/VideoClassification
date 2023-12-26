import torch
from torch import optim
from torch import nn
from timm.scheduler import CosineLRScheduler
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse
import os

# custom script
from utils.seed import seed_worker
from utils.loader import train_valid_dataloader, test_dataloader
from model.custom import VideoXception


# model list
TRAINABLE_MODEL = {
    "xception": VideoXception(),
}

# class list
CLASS_LIST = [
    "BandMarching",
    "BenchPress",
    "Bowling",
    "BoxingPunchingBag",
    "CricketShot",
    "Drumming",
    "HorseRiding",
    "IceDancing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingGuitar",
    "PlayingSitar",
    "Punch",
    "ShavingBeard",
    "TennisSwing",
]


def train(args: argparse.Namespace, model, train_loader, optimizer, criterion) -> None:
    model.train()
    for movies, labels in train_loader:
        labels = labels.view(-1)
        movies, labels = movies.to(args.device), labels.to(args.device)

        # zero grad
        optimizer.zero_grad()

        outputs = model(movies)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


def eval(args: argparse.Namespace, model, train_loader, valid_loader, criterion):
    model.eval()

    with torch.no_grad():
        # train
        train_acc_sum = 0.0
        train_loss_sum = 0.0
        for movies, labels in train_loader:
            labels = labels.view(-1)
            movies, labels = movies.to(args.device), labels.to(args.device)

            outputs = model(movies)

            loss = criterion(outputs, labels)
            acc = (torch.argmax(outputs, dim=1) == labels).sum().item()

            train_loss_sum += loss.item() / len(train_loader.dataset)
            train_acc_sum += acc / len(train_loader.dataset)

        # valid
        valid_acc_sum = 0.0
        valid_loss_sum = 0.0
        for movies, labels in valid_loader:
            labels = labels.view(-1)
            movies, labels = movies.to(args.device), labels.to(args.device)

            outputs = model(movies)
            loss = criterion(outputs, labels)
            acc = (torch.argmax(outputs, dim=1) == labels).sum().item()

            valid_loss_sum += loss.item() / len(valid_loader.dataset)
            valid_acc_sum += acc / len(valid_loader.dataset)

    return (train_acc_sum, train_loss_sum, valid_acc_sum, valid_loss_sum)


def predict_test(args: argparse.Namespace, model, test_loader):
    model.eval()

    with torch.no_grad():
        all_predict_row = []
        all_predict_class = []

        for movies in test_loader:
            movies = movies.to(args.device)
            outputs = model(movies)

            predict_row = outputs.cpu().numpy()
            all_predict_row.append(predict_row)
            predict_class = [CLASS_LIST[index.item()] for index in torch.argmax(outputs, dim=1)]
            all_predict_class.extend(predict_class)

    np.savetxt(f"{args.save_dir}/predict_row.csv", np.vstack(all_predict_row), delimiter=",")
    np.savetxt(
        f"{args.save_dir}/predict.csv",
        np.array(all_predict_class),
        delimiter=",",
        fmt="%s",
    )


def save_model_weight(args, epoch, model, history):
    if epoch >= 1 and max(history["valid_loss"]) == history["valid_loss"][-1]:
        # save weight
        torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")

    elif epoch == 0:
        torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")

    elif epoch == args.epoch - 1:
        torch.save(model.state_dict(), f"{args.save_dir}/last_model.pth")


def plot_history(args, history: dict) -> None:
    acc_fig, acc_ax = plt.subplots()

    acc_ax.plot(range(args.epoch), history["train_acc"], label="train")
    acc_ax.plot(range(args.epoch), history["valid_acc"], label="valid")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("acc")
    plt.legend()
    plt.savefig(f"{args.save_dir}/acc.png")

    loss_fig, loss_ax = plt.subplots()
    loss_ax.plot(range(args.epoch), history["train_loss"], label="train")
    loss_ax.plot(range(args.epoch), history["valid_loss"], label="valid")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    plt.legend()
    plt.savefig(f"{args.save_dir}/loss.png")


def save_score(args, history: dict) -> None:
    score_df = pd.DataFrame(history)
    score_df.to_csv(f"{args.save_dir}/score.csv", index=None)


def main(args: argparse.Namespace):
    # fix random_state
    seed_worker(args=args)

    # make directory
    os.makedirs(name=args.save_dir)

    # load data
    train_loader, valid_loader = train_valid_dataloader(args)
    test_loader = test_dataloader(args=args)

    # setting model
    model = TRAINABLE_MODEL[args.model].to(args.device)

    # optimizer, scheduler & criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epoch, lr_min=1e-6, warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True
    )
    criterion = nn.CrossEntropyLoss()

    # train & eval
    history = {
        "epoch": [i for i in range(args.epoch)],
        "train_acc": [],
        "train_loss": [],
        "valid_acc": [],
        "valid_loss": [],
    }

    for epoch in tqdm(range(args.epoch)):
        scheduler.step(epoch)
        train(
            args=args,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )

        (
            train_acc_per_epoch,
            train_loss_per_epoch,
            valid_acc_per_epoch,
            valid_loss_per_epoch,
        ) = eval(
            args=args,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        history["train_acc"].append(train_acc_per_epoch)
        history["train_loss"].append(train_loss_per_epoch)
        history["valid_acc"].append(valid_acc_per_epoch)
        history["valid_loss"].append(valid_loss_per_epoch)

        save_model_weight(args=args, epoch=epoch, model=model, history=history)

    # predict test_data
    predict_test(args=args, model=model, test_loader=test_loader)

    # plot history
    plot_history(args, history=history)

    # save score history
    save_score(args=args, history=history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["xception"],
        default="xception",
    )
    parser.add_argument("--device", type=str, choices=["cuda", "mps"], default="cuda")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/tsubasa/Competition/Rikkyo/VideoClassification/data",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/tsubasa/Competition/Rikkyo/VideoClassification/result/xception01",
    )
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    args = parser.parse_args()

    main(args)
