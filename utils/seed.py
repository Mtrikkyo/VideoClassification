import torch
import random
import numpy as np

import argparse


def seed_worker(args: argparse.Namespace) -> None:
    # random
    random.seed(args.random_state)
    # numpy
    np.random.seed(args.random_state)

    # torch
    torch.manual_seed(args.random_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radom_state", type=int, default=42)

    args = parser.parse_args()
    seed_worker(args)
