import argparse

import torch
from torch.utils.data import DataLoader

from utils.dataset import FreiHAND
from utils.utils import show_data

# TODO: adjust following config vals
config = {
    "data_dir": "data/FreiHAND_pub_v2",
    "epochs": 1000,
    "batch_size": 64,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20,
    "learning_rate": 0.01,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our BezierModel model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train", help="Train", action="store_true", dest="train")

    parser.add_argument("--test", help="Test", action="store_true", dest="test")

    parser.add_argument(
        "--weights",
        default="",
        help="Path for the weights to use in training/testing",
        dest="weights",
    )

    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of training epochs",
        dest="epochs",
    )

    parser.add_argument(
        "--visualize",
        help="Save a visualization of model architecture",
        action="store_true",
        dest="viz",
    )

    return parser.parse_args()


def get_split_data():
    # Training dataset split
    train_dataset = FreiHAND(config=config, set_type="train")
    train_dataloader = DataLoader(
        train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
    )

    # Validation dataset split
    val_dataset = FreiHAND(config=config, set_type="val")
    print(val_dataset.__len__())
    val_dataloader = DataLoader(
        val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
    )

    # Visualize a random batch of data train samples & labels
    show_data(train_dataset)


def main(args: argparse.Namespace) -> None:
    print("hello")
    if args.train:
        print("getting data")
        get_split_data()


if __name__ == "__main__":
    args = parse_args()
    main(args)
