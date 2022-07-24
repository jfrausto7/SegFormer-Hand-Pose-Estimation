import argparse

import torch

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
