import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary  # type: ignore
from models.model import IoULoss, ViT

from utils.dataset import FreiHAND
from utils.utils import N_KEYPOINTS, epoch_eval, epoch_train, show_data

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
        "--inference", help="Inference", action="store_true", dest="inference"
    )

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
    val_dataloader = DataLoader(
        val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
    )

    # Visualize a random batch of data train samples & labels
    show_data(train_dataset)

    return train_dataloader, val_dataloader


def train(
    train_dataloader,
    val_dataloader,
    epochs,
    loss,
    optimizer,
    criterion,
    scheduler,
    checkpoint_frequency,
    model,
    early_stopping_avg,
    early_stopping_precision,
    early_stopping_epochs,
):
    for epoch in range(epochs):
        epoch_train(
            train_dataloader,
            config["device"],
            model,
            optimizer,
            criterion,
            config["batches_per_epoch"],
        )
        epoch_eval(
            val_dataloader,
            config["device"],
            model,
            criterion,
            config["batches_per_epoch_val"],
        )
        print(
            "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                epoch + 1,
                epochs,
                np.round(loss["train"][-1], 10),
                np.round(loss["val"][-1], 10),
            )
        )

        # reducing LR if no improvement
        if scheduler is not None:
            scheduler.step(loss["train"][-1])

        # save model
        if (epoch + 1) % checkpoint_frequency == 0:
            torch.save(model.state_dict(), "model_{}".format(str(epoch + 1).zfill(3)))

        # stop early
        if epoch < early_stopping_avg:
            min_val_loss = np.round(np.mean(loss["val"]), early_stopping_precision)
            no_decrease_epochs = 0

        else:
            val_loss = np.round(
                np.mean(loss["val"][-early_stopping_avg:]), early_stopping_precision,
            )
            if val_loss >= min_val_loss:
                no_decrease_epochs += 1
            else:
                min_val_loss = val_loss
                no_decrease_epochs = 0

        if no_decrease_epochs > early_stopping_epochs:
            print("Stopping early")
            break

    torch.save(model.state_dict(), "model_final")
    return model


def main(args: argparse.Namespace) -> None:

    if args.train:
        # Retrieve data
        print("Loading data...")
        train_dataloader, val_dataloader = get_split_data()

        # Instantiate model and etc.
        ViT_model = ViT(out_channels=N_KEYPOINTS)
        criterion = IoULoss()
        optimizer = optim.SGD(ViT_model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=20,
            verbose=True,
            threshold=0.00001,
        )

        # Print model summary
        summary(ViT_model, (3, 224, 224))

        # TODO: train the model

        # TODO: plot loss of training and validation sets

    if args.test:
        # TODO: evaluate model performance on test data
        pass

    if args.inference:
        # TODO: perform inference on test data
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
