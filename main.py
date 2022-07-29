import argparse
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary  # type: ignore
from models.model import IoULoss, ViT

from utils.dataset import FreiHAND
from utils.utils import (
    MODEL_IMG_SIZE,
    N_IMG_CHANNELS,
    N_KEYPOINTS,
    RAW_IMG_SIZE,
    epoch_eval,
    epoch_train,
    heatmaps_to_coordinates,
    show_batch_predictions,
    show_data,
)

# TODO: adjust following config vals
config = {
    "data_dir": "data/FreiHAND_pub_v2",
    "model_path": "weights/ViT_model_final.pth",
    "epochs": 1000,
    "batch_size": 64,
    "test_batch_size": 4,
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

    parser.add_argument(
        "--show-data",
        help="Visualize random batch of data train samples & labels",
        action="store_true",
        dest="show_data",
    )

    return parser.parse_args()


def get_split_data():
    # Training dataset split
    train_dataset = FreiHAND(config=config, set_type="train")
    train_dataloader = DataLoader(
        train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=0
    )

    # Validation dataset split
    val_dataset = FreiHAND(config=config, set_type="val")
    val_dataloader = DataLoader(
        val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=0
    )

    return train_dataloader, val_dataloader


def train(
    train_dataloader,
    val_dataloader,
    epochs,
    optimizer,
    criterion,
    scheduler,
    checkpoint_frequency,
    model,
    early_stopping_avg,
    early_stopping_precision,
    early_stopping_epochs,
):
    print("Starting training...")
    loss = {"train": [], "val": []}
    for epoch in range(epochs):
        epoch_train(
            train_dataloader,
            config["device"],
            model,
            optimizer,
            criterion,
            loss,
            config["batches_per_epoch"],
        )
        epoch_eval(
            val_dataloader,
            config["device"],
            model,
            criterion,
            loss,
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
            torch.save(
                model.state_dict(),
                "weights/ViT_model_{}".format(str(epoch + 1).zfill(3)),
            )

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

    torch.save(model.state_dict(), config["model_path"])
    return model, loss


def main(args: argparse.Namespace) -> None:

    if args.show_data:
        # Visualize a random batch of data train samples & labels
        train_dataset = FreiHAND(config=config, set_type="train")
        show_data(train_dataset)

    if args.train:
        # Retrieve data
        print("Loading data...")
        train_dataloader, val_dataloader = get_split_data()

        # Instantiate model and etc.
        ViT_model = ViT(out_channels=N_KEYPOINTS, img_size=MODEL_IMG_SIZE)
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
        summary(ViT_model, (N_IMG_CHANNELS, MODEL_IMG_SIZE, MODEL_IMG_SIZE))

        # Train the model
        _, loss = train(
            train_dataloader,
            val_dataloader,
            config["epochs"],
            optimizer,
            criterion,
            scheduler,
            25,
            ViT_model,
            10,
            5,
            10,
        )

        # Plot loss of training and validation sets
        plt.plot(loss["train"], label="train")
        plt.plot(loss["val"], label="val")
        plt.legend()
        plt.savefig("results/loss.png", bbox_inches="tight")
        plt.show()

    if args.test:
        test_dataset = FreiHAND(config=config, set_type="test")
        test_dataloader = DataLoader(
            test_dataset,
            config["test_batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        print("Loading model...")
        model = ViT(out_channels=N_KEYPOINTS, img_size=MODEL_IMG_SIZE)
        model.load_state_dict(
            torch.load(
                config["model_path"], map_location=torch.device(config["device"])
            )
        )
        model.eval()

        # Determine accuracy
        overall_acc = []

        for data in tqdm(test_dataloader):
            inputs = data["image"]
            pred_heatpoints = model(inputs)
            pred_heatpoints = pred_heatpoints.detach().numpy()
            true_keypoints = data["keypoints"].numpy()
            pred_keypoints = heatmaps_to_coordinates(pred_heatpoints)

            accuracy_keypoint = ((true_keypoints - pred_keypoints) ** 2).sum(
                axis=2
            ) ** (1 / 2)
            accuracy_image = accuracy_keypoint.mean(axis=1)
            overall_acc.extend(list(accuracy_image))

        error = np.mean(overall_acc) * 100
        print("Average error per keypoint: {:.1f}%".format(error))

        for img_size in [MODEL_IMG_SIZE, RAW_IMG_SIZE]:
            error_pixels = error * img_size
            size = f"{img_size}x{img_size}"
            print(
                "Average error per keypoint {:.0f} pixels for image {}".format(
                    error_pixels, size
                )
            )

        # visualize application on test data batch
        for data in test_dataloader:
            show_batch_predictions(data, model)
            break

    if args.inference:
        # TODO: perform inference on test data
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
