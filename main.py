import argparse
import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary  # type: ignore
from models.model import IoULoss, SegFormer

from utils.dataset import FreiHAND
from utils.utils import (
    COLORMAP,
    DATASET_MEANS,
    DATASET_STDS,
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
    "inference_dir": "inference",
    "model_path": "weights/ViT_model_final.pth",
    "epochs": 1000,
    "batch_size": 64,
    "test_batch_size": 4,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20,
    "learning_rate": 0.01,
    "num_workers": 2 if torch.cuda.is_available() else 0,
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
        train_dataset,
        config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"],
    )

    # Validation dataset split
    val_dataset = FreiHAND(config=config, set_type="val")
    val_dataloader = DataLoader(
        val_dataset,
        config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"],
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
        # ViT_model = ViT(out_channels=N_KEYPOINTS, img_size=MODEL_IMG_SIZE)
        segformer = SegFormer(
            in_channels=N_IMG_CHANNELS,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[4, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4],
            decoder_channels=256,
            scale_factors=[8, 4, 2, 1],
            num_classes=N_KEYPOINTS,
        )
        criterion = IoULoss()
        optimizer = optim.SGD(segformer.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=20,
            verbose=True,
            threshold=0.00001,
        )

        # Print model summary
        summary(segformer, (N_IMG_CHANNELS, MODEL_IMG_SIZE, MODEL_IMG_SIZE))

        # Train the model
        _, loss = train(
            train_dataloader,
            val_dataloader,
            config["epochs"],
            optimizer,
            criterion,
            scheduler,
            25,
            segformer,
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
            num_workers=config["num_workers"],
        )
        print("Loading model...")
        model = SegFormer(
            in_channels=N_IMG_CHANNELS,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[4, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4],
            decoder_channels=256,
            scale_factors=[8, 4, 2, 1],
            num_classes=N_KEYPOINTS,
        )
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
            pred_heatmaps = model(inputs)
            pred_heatmaps = pred_heatmaps.detach().numpy()
            true_keypoints = data["keypoints"].numpy()
            pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)

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
        print("Loading model...")
        model = SegFormer(
            in_channels=N_IMG_CHANNELS,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[4, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4],
            decoder_channels=256,
            scale_factors=[8, 4, 2, 1],
            num_classes=N_KEYPOINTS,
        )
        model.load_state_dict(
            torch.load(
                config["model_path"], map_location=torch.device(config["device"])
            )
        )
        model.eval()

        # Apply transformations to raw data
        image_transformed = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            ]
        )

        # Iterate through the names of contents of the folder
        for image_path in os.listdir(config["inference_dir"]):

            # create the full input path and read the file
            input_path = os.path.join(config["inference_dir"], image_path)
            raw = Image.open(input_path)

            image = image_transformed(raw)

            pred_heatmaps = model(image)
            pred_heatmaps = pred_heatmaps.detach().numpy()
            pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)

            pred_keypoints_img = pred_keypoints * RAW_IMG_SIZE
            plt.figure(figsize=[9, 4])
            plt.subplot(1, 3, 4)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 5)
            plt.imshow(image)
            for finger, params in COLORMAP.items():
                plt.plot(
                    pred_keypoints_img[params["ids"], 0],
                    pred_keypoints_img[params["ids"], 1],
                    params["color"],
                )
            plt.title("Pred Keypoints")
            plt.axis("off")
        plt.tight_layout()


if __name__ == "__main__":
    args = parse_args()
    main(args)
