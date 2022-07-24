
import argparse
import json
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils import DATASET_MEANS, DATASET_STDS, MODEL_IMG_SIZE, RAW_IMG_SIZE, projectPoints, vector_to_heatmaps


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our BezierModel model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train',
        help='Train',
        action='store_true',
        dest='train'
    )

    parser.add_argument(
        '--test',
        help='Test',
        action='store_true',
        dest='test'
    )

    parser.add_argument(
        '--weights',
        default='',
        help='Path for the weights to use in training/testing',
        dest='weights'
    )

    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='Number of training epochs',
        dest='epochs'
    )

    parser.add_argument(
        '--visualize',
        help='Save a visualization of model architecture',
        action='store_true',
        dest='viz'
    )

    return parser.parse_args()


class FreiHAND(Dataset):
    """
    Class for loading the FreiHAND dataset.
    Augmented images not to be used.

    Link to dataset: https://lmb.informatik.uni-freiburg.de/projects/freihand/
    """

    def __init__(self, config, set_type="train"):
        # Define class attributes.
        self.device = config["device"]
        self.data_dir = os.path.join(config["data_dir"], "training/rgb")
        self.data_names = np.sort(os.listdir(self.data_dir))

        # Open data files.
        fn_k_matrix = os.path.join(config["data_dir"], "training_K.json")
        with open(fn_k_matrix, "r") as file:
            self.k_matrix = np.array(json.load(file))

        fn_annotation_3d = os.path.join(
            config["data_dir"], "training_xyz.json")
        with open(fn_annotation_3d, "r") as file:
            self.annotation_3d = np.array(json.load(file))

        # Set dataset split
        # TODO: adjust these value splits based on needs
        if set_type == "train":
            n_start = 0
            n_end = 52000
        elif set_type == "val":
            n_start = 52000
            n_end = 57000
        else:
            n_start = 57000
            n_end = len(self.annotation_3d)

        # Utilize variables to split dataset
        self.data_names = self.data_names[n_start:n_end]
        self.k_matrix = self.k_matrix[n_start:n_end]
        self.annotation_3d = self.annotation_3d[n_start:n_end]

        # Apply transformations to raw data
        self.image_raw_transformed = transforms.ToTensor()
        self.image_transformed = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            ]
        )

    def __getitem__(self, index):
        # Pull data from index
        name = self.data_names[index]
        raw = Image.open(os.path.join(self.data_dir, name))
        image_raw = self.image_raw_transformed(raw)
        image = self.image_transformed(raw)

        # Initialize keypoints & heatmaps
        keypoints = (projectPoints(
            self.annotation_3d[index], self.k_matrix[index])) / RAW_IMG_SIZE
        heatmaps = vector_to_heatmaps(keypoints)

        # Convert both to tensors
        keypoints = torch.from_numpy(keypoints)
        heatmaps = torch.from_numpy(np.float32(heatmaps))

        return {
            "image" : image,
            "image_name" : name,
            "image_raw" : raw,
            "keypoints" : keypoints,
            "heatmaps" : heatmaps,
        }
