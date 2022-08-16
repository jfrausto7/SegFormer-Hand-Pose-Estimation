import json
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils.utils import (
    DATASET_MEANS,
    DATASET_STDS,
    MODEL_IMG_SIZE,
    RAW_IMG_SIZE,
    projectPoints,
    vector_to_heatmaps,
)


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

        # Open data files
        fn_k_matrix = os.path.join(config["data_dir"], "training_K.json")
        with open(fn_k_matrix, "r") as file:
            self.k_matrix = np.array(json.load(file))

        fn_annotation_3d = os.path.join(config["data_dir"], "training_xyz.json")
        with open(fn_annotation_3d, "r") as file:
            self.annotation_3d = np.array(json.load(file))

        # Set dataset split
        if set_type == "train":
            n_start = 0
            n_end = 26000
        elif set_type == "val":
            n_start = 26000
            n_end = 31000
        else:
            n_start = 31000
            n_end = len(self.annotation_3d)

        # Utilize variables to split dataset
        self.data_names = self.data_names[n_start:n_end]
        self.k_matrix = self.k_matrix[n_start:n_end]
        self.annotation_3d = self.annotation_3d[n_start:n_end]

        print(self.data_names)

        # Apply transformations to raw data
        self.image_raw_transformed = transforms.ToTensor()
        self.image_transformed = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            ]
        )
    
    def __len__(self):
        return len(self.annotation_3d)

    def __getitem__(self, index):
        # Pull data from index
        image_name = self.data_names[index]
        raw = Image.open(os.path.join(self.data_dir, image_name))
        image_raw = self.image_raw_transformed(raw)
        image = self.image_transformed(raw)

        # Initialize keypoints & heatmaps
        keypoints = projectPoints(self.anno[index], self.K_matrix[index])
        keypoints = keypoints / RAW_IMG_SIZE
        heatmaps = vector_to_heatmaps(keypoints)

        # Convert both to tensors
        keypoints = torch.from_numpy(keypoints)
        heatmaps = torch.from_numpy(np.float32(heatmaps))

        return {
            "image": image,
            "image_name": image_name,
            "image_raw": image_raw,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
        }
