import pandas as pd
from torch.utils.data import Dataset
import os
import cv2
import torch
from PIL import Image


import os
from torch.utils.data import Dataset
from PIL import Image


class LandmarkDataset(Dataset):
    def __init__(self, features, target, transform=None, **parameters):
        """
        Args:
            features (list): list of image IDs
            target (dict): mapping {id: class_id}
            transform (callable, optional): torchvision/albumentations transform
            parameters:
                - format: file extension (default "jpg")
                - colorspace: image mode (default "RGB")
                - directory: dataset root (default cwd)
        """
        self.features = features
        self.target = target
        self.format = parameters.get("format", "jpg")
        self.colorspace = parameters.get("colorspace", "RGB")
        self.directory = parameters.get("directory", os.getcwd())
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature_id = self.features[index]
        target_id = self.target[feature_id]

        if len(feature_id) <= 3:
            raise ValueError(f"[Error] - invalid feature ID {feature_id}")

        path = os.path.join(
            self.directory,
            f"{feature_id[0]}/{feature_id[1]}/{feature_id[2]}/{feature_id}.{self.format}",
        )

        try:
            feature = Image.open(path).convert(self.colorspace)
        except Exception as e:
            raise RuntimeError(f"Could not load image {path}: {e}")

        if self.transform:
            feature = self.transform(feature)

        return feature, target_id
