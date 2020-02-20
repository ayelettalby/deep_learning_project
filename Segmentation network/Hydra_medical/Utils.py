import torch
from torch.utils.data import Dataset as BaseDataset
import os
import numpy as np


class Seg_Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, num_classes: int, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.transforms = transforms
        self.device = "cuda"

    def __getitem__(self, idx):
        images = os.listdir(self.images_dir)
        image = np.load(self.images_dir + '/' + images[idx])
        # image = image.astype(np.float32)

        if self.transforms:
            image = self.transforms(image)

        masks = os.listdir(self.masks_dir)
        mask = np.load(self.masks_dir + '/' + masks[idx])
        if self.transforms:
            mask = self.transforms(mask)
        return image, mask

    def __len__(self):
        return len(os.listdir(self.images_dir))


def make_one_hot(labels, batch_size, num_classes, image_shape_0, image_shape_1):
    one_hot = torch.zeros([batch_size, num_classes, image_shape_0, image_shape_1], dtype=torch.float64)
    labels = labels.unsqueeze(1)
    result = one_hot.scatter_(1, labels.data, 1)
    return result