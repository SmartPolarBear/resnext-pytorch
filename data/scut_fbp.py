import os

import torch
import torch.utils.data

import PIL.Image

import numpy as np

import torchvision.transforms as transforms
import torchvision.transforms.functional

from pathlib import Path


class SCUTFBP_dataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, random_flip: bool = True):
        self.directory = dir
        self.images = []
        self.labels = []
        self.random_flip = random_flip

        for f in os.listdir(dir):
            label = float(f.split('-')[0])
            img = Path(dir) / Path(f)
            self.images.append(str(img))
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = self.images[item]
        label = self.labels[item]

        image = PIL.Image.open(img_path)
        image = transforms.functional.to_tensor(image)

        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)

        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.random_flip and float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)

        return image, torch.tensor([label]).float()
