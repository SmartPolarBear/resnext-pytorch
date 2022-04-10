import argparse

import PIL.Image

import numpy as np

import torch.utils.data

import torch.optim as optim
import torch.nn.functional as F

from model.resnext import resnext101_32x4d

import torchvision.transforms as transforms
import torchvision.transforms.functional

from typing import Final

BEST_MODEL_PATH: Final = 'best_model.pth'


def preprocess(path: str, device):
    image = PIL.Image.open(path)
    image = transforms.functional.to_tensor(image)

    image = image.numpy()[::-1].copy()
    image = torch.from_numpy(image)

    image = transforms.functional.resize(image, (224, 224))
    image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image.unsqueeze(0).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Image used for the demo", required=True)
    parser.add_argument("--device", help="CUDA device used", default='cuda')

    args = parser.parse_args()

    device = torch.device(args.device)

    img_path = args.image

    image = preprocess(img_path, device)

    model = resnext101_32x4d(1)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model = model.to(device)

    model.eval()

    rating = model(image).detach().float().cpu().numpy().flatten()

    print("Rating: {}".format(rating))
