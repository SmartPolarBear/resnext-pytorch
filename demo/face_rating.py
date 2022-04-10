import argparse

import torch.utils.data

import torch.optim as optim
import torch.nn.functional as F

from model.resnext import resnext101_32x4d
from data.scut_fbp import SCUTFBP_dataset

from typing import Final

from pathlib import Path

from tqdm import tqdm

NUM_EPOCHS: Final = 25
BEST_MODEL_PATH: Final = 'best_model.pth'


def train_model(dataset_path: Path, device):
    train_dataset = SCUTFBP_dataset(str(dataset_path), random_flip=True)

    test_percent: Final = 0.1
    num_test: Final = int(test_percent * len(train_dataset))

    train, test = torch.utils.data.random_split(train_dataset, [len(train_dataset) - num_test, num_test])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    model = resnext101_32x4d(1)
    model = model.to(device)

    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(iter(train_loader), position=0, leave=True, desc='Train'):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = F.mse_loss(outputs, labels)

            train_loss += float(loss)

            loss.backward()
            optimizer.step()

        scheduler.step()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        for images, labels in tqdm(iter(test_loader), position=0, leave=True, desc='Test'):
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = F.mse_loss(outputs, labels)

            test_loss += float(loss)
        test_loss /= len(test_loader)

        print('[Epoch {}] Train loss {}, test loss {}'.format(epoch + 1, train_loss, test_loss))
        if test_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss

        scheduler.step()


def test_model(dataset_path: Path, device):
    eval_dataset = SCUTFBP_dataset(str(dataset_path), random_flip=False)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=True,
        num_workers=0
    )

    model = resnext101_32x4d(1)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model = model.to(device)

    model.eval()

    eval_loss = 0.0
    for images, labels in iter(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        eval_loss += float(loss)
    eval_loss /= len(eval_loader)

    print("Evaluation loss: {}".format(eval_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Dataset used for training", required=True)
    parser.add_argument("--eval_data", help="Dataset used for evaluation")
    parser.add_argument("--device", help="CUDA device used", default='cuda')

    args = parser.parse_args()
    device = torch.device(args.device)

    train_model(args.train_data, device)

    if args.eval_data:
        test_model(args.eval_data, device)
