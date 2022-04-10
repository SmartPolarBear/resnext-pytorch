import torch.utils.data

import torch.optim as optim
import torch.nn.functional as F

from model.resnext import resnext50_32x4d
from data.scut_fbp import SCUTFBP_dataset

from typing import Final

train_dataset = SCUTFBP_dataset('data/datasets/face_data_5/face_image_train', random_flip=True)

test_percent: Final = 0.1
num_test: Final = int(test_percent * len(train_dataset))

train, test = torch.utils.data.random_split(train_dataset, [len(train_dataset) - num_test, num_test])

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

model = resnext50_32x4d(1)

device = torch.device('cuda')
model = model.to(device)

NUM_EPOCHS: Final = 70
BEST_MODEL_PATH: Final = 'best_model.pth'

best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in iter(train):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    for images, labels in iter(test):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += float(loss)
    test_loss /= len(test_loader)

    print('%f, %f' % (train_loss, test_loss))
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss

eval_dataset = SCUTFBP_dataset('data/datasets/face_data_5/face_image_train', random_flip=False)
