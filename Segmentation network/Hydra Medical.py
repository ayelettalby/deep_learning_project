import torch
import torch.nn as nn
import os
from torch import utils
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.utils.base import Activation
from typing import Optional, Union, List
import torch.utils.model_zoo as model_zoo
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from Hydra_medical.model import Hydra
from Hydra_medical.Losses import *
from Hydra_medical.Utils import *

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = Hydra(encoder_name="resnet18", encoder_depth=5, encoder_weights='imagenet',
                decoder_use_batchnorm=True, decoder_channels=[256, 128, 64, 32, 16],
                in_channels=3, classes=2, activation='softmax')
model = model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

path = 'E:\Deep learning\Datasets_organized\Prepared_Data\Spleen'
x_train_dir = os.path.join(path, 'Training')
y_train_dir = os.path.join(path, 'Training_Labels')
x_val_dir = os.path.join(path, 'Validation')
y_val_dir = os.path.join(path, 'Validation_Labels')
x_test_dir = os.path.join(path, 'Test')
y_test_dir = os.path.join(path, 'Test_Labels')

train_dataset = Seg_Dataset(x_train_dir, y_train_dir, 2)
val_dataset = Seg_Dataset(x_val_dir, y_val_dir, 2)
batchsize = 4
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
valid_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

criterion = SA_diceloss()  # smp.utils.losses.DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
metrics = [smp.utils.metrics.IoU(threshold=0.5), ]
epochs = 10
num_classes = 2
Training_Loss = []
validation_loss = []
total_steps = len(train_loader)

print(f"{epochs} epochs, {total_steps} total_steps per epoch")
for epoch in range(epochs):
    for i, (images, masks) in enumerate(train_loader, 1):
        masks = masks.type(torch.LongTensor)
        masks = masks.unsqueeze(1)
        # masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        one_hot = torch.DoubleTensor(masks.size(0), num_classes, masks.size(2), masks.size(3)).zero_()
        masks = one_hot.scatter_(1, masks.data, 1)
        masks = masks.double()
        # Forward pass
        outputs = model(images)

        if masks.shape[0] == batchsize:
            loss = criterion(outputs, masks, batchsize, num_classes)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
    Training_Loss.append(loss.item())

    val_total = 0
    total_val_loss = 0
    correct = 0
    iou = 0

    with torch.no_grad():
        for j, (images, masks) in enumerate(valid_loader, 0):
            masks = masks.type(torch.LongTensor)
            masks = masks.unsqueeze(1)
            one_hot = torch.DoubleTensor(masks.size(0), num_classes, masks.size(2), masks.size(3)).zero_()
            masks = one_hot.scatter_(1, masks.data, 1)

            masks = masks.double()
            val_outputs = model(images)
            # softmax = F.log_softmax(outputs, dim=1)
            if val_outputs.shape[0] == batchsize:
                val_loss = criterion(val_outputs, masks, batchsize, num_classes)

            total_val_loss += val_loss
            val_total += 1

            # _, val_predicted = torch.max(val_outputs, 1)
            _, val_predicted = torch.max(val_outputs.data, 1)

            val_predicted = val_predicted.cpu()
            masks = masks.cpu()
            if val_predicted.shape[0] == batchsize:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(val_predicted[1, :, :], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(masks[0, 1, :, :], cmap="gray")
                plt.show()

        val_loss = total_val_loss / (val_total)
        print('val_loss' + '=' + str(val_loss))
        validation_loss.append(val_loss)

plt.figure
plt.plot(Training_Loss, label='Training')
plt.plot(validation_loss, label='Validation')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation losses')
plt.show()



