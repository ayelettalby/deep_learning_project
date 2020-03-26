import torch

from typing import Optional, Union, List
import torch.nn as nn
import os
from torch import utils
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from PIL import Image
import csv
from torch.autograd import Variable
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from  segmentation_models_pytorch.utils.base import Activation
from typing import Optional, Union, List
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from segmentation_models_pytorch.encoders.dpn import dpn_encoders
from segmentation_models_pytorch.encoders.vgg import vgg_encoders
from segmentation_models_pytorch.encoders.senet import senet_encoders
from segmentation_models_pytorch.encoders.densenet import densenet_encoders
from segmentation_models_pytorch.encoders.inceptionresnetv2 import inceptionresnetv2_encoders
from segmentation_models_pytorch.encoders.inceptionv4 import inceptionv4_encoders
from segmentation_models_pytorch.encoders.efficientnet import efficient_net_encoders
from segmentation_models_pytorch.encoders.mobilenet import mobilenet_encoders
from segmentation_models_pytorch.encoders.xception import xception_encoders

from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm):
        conv1 = DoubleConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                use_batchnorm=use_batchnorm)
        super().__init__(conv1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, use_transpose_conv=True, mode='nearest'):
        super().__init__()

        self.mode = mode
        self.use_transpose_conv = use_transpose_conv
        self.upconv = nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=2, stride=2)
        if use_transpose_conv:
            self.conv = DoubleConvBlock(int(in_channels / 2), out_channels, kernel_size=3, stride=1, padding=1,
                                        use_batchnorm=use_batchnorm)
        else:
            self.conv = DoubleConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                        use_batchnorm=use_batchnorm)

    def forward(self, x):
        if self.use_transpose_conv:
            x = self.upconv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        x = self.conv(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, activation=None, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        if activation is None or activation == 'identity':
            self.activation = Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax2d':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'logsoftmax':
            self.activation = nn.LogSoftmax()
        else:
            raise ValueError('Activation should be sigmoid/softmax/logsoftmax/None; got {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)


class UnetDecoder2D(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=True, center=False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError("Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(n_blocks,
                                                                                                           len(
                                                                                                               decoder_channels)))
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels)
        out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = Identity()
        blocks = [DecoderBlock(in_ch, out_ch)
                  for in_ch, out_ch in zip(in_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x)

        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
        super(DoubleConvBlock, self).__init__()
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def initialize_decoder(self, module):
        for m in module.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Unet_2D(SegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet18",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: str = 'softmax'):
        super(Unet_2D, self).__init__()

        # encoder
        self.encoder = self.get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                        weights=encoder_weights)

        # decoder
        self.decoder = UnetDecoder2D(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels,
                                     n_blocks=encoder_depth, use_batchnorm=decoder_use_batchnorm,
                                     center=True if encoder_name.startswith("vgg") else False)

        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=classes,
                                                  activation=activation,
                                                  kernel_size=3)

        self.name = 'u-{}'.format(encoder_name)
        self.initialize()

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(*features)
        output = self.segmentation_head(x)
        return output

    def get_encoder(self, name, in_channels=3, depth=5, weights=None):
        Encoder = encoders[name]["encoder"]
        params = encoders[name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            settings = encoders[name]["pretrained_settings"][weights]
            encoder.load_state_dict(model_zoo.load_url(settings["url"]))

        encoder.set_in_channels(in_channels)

        return encoder


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = Unet_2D(encoder_name="resnet18",
                encoder_depth=5,
                encoder_weights="imagenet",
                decoder_use_batchnorm="True",
                decoder_channels=[256, 128, 64, 32, 16],
                in_channels=3,
                classes=2,
                activation='softmax')
model = model.double()
#model.load_state_dict(torch.load('D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/model_weights/model_weights_03_01_20_19_23.pth',map_location=torch.device('cpu')))
#model.cuda(0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

path = 'D:\Documents\ASchool\year 4\prepared\Spleen'
x_train_dir = os.path.join(path, 'Training')
y_train_dir = os.path.join(path, 'Training_Labels')
x_val_dir = os.path.join(path, 'Validation')
y_val_dir = os.path.join(path, 'Validation_Labels')
x_test_dir = os.path.join(path, 'Test')
y_test_dir = os.path.join(path, 'Test_Labels')


class Seg_Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, num_classes: int, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.transforms = transforms
        self.device = "cpu"

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


class SA_diceloss(base.Loss):
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def make_one_hot(self,labels, batch_size, num_classes, image_shape_0, image_shape_1):
        one_hot = torch.zeros([batch_size, num_classes, image_shape_0, image_shape_1], dtype=torch.float64)
        labels = labels.unsqueeze(1)
        result = one_hot.scatter_(1, labels.data, 1)
        return result

    def diceloss(self, masks,outputs,batch_size,num_classes):
        eps = 1e-10
        values, indices = torch.max(outputs, 1)
        y_pred=self.make_one_hot(indices,batch_size,num_classes,indices.size(1),indices.size(2))
        batch_intersection=torch.sum(masks*y_pred,(0,2,3))
        #fp=torch.sum(y_pred,(0,2,3))-batch_intersection
        #fn=torch.sum(masks,(0,2,3))-batch_intersection
        batch_union = torch.sum(y_pred, (0, 2, 3)) + torch.sum(masks, (0, 2, 3))
        loss = (2 * batch_intersection + eps) / (batch_union + eps)
        #loss=(2*batch_intersection+eps)/(2*batch_intersection+2*fn+fp+eps)
        bg=loss[0].item()
        t=loss[1].item()
        total_loss=(bg*0.2+t*0.8)
        return (1-total_loss)

    def forward(self, y_pr, y_gt,batch_size,class_num):
        y_pr = self.activation(y_pr)
        return self.diceloss(y_pr, y_gt,batch_size,class_num)

train_dataset = Seg_Dataset(x_train_dir, y_train_dir, 2)

num_classes = train_dataset.num_classes
# print (train_dataset[1][0].shape)

val_dataset = Seg_Dataset(x_val_dir, y_val_dir, 2)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0)
valid_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=0)

# Use gpu for training if available else use cpu
# device = torch.cuda
# Here is the loss and optimizer definition###
criterion = SA_diceloss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
metrics = [smp.utils.metrics.IoU(threshold=0.5), ]

# The training loop
epochs = 5


batch_size = 3
total_steps = len(train_loader)
total_val_steps =len(valid_loader)
bg_loss=[]
t_loss=[]
weighted_loss=[]
training_loss = []
validation_loss = []

print(f"{epochs} epochs, {total_steps} total_steps per epoch")
for epoch in range(epochs):
    for i, (images, masks) in enumerate(train_loader, 1):
        images = torch.tensor(images)
        masks = torch.tensor(masks)
        masks=masks.unsqueeze(1)
        masks = masks.long()
        one_hot = torch.DoubleTensor(batch_size, num_classes, masks.size(2), masks.size(3)).zero_()
        masks = one_hot.scatter_(1, masks.data, 1)
        # masks = Variable(masks)
        masks = masks.double()

        #images = images.to("cuda")
        #masks = masks.type(torch.LongTensor)
        #masks = masks.to("cuda")

        # Forward pass
        outputs = model(images)
        loss=criterion(masks,outputs,batch_size,num_classes)
        # bg_loss.append(bg)
        # t_loss.append(t)
        # weighted_loss.append(weighted)
        #loss = criterion(outputs, masks)
        loss = torch.tensor(loss)
        loss.requires_grad = True
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}")
    training_loss.append(loss.item())
    total_val_loss = 0
    iou = 0
    val_total = 0

    with torch.no_grad():
        for j, (images, masks) in enumerate(valid_loader, 1):
            images = torch.tensor(images)

            masks = torch.tensor(masks)
            masks = masks.unsqueeze(1)
            masks = masks.long()
            one_hot = torch.DoubleTensor(batch_size, num_classes, masks.size(2), masks.size(3)).zero_()
            masks = one_hot.scatter_(1, masks.data, 1)
            masks = Variable(masks)
            masks = masks.double()

            #images = images.to("cuda")
            #masks = masks.type(torch.LongTensor)
            #masks = masks.to("cuda")

            val_outputs = model(images)
            val_loss = criterion(val_outputs, masks)
            total_val_loss += val_loss.item()

           # _, val_predicted = torch.max(val_outputs, 1)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += masks.size(0)


        val_loss = total_val_loss/total_val_steps
        validation_loss.append(val_loss)
        print('val_loss' + '=' + str(val_loss))

        val_predicted = val_predicted.cpu()
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(val_predicted[0, :, :], cmap="gray")
        # plt.subplot(1, 3, 2)
        # plt.imshow(val_predicted[1, :, :], cmap="gray")
        # plt.subplot(1, 3, 3)
        # plt.imshow(val_predicted[2, :, :], cmap="gray")
        # plt.show()

plt.figure()
plt.plot(training_loss, label='Training')
plt.plot(validation_loss, label='Validation')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation losses')
plt.show()
path_saved_network='./model_weights.pth'
torch.save(model.state_dict(), path_saved_network)
