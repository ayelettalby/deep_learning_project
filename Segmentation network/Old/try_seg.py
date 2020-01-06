import torch
from typing import Optional, Union, List
import torch.nn as nn
import os
from torch import utils
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from PIL import Image
import csv
import segmentation_models_pytorch as smp
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
from ayelet_shiri import unet_2d as F
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from ayelet_shiri.segmentation_models_practice.segmentation_models_pytorch.utils import losses
from scipy import ndimage
#create sample data: (1D slices of training and labels)

def re_sample(slice, end_shape, order=3):
    zoom_factor = [n / float(o) for n, o in zip(end_shape, slice.shape)]
    if not np.all(zoom_factor == (1, 1)):
        data = ndimage.zoom(slice, zoom=zoom_factor, order=order, mode='constant')
    else:
        data = slice
    return data

tmp='D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Spleen data/Val_Lab/spleen_6.nii.gz'
img=nib.load(tmp)
img=img.get_data()
end_shape=(256,256)
# for i in range(90):
#     new=torch.from_numpy(img[:,:,i])
#     new = new.numpy()
#     new=re_sample(new,end_shape)
#     new=torch.from_numpy(new)
#     new=new.unsqueeze(0)
#     new=new.numpy()
#     output=np.empty((3,256,256))
#     output[0,:,:]=new
#     output[1, :, :] = new
#     output[2, :, :] = new
#
#     np.save('D:/Documents/ASchool/year 4/my_seg_try/Labels'+'/Slice' + str(i),output)
a=np.load('D:/Documents/ASchool/year 4/my_seg_try/imgs/Slice2.npy')
print (type(a))

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# model = smp.FPN(encoder_name='efficientnet-b7',
#                        encoder_depth=5,
#                        encoder_weights= "imagenet",
#                        in_channels=1,
#                        classes=1,
#                        activation='sigmoid')
#
# #model.cuda(0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# path='D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Seg_data_try'
# x_train_dir=os.path.join(path,'Training')
# y_train_dir=os.path.join(path, 'Training_Lab')
# x_val_dir=os.path.join(path,'Training')
# y_val_dir=os.path.join(path, 'Val_Lab')
# x_test_dir=os.path.join(path,'Test')
# y_test_dir=os.path.join(path, 'Test_Lab')
#
# def visualize(images):
#     """PLot images in one row."""
#     n = len(images)
#     print (n)
#     #plt.figure(figsize=(10,1))
#     for i, image in enumerate(images[1:10]):
#         plt.subplot(1,10,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         img=np.load(x_train_dir+'/'+image)
#         plt.imshow(np.array(img))
#     plt.show()
#
# t=os.listdir(x_train_dir)
# img=np.load(x_train_dir+'/'+t[1])
# print (img.shape)
# #visualize(t)
#
# class Seg_Dataset(BaseDataset):
#     CLASSES=[0,1]
#     def __init__(
#             self,
#             images_dir,
#             masks_dir,
#             classes=None,
#             augmentation=None,
#             preprocessing=None,
#     ):
#         self.ids = os.listdir(images_dir)
#         self.images_dir=images_dir
#         self.masks_dir=masks_dir
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#
#     def __getitem__(self, i):
#
#         t=os.listdir(self.images_dir)
#         image = np.load(x_train_dir+'/'+t[i])
#
#         # extract certain classes from mask (e.g. cars)
#         masks = os.listdir(self.masks_dir)
#         mask = np.load(y_train_dir+'/'+masks[i])
#
#
#         return image, mask
#
#     def __len__(self):
#         return len(self.ids)
#
# train_dataset=Seg_Dataset(x_train_dir,y_train_dir,classes=['0','1'])
# # plt.figure()
# # for i in range(5):
# #     image, mask = train_dataset[i] # get some sample
# #
# #     plt.subplot(5,2,i*2+1)
# #     plt.imshow(image,cmap='gray')
# #     plt.subplot(5,2,i*2+2)
# #     plt.imshow(mask,cmap='gray')
# # plt.show()
#
# #val_dataset=Seg_Dataset(x_val_dir,y_val_dir,classes=['0','1'])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
# #valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# loss = smp.utils.losses.DiceLoss()
# metrics = [smp.utils.metrics.IoU(threshold=0.5),]
#
# train_epoch = smp.utils.train.TrainEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     optimizer=optimizer,
#     device=None,
#     verbose=True,
# )
#
# # valid_epoch = smp.utils.train.ValidEpoch(
# #     model,
# #     loss=loss,
# #     metrics=metrics,
# #     verbose=True,
# # )
#
# max_score = 0
#
# for i in range(0, 10):
#
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     #valid_logs = valid_epoch.run(valid_loader)
#
#     # do something (save model, change lr, etc.)
#     # if max_score < valid_logs['iou_score']:
#     #     max_score = valid_logs['iou_score']
#     #     torch.save(model, './best_model.pth')
#     #     print('Model saved!')
#
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')