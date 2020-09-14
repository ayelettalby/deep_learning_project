import torch.backends.cudnn as cudnn
import torch
import os
import time
# import wandb
import numpy as np
import torch.nn as nn
from CT_settings import SegSettings
import nibabel as nb

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.losses import DiceLoss as DL
import random
from PIL import Image
#from torchsummary import summary
from torchvision import transforms
import json
import time
import ayelet_shiri.SegmentationModule.CT_only.CT_models as models
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import Subset
import logging
import sys

matplotlib.use('TkAgg')
cudnn.benchmark = True

user='remote'
if user == 'ayelet':
    device ='cpu'
    json_path = r'C:\Users\Ayelet\Desktop\school\fourth_year\deep_learning_project\ayelet_shiri\sample_Data\exp_1\exp_1.json'
elif user=='remote':
    device='cuda:0'
    json_path = r'G:/Deep learning/Datasets_organized/Prepared_Data/CT_experiments/exp_1/exp_1.json'
elif user=='shiri':
    device='cpu'
    json_path = r'F:/Prepared Data/exp_1/exp_1.json'


class Seg_Dataset(BaseDataset):
    def __init__(self, task,images_dir,masks_dir, num_classes, transforms=None):
        self.task=task
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.num_classes = num_classes

    def __getitem__(self,idx):
        images = os.listdir(self.images_dir)
        image = np.load(self.images_dir + '/' + images[idx])

        masks = os.listdir(self.masks_dir)
        mask = np.load(self.masks_dir + '/' + masks[idx])

        if settings.pre_process==True:
            image = pre_processing(image,self.task,settings)
        if settings.augmentation==True:
            image, mask = create_augmentations(image, mask)

        sample={'image':image.astype('float64'), 'mask':mask.astype('float64'), 'task':self.task, 'num_classes':self.num_classes }
        return sample

    def __len__(self):
        return len(os.listdir(self.images_dir))

class DiceLoss(nn.Module):
    def __init__(self, classes, dimension, mask_labels_numeric, mask_class_weights_dict, is_metric):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.dimension = dimension
        self.mask_labels_numeric = mask_labels_numeric
        self.mask_class_weights_dict = mask_class_weights_dict
        self.is_metric = is_metric
        self.eps = 1e-10
        self.tot_weight = torch.sum(torch.Tensor(list(mask_class_weights_dict.values()))).item()

    def forward(self, pred, target):
        # plt.subplot(1,3,1)
        # plt.imshow(pred.cpu().detach().numpy()[0,0,:,:],cmap="gray")
        # plt.subplot(1, 3, 2)
        # plt.imshow(pred.cpu().detach().numpy()[0, 1, :, :], cmap="gray")
        # plt.subplot(1,3,3)
        # plt.imshow(target.cpu().detach().numpy()[0, 1, :, :], cmap="gray")
        # plt.show()
        if self.is_metric:
            if self.classes >1:
                pred = torch.argmax(pred, dim=1)
                # plt.subplot(1, 3, 1)
                # plt.imshow(pred.cpu().detach().numpy()[0, :, :], cmap="gray")
                pred = torch.eye(self.classes)[pred]
                # plt.subplot(1, 3, 2)
                # plt.imshow(pred.cpu().detach().numpy()[0, 1, :, :], cmap="gray")
                pred = pred.transpose(1, 3).to(device)
                # plt.subplot(1, 3, 3)
                # plt.imshow(pred.cpu().detach().numpy()[0, 1, :, :], cmap="gray")
                # plt.show()
            else:
                pred_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                pred_copy[:, 1, :, :][pred[:, 0, :, :]  > 0.5] = 1
                pred_copy[:, 0, :, :][pred[:, 0, :, :] <= 0.5] = 1
                target_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                target_copy[:, 1, :, :][target[:, 0, :, :] == 1.0] = 1
                target_copy[:, 0, :, :][target[:, 0, :, :] == 0.0] = 1

                pred = pred_copy
                target = target_copy.to(device)
        target=target.transpose(2,3)
        # plt.subplot(1, 3, 1)
        # plt.imshow(pred.cpu().detach().numpy()[0, 0, :, :], cmap="gray")
        # plt.subplot(1, 3, 2)
        # plt.imshow(pred.cpu().detach().numpy()[0, 1, :, :], cmap="gray")
        # plt.subplot(1, 3, 3)
        # plt.imshow(target.cpu().detach().numpy()[0, 1, :, :], cmap="gray")
        # plt.show()
        batch_intersection = torch.sum(pred * target.float(), dim=tuple(list(range(2, self.dimension + 2))))
        batch_union = torch.sum(pred, dim=tuple(list(range(2, self.dimension + 2)))) + torch.sum(target.float(),dim=tuple(list(range(2,self.dimension + 2))))
        tumour_dice=None
        background_dice = (2 * batch_intersection[:, self.mask_labels_numeric['background']] + self.eps) / (
                batch_union[:, self.mask_labels_numeric['background']] + self.eps)
        organ_dice = (2 * batch_intersection[:, self.mask_labels_numeric['organ']] + self.eps) / (
                batch_union[:, self.mask_labels_numeric['organ']] + self.eps)

        mean_dice_val = torch.mean((background_dice * self.mask_class_weights_dict['background'] +
                                    organ_dice * self.mask_class_weights_dict['organ']) * 1 / self.tot_weight, dim=0)

        if 'tumour' in self.mask_labels_numeric:
            tumour_dice = (2 * batch_intersection[:, self.mask_labels_numeric['tumour']] + self.eps) / (
                    batch_union[:, self.mask_labels_numeric['tumour']] + self.eps)
            mean_dice_val = torch.mean((background_dice * self.mask_class_weights_dict['background'] +
                                        organ_dice * self.mask_class_weights_dict['organ']+tumour_dice * self.mask_class_weights_dict['tumour']) * 1 / self.tot_weight,dim=0)
            if self.is_metric:
                return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),tumour_dice.mean().item()]

        if self.is_metric:
            return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item()]
        else:
            return -mean_dice_val

def create_augmentations(image,mask):
    augmentation=np.rot90
    p=random.choice([0,1,2])
    if p==1: #1/3 change of augmentation
        new_image=np.zeros(image.shape)
        new_mask=np.zeros(mask.shape)
        k=random.choice([0,1,2,3])
        if k==0:
            augmentation = np.rot90
        if k==1:
            augmentation = np.fliplr
        if k==2:
            augmentation = np.flipud
        if k==3:
            augmentation = np.rot90
            new_mask = augmentation(mask,axes=(1,0))
            for i in range(image.shape[0]):
                new_image[i,:,:] = augmentation(image[i,:,:],axes = (1,0))
            return (new_image.copy(), new_mask.copy())

        new_mask = augmentation(mask)
        for i in range(image.shape[0]):
            new_image[i,:,:] = augmentation(image[i,:,:])
        return (new_image.copy(), new_mask.copy())

    else: #no agumentation
        return(image,mask)

def pre_processing(input_image, task, settings):
    if task == ('spleen' or 'lits' or 'pancreas' or 'hepatic vessel'):  # CT, clipping, Z_Score, normalization btw0-1
        clipped_image = clip(input_image, settings)
        c_n_image = zscore_normalize(clipped_image)
        min_val = np.amin(c_n_image)
        max_val = np.amax(c_n_image)
        eps=0.000001
        final = (c_n_image - min_val) / (max_val - min_val+eps)
        final[final > 1] = 1
        final[final < 0] = 0

    else: #MRI, Z_score, normalization btw 0-1
        norm_image = zscore_normalize(input_image)
        min_val = np.amin(norm_image)
        max_val = np.amax(norm_image)
        eps = 0.000001
        final = (norm_image - min_val) / (max_val - min_val+eps)
        final[final > 1] = 1
        final[final < 0] = 0
        final=norm_image

    return final

def clip(data, settings):
    # clip and normalize
    min_val = settings.min_clip_val
    max_val = settings.max_clip_val
    data[data > max_val] = max_val
    data[data < min_val] = min_val

    return data

def clip_n_normalize(data, settings):
    # clip and normalize
    min_val = settings.min_clip_val
    max_val = settings.max_clip_val
    data = ((data - min_val) / (max_val - min_val))
    data[data > 1] = 1
    data[data < 0] = 0

    return data

def zscore_normalize(img):
    eps=0.0001
    mean = img.mean()
    std = img.std()+eps
    normalized = (img - mean) / std
    return normalized

def save_samples_nimrod(model, iter, epoch, samples_list, snapshot_dir, settings):
    samples_imgs = samples_list
    #samples_imgs = ['ct_122_268_0.4234.npy', 'ct_122_350_0.5529.npy', 'ct_122_365_0.5766.npy', 'ct_122_383_0.6051.npy']
    fig = plt.figure()
    k=0
    organ_to_seg = settings.organ_to_seg
    if organ_to_seg=='liver':
        data_dir = r'I:\DADE\lits\test'
    else:
        data_dir = r'I:\DADE\kits\test'
    for sample_path in samples_imgs:
        image_path = os.path.join(data_dir, sample_path)
        seg_path = sample_path.replace('ct', 'seg')
        seg_path = os.path.join(data_dir, seg_path)
        img_sample = np.load(image_path)
        image = clip_n_normalize(img_sample, settings)
        tensor_transform = transforms.ToTensor()
        image = tensor_transform(image).to(device)
        image = image.unsqueeze(0)
        mask = np.load(seg_path).astype('uint8')
        mask[mask == 2] = 1
        mask = np.eye(2)[mask]
        mask = tensor_transform(mask)
        mask = torch.argmax(mask, dim=0).to(device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        pred = model(image.float())
        _, _, liver_dice = dice(pred, mask, settings)
        liver_dice = round(liver_dice, ndigits=3)

        pred=pred.cpu().detach().numpy()
        pred[pred>0.5]=1
        pred[pred<=0.5]=0

        plt.subplot(len(samples_imgs), 3, 3 * k + 1)
        plt.imshow(image.cpu().numpy()[0, 1,:,:], cmap='gray')
        plt.subplot(len(samples_imgs), 3, 3 * k + 2)
        plt.imshow(pred[0,0,:,:], cmap='gray')
        plt.title('pred iter: {} epoch: {} {} dice: {}'.format(iter, epoch, organ_to_seg, liver_dice))
        plt.subplot(len(samples_imgs), 3, 3* k + 3)
        plt.imshow(mask.cpu().detach().numpy()[0,0,:,:], cmap='gray')

        k += 1
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir, 'pred_{}_{}.{}'.format(iter, epoch, 'png')))

def save_samp(image,mask,task,prediction,epoch,iter,snapshot_dir,loss):
    activation=nn.Softmax(dim=1)
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction1 = prediction.cpu().detach().numpy()
    prediction1 = np.argmax(prediction1, axis=1)
    prediction1 = np.squeeze(prediction1)

    plt.subplot(1, 3, 3)
    plt.imshow(prediction1, cmap="gray")
    plt.title('Prediction')

    # prediction2=activation(prediction)
    # prediction2 = prediction2.cpu().detach().numpy()
    # prediction2 = np.argmax(prediction2, axis=1)
    # prediction2 = np.squeeze(prediction2)
    #
    # plt.subplot(1, 4, 4)
    # plt.imshow(prediction2, cmap="gray")
    # plt.title('Prediction with softmax')

    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Dice: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir,task,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
    plt.close('all')

def save_samp_validation(image,mask,task,prediction,epoch,iter,loss):
    activation=nn.Softmax(dim=1)
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction1 = prediction.cpu().detach().numpy()
    prediction1 = np.argmax(prediction1, axis=1)
    prediction1 = np.squeeze(prediction1)
    plt.subplot(1, 3, 3)
    plt.imshow(prediction1, cmap="gray")
    plt.title('Prediction')

    # prediction2 = activation(prediction)
    # prediction2 = prediction2.cpu().detach().numpy()
    # prediction2 = np.argmax(prediction2, axis=1)
    # prediction2 = np.squeeze(prediction2)
    #
    # plt.subplot(1, 4, 4)
    # plt.imshow(prediction2, cmap="gray")
    # plt.title('Prediction with softmax')

    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Dice: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(settings.validation_snapshot_dir,task,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
    plt.close('all')

def save_samp_val(model, epoch, settings):
    path=r'G:\Deep learning\Datasets_organized\small_dataset'
    spleen_image= os.path.join(path ,'Spleen/Validation/0_slice_15.npy')
    spleen_label = os.path.join(path, 'Spleen/Validation_Labels/0_slice_15.npy')
    prostate_image = os.path.join(path, 'Prostate/Validation/1_slice_13.npy')
    prostate_label = os.path.join(path, 'Prostate/Validation_Labels/1_slice_13.npy')
    pancreas_image=os.path.join(path, 'Pancreas/Validation/1_slice_25.npy')
    pancreas_label = os.path.join(path, 'Pancreas/Validation_Labels/1_slice_25.npy')
    lits_image = os.path.join(path, 'Lits/Validation/0_slice_152.npy')
    lits_label = os.path.join(path, 'Lits/Validation_Labels/0_slice_152.npy')
    hepatic_vessel_image = os.path.join(path, 'Hepatic Vessel/Validation/0_slice_29.npy')
    hepatic_vessel_label = os.path.join(path, 'Hepatic Vessel/Validation_Labels/0_slice_29.npy')
    left_atrial = os.path.join(path, 'Left Atrial/Validation/0_slice_50.npy')
    left_atrial_label = os.path.join(path, 'Left Atrial/Validation_Labels/0_slice_50.npy')
    brain_image = os.path.join(path, 'BRATS/Validation/0_slice_50.npy')
    brain_label = os.path.join(path, 'BRATS/Validation_labels/0_slice_50.npy')
    i = 0
    image_list=[spleen_image,prostate_image,pancreas_image,brain_image,lits_image,hepatic_vessel_image,left_atrial]
    label_list = [spleen_label, prostate_label, pancreas_label, brain_label, lits_label, hepatic_vessel_label,left_atrial_label]
    for task in ['spleen','prostate','pancreas','brain','lits','hepatic_vessel','left_atrial']:
        image=np.load(image_list[i]).astype('float64')
        label=np.load(label_list[i]).astype('float64')
        image=torch.from_numpy(image)
        label=torch.from_numpy(label)
        label=label.type(torch.LongTensor)
        label = label.view(1,label.shape[0], label.shape[0])
        label=label.unsqueeze(0)
        image=image.unsqueeze(0)
        image = image.to(device)
        label = label.to(device)
        output=model(image,[task])
        activation=nn.Softmax(dim=1)
        image=activation(image)
        num_class=2
        one_hot = torch.DoubleTensor(label.size(0), num_class, label.size(2), label.size(3)).zero_()
        one_hot = one_hot.to(device)
        labels_dice = one_hot.scatter_(1, label.data, 1)
        loss = dice(output, labels_dice, num_class, settings)
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image.cpu().detach().numpy()[0,0, :, :], cmap="gray")
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(label.cpu().detach().numpy()[0,0,:,:], cmap="gray")
        plt.title('Mask(GT)')
        plt.subplot(1, 3, 3)
        plt.imshow(output.cpu().detach().numpy()[0][1], cmap="gray")
        plt.title('Prediction')
        plt.suptitle('Task: ' + task + ' Epoch: ' + str(epoch) + ' Loss: '+str(loss[0]) )
        plt.tight_layout()
        fig.savefig(os.path.join(settings.val_snapshot_dir, task, '{}_pred_ep_{}.{}'.format(task,epoch, 'png')))
        plt.close('all')
        i+=1

    return

def dice(pred, target, num_classes,settings):
    if num_classes==2:
        mask_labels={'background': 0,  'organ': 1}
        loss_weights= {'background': 1, 'organ': 10}
    elif num_classes==3: ## pancreas only
        mask_labels = {'background': 0, 'organ': 1, 'tumour':2} ##check if this is true
        loss_weights = {'background': 1, 'organ': 10,'tumour':20}
    dice_measurement = DiceLoss(classes=num_classes,
                               dimension=settings.dimension,
                               mask_labels_numeric=mask_labels,
                               mask_class_weights_dict=loss_weights,
                               is_metric=True)
    [*dices] = dice_measurement(pred, target)
    return dices

def make_one_hot(labels, batch_size, num_classes, image_shape_0, image_shape_1):
    one_hot = torch.zeros([batch_size, num_classes, image_shape_0, image_shape_1], dtype=torch.float64)
    one_hot = one_hot.to(device)
    labels = labels.unsqueeze(1)
    result = one_hot.scatter_(1, labels.data, 1)
    return result

def my_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logging.basicConfig(
        filename=logger_name,
        filemode='w',
        format='%(asctime)s, %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    logger.addHandler(file_handler)
    return logger

def plot_graph(num_epochs,settings,train_organ_dice_tot,train_background_dice_tot,train_loss_tot,
               val_organ_dice_tot, val_background_dice_tot,val_loss_tot,
               train_organ_dice_spleen, train_background_dice_spleen, train_loss_tot_spleen,
               val_organ_dice_spleen,val_background_dice_spleen,val_loss_tot_spleen,
               train_organ_dice_pancreas,  train_background_dice_pancreas, train_loss_tot_pancreas,
               val_organ_dice_pancreas, val_background_dice_pancreas, val_loss_tot_pancreas,
               train_organ_dice_lits, train_background_dice_lits,train_loss_tot_lits,
               val_organ_dice_lits, val_background_dice_lits, val_loss_tot_lits,train_organ_dice_kits, train_background_dice_kits,train_loss_tot_kits,
               val_organ_dice_kits, val_background_dice_kits, val_loss_tot_kits):

    plt.figure()  # Total dice losses
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_tot, 'r')
    plt.plot(x, train_background_dice_tot, 'b')
    plt.plot(x, train_loss_tot, 'c')
    plt.plot(x, val_organ_dice_tot, 'y')
    plt.plot(x, val_background_dice_tot, 'k')
    plt.plot(x, val_loss_tot, 'g')
    plt.title('Total Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss','Val Organ Dice', 'Val Background Dice', 'Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Total Training & Validation Loss w_dice.png'))

    plt.figure()  # Total losses
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_loss_tot, 'c')
    plt.plot(x, val_loss_tot, 'g')
    plt.title('Training & Validation CE Loss vs Epoch Num')
    plt.legend([ 'Training', 'Vaidation',
                'Val CE Loss'], loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Total Training & Validation Loss.png'))

    plt.figure()  # spleen
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_spleen, 'r')
    plt.plot(x, train_background_dice_spleen, 'b')
    plt.plot(x, train_loss_tot_spleen, 'c')
    plt.plot(x, val_organ_dice_spleen, 'y')
    plt.plot(x, val_background_dice_spleen, 'k')
    plt.plot(x, val_loss_tot_spleen, 'g')
    plt.title('Spleen Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice','Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Spleen Training & Validation Loss.png'))


    plt.figure()  # pancreas
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_pancreas, 'r')
    plt.plot(x, train_background_dice_pancreas, 'b')
    plt.plot(x, train_loss_tot_pancreas, 'c')

    plt.plot(x, val_organ_dice_pancreas, 'y')
    plt.plot(x, val_background_dice_pancreas, 'k')
    plt.plot(x, val_loss_tot_pancreas, 'g')

    plt.title('Pancreas Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss','Val Organ Dice', 'Val Background Dice','Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Pancreas Training & Validation Loss.png'))

    plt.figure()  # lits
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_lits, 'r')
    plt.plot(x, train_background_dice_lits, 'b')
    plt.plot(x, train_loss_tot_lits, 'c')
    plt.plot(x, val_organ_dice_lits, 'y')
    plt.plot(x, val_background_dice_lits, 'k')
    plt.plot(x, val_loss_tot_lits, 'g')
    plt.title('Lits Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice','Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Lits Training & Validation Loss.png'))

    plt.figure()  # kits
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_kits, 'r')
    plt.plot(x, train_background_dice_kits, 'b')
    plt.plot(x, train_loss_tot_kits, 'c')
    plt.plot(x, val_organ_dice_kits, 'y')
    plt.plot(x, val_background_dice_kits, 'k')
    plt.plot(x, val_loss_tot_kits, 'g')
    plt.title('kits Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice','Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'kits Training & Validation Loss.png'))
    return

def dataloader_CT(settings, batch_size):
    train_dataset_spleen = Seg_Dataset('spleen', settings.data_dir_spleen + '/Training',
                                       settings.data_dir_spleen + '/Training_Labels', 2)
    val_dataset_spleen = Seg_Dataset('spleen', settings.data_dir_spleen + '/Validation',
                                     settings.data_dir_spleen + '/Validation_Labels', 2)

    train_dataset_pancreas = Seg_Dataset('pancreas', settings.data_dir_pancreas + '/Training',
                                         settings.data_dir_pancreas + '/Training_Labels', 2)
    val_dataset_pancreas = Seg_Dataset('pancreas', settings.data_dir_pancreas + '/Validation',
                                       settings.data_dir_pancreas + '/Validation_Labels', 2)
    train_dataset_lits = Seg_Dataset('lits', settings.data_dir_lits + '/Training',
                                     settings.data_dir_lits + '/Training_Labels', 2)
    val_dataset_lits = Seg_Dataset('lits', settings.data_dir_lits + '/Validation',
                                   settings.data_dir_lits + '/Validation_Labels', 2)
    train_dataset_kits = Seg_Dataset('kits', settings.data_dir_kits + '/Training',
                                     settings.data_dir_kits + '/Training_Labels', 2)
    val_dataset_kits = Seg_Dataset('kits', settings.data_dir_kits + '/Validation',
                                   settings.data_dir_kits + '/Validation_Labels', 2)

    train_dataset_list = [train_dataset_spleen,train_dataset_lits,
                          train_dataset_pancreas,train_dataset_kits]
    val_dataset_list = [val_dataset_kits,val_dataset_spleen,val_dataset_lits,
                        val_dataset_pancreas]

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader,val_dataloader

def train(setting_dict, exp_ind):
    global settings
    settings = SegSettings(setting_dict, write_logger=True)
    logger = my_logger(settings.simulation_folder + '\logger')
    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)
    model.to(device)
    model = model.double()
    #summary(model, tuple(settings.input_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ## initialization of all variables for plots
    train_loss_tot = [] ##cross entropy
    train_loss_tot_spleen = []
    train_loss_tot_lits = []
    train_loss_tot_pancreas = []
    train_loss_tot_kits = []

    train_organ_dice_tot = []
    train_organ_dice_spleen = []
    train_organ_dice_lits = []
    train_organ_dice_pancreas = []
    train_organ_dice_kits = []

    train_background_dice_tot = []
    train_background_dice_spleen = []
    train_background_dice_lits = []
    train_background_dice_pancreas = []
    train_background_dice_kits = []


    val_loss_tot = []  ##cross entropy
    val_loss_tot_spleen = []
    val_loss_tot_lits = []
    val_loss_tot_pancreas = []
    val_loss_tot_kits = []

    val_organ_dice_tot = []
    val_organ_dice_spleen = []
    val_organ_dice_lits = []
    val_organ_dice_pancreas = []
    val_organ_dice_kits = []

    val_background_dice_tot = []
    val_background_dice_spleen = []
    val_background_dice_lits = []
    val_background_dice_pancreas = []
    val_background_dice_kits = []

    train_total_dice_tot=[]
    val_total_dice_tot=[]

    batch_size = settings.batch_size
    train_dataloader, val_dataloader=dataloader_CT(settings,batch_size)

    print('Training... ')
    num_epochs=10
    for epoch in range(0, num_epochs):
         epoch_start_time = time.time()
         train_loss_tot_cur = []  ##cross entropy
         train_loss_tot_spleen_cur = []
         train_loss_tot_lits_cur = []
         train_loss_tot_pancreas_cur = []
         train_loss_tot_kits_cur = []

         train_organ_dice_tot_cur = []
         train_organ_dice_spleen_cur = []
         train_organ_dice_lits_cur = []
         train_organ_dice_pancreas_cur = []
         train_organ_dice_kits_cur = []

         train_background_dice_tot_cur = []
         train_background_dice_spleen_cur = []
         train_background_dice_lits_cur = []
         train_background_dice_pancreas_cur = []
         train_background_dice_kits_cur = []

         val_loss_tot_cur = []  ##cross entropy
         val_loss_tot_spleen_cur = []
         val_loss_tot_lits_cur = []
         val_loss_tot_pancreas_cur = []
         val_loss_tot_kits_cur = []

         val_organ_dice_tot_cur = []
         val_organ_dice_spleen_cur = []
         val_organ_dice_lits_cur = []
         val_organ_dice_pancreas_cur = []
         val_organ_dice_kits_cur = []

         val_background_dice_tot_cur = []
         val_background_dice_spleen_cur = []
         val_background_dice_lits_cur = []
         val_background_dice_pancreas_cur = []
         val_background_dice_kits_cur = []

         total_steps = len(train_dataloader)
         for i,sample in enumerate(train_dataloader,1):
             epoch_start_time = time.time()
             model.train()
             print(sample['task'])
             images=sample['image'].double()
             masks = sample['mask'].type(torch.LongTensor)
             masks = masks.unsqueeze(1)
             masks = masks.type(torch.LongTensor)
             images=images.to(device)
             masks = masks.to(device)


             #Forward pass
             outputs = model(images,sample['task'])
             outputs = outputs.to(device)


             if epoch==5:
                 optimizer = torch.optim.Adam(model.parameters(), lr=0.00001/2)
             if epoch==7:
                 optimizer = torch.optim.Adam(model.parameters(), lr=0.00001/4)
             if epoch==9:
                 optimizer = torch.optim.Adam(model.parameters(), lr=0.00001/ 8)

             loss = criterion(outputs.double(), masks[:,0,:,:])

             # Backward and optimize
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             #print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
             logger.info('current task: ' + sample['task'][0])
             logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )


             one_hot = torch.DoubleTensor(masks.size(0), sample['num_classes'][0], masks.size(2), masks.size(3)).zero_()
             one_hot = one_hot.to(device)
             masks_dice = one_hot.scatter_(1, masks.data, 1)
             activation=nn.Softmax(dim=1)
             # outputs_dice=one_hot.scatter_(1,outputs.type(torch.LongTensor),1)
             dices = dice(activation(outputs), masks_dice, sample['num_classes'][0], settings)
             mean_dice = dices[0]
             background_dice = dices[1]
             organ_dice = dices[2]


             if sample['task'][0] == 'lits':
                 train_organ_dice_lits_cur.append(organ_dice)
                 train_background_dice_lits_cur.append(background_dice)
                 train_loss_tot_lits_cur.append(loss.item())
             if sample['task'][0] == 'spleen':
                 train_organ_dice_spleen_cur.append(organ_dice)
                 train_background_dice_spleen_cur.append(background_dice)
                 train_loss_tot_spleen_cur.append(loss.item())
             if sample['task'][0] == 'pancreas':
                 train_organ_dice_pancreas_cur.append(organ_dice)
                 train_background_dice_pancreas_cur.append(background_dice)
                 train_loss_tot_pancreas_cur.append(loss.item())
             if sample['task'][0] == 'kits':
                 train_organ_dice_kits_cur.append(organ_dice)
                 train_background_dice_kits_cur.append(background_dice)
                 train_loss_tot_kits_cur.append(loss.item())


             train_organ_dice_tot_cur.append(organ_dice)
             train_background_dice_tot_cur.append(background_dice)
             train_loss_tot_cur.append(loss.item())

             if i % 30 == 0:
                 save_samp(sample['image'][0], sample['mask'][0], sample['task'][0], outputs, epoch, i,
                           settings.snapshot_dir, organ_dice)

                 # save_out = outputs.cpu().detach().numpy()
                 #
                 # save_samp(sample['image'][0], sample['mask'][0], sample['task'][0], save_out[0][1], epoch, i,
                 #           settings.snapshot_dir, organ_dice)

             if (i + 1) % 50 == 0:

                 print('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                       'iter: {}/{}'.format(np.mean(train_loss_tot_cur),
                                            np.mean(train_organ_dice_tot_cur),
                                            np.mean(train_background_dice_tot_cur),
                                            i + 1, len(train_dataloader)))
                 logger.info('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                              'iter: {}/{}'.format(np.mean(train_loss_tot_cur),
                                                   np.mean(train_organ_dice_tot_cur),
                                                   np.mean(train_background_dice_tot_cur),
                                                   i + 1, len(train_dataloader)))

         train_organ_dice_lits.append(np.mean(train_organ_dice_lits_cur))
         train_background_dice_lits.append(np.mean(train_background_dice_lits_cur))
         train_loss_tot_lits.append(np.mean(train_loss_tot_lits_cur))

         train_organ_dice_spleen.append(np.mean(train_organ_dice_spleen_cur))
         train_background_dice_spleen.append(np.mean(train_organ_dice_spleen_cur))
         train_loss_tot_spleen.append(np.mean(train_loss_tot_spleen_cur))

         train_organ_dice_pancreas.append(np.mean(train_organ_dice_pancreas_cur))
         train_background_dice_pancreas.append(np.mean(train_background_dice_pancreas_cur))
         train_loss_tot_pancreas.append(np.mean(train_loss_tot_pancreas_cur))

         train_organ_dice_kits.append(np.mean(train_organ_dice_kits_cur))
         train_background_dice_kits.append(np.mean(train_background_dice_kits_cur))
         train_loss_tot_kits.append(np.mean(train_loss_tot_kits_cur))

         train_organ_dice_tot.append(np.mean(train_organ_dice_tot_cur))
         train_background_dice_tot.append(np.mean(train_background_dice_tot_cur))
         train_loss_tot.append(np.mean(train_loss_tot_cur)) #cross entropy

         torch.save(model.state_dict(), os.path.join(settings.checkpoint_dir, 'exp_{}_epoch_{}.pt'.format(exp_ind,epoch)))
         torch.save(model.encoder.state_dict(), os.path.join(settings.checkpoint_dir, 'encoder_exp_{}_epoch_{}.pt'.format(exp_ind,epoch)))

         total_steps=len(val_dataloader)

         for i, data in enumerate(val_dataloader):
             torch.no_grad()
                # model.eval()

             images = data['image'].double()
             masks = data['mask'].type(torch.LongTensor)
             masks = masks.unsqueeze(1)
             images = images.to(device)
             masks = masks.to(device)
             # one_hot = torch.DoubleTensor(masks.size(0), data['num_classes'][0], masks.size(2),
             #                              masks.size(3)).zero_()
             # one_hot = one_hot.to(device)
             #masks = one_hot.scatter_(1, masks.data, 1)

             outputs = model(images, data['task'])
             outputs = outputs.to(device)

             # visualize_features(model,outputs, images, sample['task'])
             loss = criterion(outputs.double(), masks[:,0,:,:])
             print(f"Validation Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
             logger.info('current task: ' + data['task'][0])
             logger.info(f"Validation Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )

             one_hot = torch.DoubleTensor(masks.size(0), data['num_classes'][0], masks.size(2), masks.size(3)).zero_()
             one_hot = one_hot.to(device)
             masks_dice = one_hot.scatter_(1, masks.data, 1)
             activation = nn.Softmax(dim=1)
             dices = dice(activation(outputs), masks_dice, data['num_classes'][0], settings)
             mean_dice = dices[0]
             background_dice = dices[1]
             organ_dice = dices[2]

             if i%30==0:
                 # save_out = outputs.cpu().detach().numpy()
                 # save_samp_validation(data['image'][0], data['mask'][0], data['task'][0], save_out[0][1], epoch, i,
                 #                     organ_dice)
                 save_samp_validation(data['image'][0], data['mask'][0], data['task'][0], outputs, epoch, i,
                                      organ_dice)
             val_organ_dice_tot_cur.append(organ_dice)
             val_background_dice_tot_cur.append(background_dice)
             val_loss_tot_cur.append(loss.item())

             if data['task'][0] == 'lits':
                 val_organ_dice_lits_cur.append(organ_dice)
                 val_background_dice_lits_cur.append(background_dice)
                 val_loss_tot_lits_cur.append(loss.item())
             if data['task'][0] == 'spleen':
                 val_organ_dice_spleen_cur.append(organ_dice)
                 val_background_dice_spleen_cur.append(background_dice)
                 val_loss_tot_spleen_cur.append(loss.item())
             if data['task'][0] == 'pancreas':
                 val_organ_dice_pancreas_cur.append(organ_dice)
                 val_background_dice_pancreas_cur.append(background_dice)
                 val_loss_tot_pancreas_cur.append(loss.item())
             if data['task'][0] == 'kits':
                 val_organ_dice_kits_cur.append(organ_dice)
                 val_background_dice_kits_cur.append(background_dice)
                 val_loss_tot_kits_cur.append(loss.item())


         val_organ_dice_lits.append(np.mean(val_organ_dice_lits_cur))
         val_background_dice_lits.append(np.mean(val_background_dice_lits_cur))
         val_loss_tot_lits.append(np.mean(val_loss_tot_lits_cur))

         val_organ_dice_spleen.append(np.mean(val_organ_dice_spleen_cur))
         val_background_dice_spleen.append(np.mean(val_organ_dice_spleen_cur))
         val_loss_tot_spleen.append(np.mean(val_loss_tot_spleen_cur))

         val_organ_dice_pancreas.append(np.mean(val_organ_dice_pancreas_cur))
         val_background_dice_pancreas.append(np.mean(val_background_dice_pancreas_cur))
         val_loss_tot_pancreas.append(np.mean(val_loss_tot_pancreas_cur))

         val_organ_dice_kits.append(np.mean(val_organ_dice_kits_cur))
         val_background_dice_kits.append(np.mean(val_background_dice_kits_cur))
         val_loss_tot_kits.append(np.mean(val_loss_tot_kits_cur))

         val_organ_dice_tot.append(np.mean(val_organ_dice_tot_cur))
         val_background_dice_tot.append(np.mean(val_background_dice_tot_cur))
         val_loss_tot.append(np.mean(val_loss_tot_cur)) #cross entropy

         np.save(settings.dices_dir + r'\train_organ_dice_spleen_epoch_{}.npy'.format(epoch), np.array(train_organ_dice_spleen))
         np.save(settings.dices_dir + r'\train_organ_dice_pancreas_epoch_{}.npy'.format(epoch), np.array(train_organ_dice_pancreas))
         np.save(settings.dices_dir + r'\train_organ_dice_lits_epoch_{}.npy'.format(epoch), np.array(train_organ_dice_lits))
         np.save(settings.dices_dir + r'\train_organ_dice_kits_epoch_{}.npy'.format(epoch), np.array(train_organ_dice_kits))
         np.save(settings.dices_dir + r'\train_organ_dice_tot_epoch_{}.npy'.format(epoch), np.array(train_organ_dice_tot))

         np.save(settings.dices_dir + r'\val_organ_dice_spleen_epoch_{}.npy'.format(epoch), np.array(val_organ_dice_spleen))
         np.save(settings.dices_dir + r'\val_organ_dice_pancreas_epoch_{}.npy'.format(epoch), np.array(val_organ_dice_pancreas))
         np.save(settings.dices_dir + r'\val_organ_dice_lits_epoch_{}.npy'.format(epoch), np.array(val_organ_dice_lits))
         np.save(settings.dices_dir + r'\val_organ_dice_kits_epoch_{}.npy'.format(epoch), np.array(val_organ_dice_kits))
         np.save(settings.dices_dir + r'\val_organ_dice_tot_epoch_{}.npy'.format(epoch), np.array(val_organ_dice_tot))

         print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,(time.time() - epoch_start_time) / 60))
         print('train loss: {} val_loss: {}'.format(np.mean(train_loss_tot_cur), np.mean(val_loss_tot_cur)))
         print('train organ dice: {}  train background dice: {} val organ dice: {}  val background dice: {}'.format(
            np.mean(train_organ_dice_tot_cur), np.mean(train_background_dice_tot_cur), np.mean(val_organ_dice_tot_cur),
            np.mean(val_background_dice_tot_cur)))

    plot_graph(num_epochs,settings,train_organ_dice_tot,train_background_dice_tot,train_loss_tot,val_organ_dice_tot, val_background_dice_tot,val_loss_tot,
               train_organ_dice_spleen, train_background_dice_spleen, train_loss_tot_spleen,val_organ_dice_spleen,val_background_dice_spleen,val_loss_tot_spleen,
               train_organ_dice_pancreas,  train_background_dice_pancreas, train_loss_tot_pancreas,val_organ_dice_pancreas, val_background_dice_pancreas, val_loss_tot_pancreas,
               train_organ_dice_lits, train_background_dice_lits,train_loss_tot_lits,val_organ_dice_lits, val_background_dice_lits, val_loss_tot_lits,
               train_organ_dice_kits, train_background_dice_kits,train_loss_tot_kits,val_organ_dice_kits, val_background_dice_kits, val_loss_tot_kits)


    # if not os.path.exists(settings.simulation_folder + r'\dices'):
    #     os.mkdir(settings.simulation_folder + r'\dices')
    # np.save(settings.simulation_folder + r'\dices\train_organ_dice_spleen.npy',np.array(train_organ_dice_spleen))
    # np.save(settings.simulation_folder + r'\dices\train_organ_dice_pancreas.npy', np.array(train_organ_dice_pancreas))
    # np.save(settings.simulation_folder + r'\dices\train_organ_dice_lits.npy', np.array(train_organ_dice_lits))
    # np.save(settings.simulation_folder + r'\dices\train_organ_dice_kits.npy', np.array(train_organ_dice_kits))
    # np.save(settings.simulation_folder + r'\dices\train_organ_dice_tot.npy', np.array(train_organ_dice_tot))
    #
    # np.save(settings.simulation_folder + r'\dices\val_organ_dice_spleen.npy',np.array(val_organ_dice_spleen))
    # np.save(settings.simulation_folder + r'\dices\val_organ_dice_pancreas.npy', np.array(val_organ_dice_pancreas))
    # np.save(settings.simulation_folder + r'\dices\val_organ_dice_lits.npy', np.array(val_organ_dice_lits))
    # np.save(settings.simulation_folder + r'\dices\val_organ_dice_lits.npy', np.array(val_organ_dice_kits))
    # np.save(settings.simulation_folder + r'\dices\val_organ_dice_tot.npy', np.array(val_organ_dice_tot))


if __name__ == '__main__':
    exp_ind =3
    print('start with experiment: {}'.format(exp_ind))
    with open(r'G:\Deep learning\Datasets_organized\Prepared_Data\CT_experiments\exp_{}\exp_{}.json'.format(
            exp_ind, exp_ind)) as json_file:
        setting_dict = json.load(json_file)


    train(setting_dict, exp_ind=exp_ind)
