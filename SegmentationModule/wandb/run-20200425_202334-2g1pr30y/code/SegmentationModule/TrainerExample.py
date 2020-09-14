import torch.backends.cudnn as cudnn
import torch
import os
import wandb
import numpy as np
import torch.nn as nn
from SegmentationSettings import SegSettings
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
import ayelet_shiri.SegmentationModule.Models as models
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
    device='cuda:1'
    json_path = r'G:/Deep learning/Datasets_organized/small_dataset/Experiments/exp_1/exp_1.json'
elif user=='shiri':
    device='cpu'
    json_path = r'F:/Prepared Data/exp_1/exp_1.json'

# with open(json_path) as f:
#   setting_dict = json.load(f)
# settings= SegSettings(setting_dict, write_logger=True)

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
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Loss: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir,task,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
    plt.close('all')

def save_samp_validation(image,mask,task,prediction,epoch,iter,loss):
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Loss: '+ str(loss))
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
        if task=='pancreas':
            num_class=3
        else:
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

def generate_batched_dataset(dataset_list,indices,total_dataset,batch_size):
    while len(indices) != 0:
        p_task = random.randint(0, len(indices) - 1)
        batch_ind = random.sample(indices[p_task], batch_size)
        subset = Subset(dataset_list[p_task], batch_ind)

        total_dataset = torch.utils.data.ConcatDataset([total_dataset, subset])

        for j in batch_ind:
            indices[p_task].remove(j)

        if len(indices[p_task]) <= 1:
            del (indices[p_task])

    return (total_dataset)

def visualize_features(model,outputs,image,task):
    modules = list(model.children())
    mod = nn.Sequential(*modules)[0]
    print (mod)
    for p in mod.parameters():
        p.requires_grad = False

    mod = mod.double()
    out = mod(image)
    feature = out[0].numpy()
    print (feature.shape)
    plt.imshow(feature[0,0,:,:], cmap="gray")
    plt.show()

def weight_vis(model,epoch,settings):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights=m.weight.data

    print (weights.shape)
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(weights[0, i, :, :].cpu().detach().numpy(), cmap="gray")
    plt.savefig(os.path.join(settings.weights_dir, 'weights_{}.{}'.format(epoch, 'png')))
    plt.close('all')


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
               train_organ_dice_brain, train_background_dice_brain,train_loss_tot_brain,
               val_organ_dice_brain, val_background_dice_brain, val_loss_tot_brain,
               train_organ_dice_prostate,  train_background_dice_prostate, train_loss_tot_prostate,
               val_organ_dice_prostate, val_background_dice_prostate, val_loss_tot_prostate,
               train_organ_dice_left_atrial, train_background_dice_left_atrial, train_loss_tot_left_atrial,
               val_organ_dice_left_atrial, val_background_dice_left_atrial,val_loss_tot_left_atrial,
               train_organ_dice_hepatic_vessel, train_background_dice_hepatic_vessel,train_loss_tot_hepatic_vessel,
               val_organ_dice_hepatic_vessel, val_background_dice_hepatic_vessel,val_loss_tot_hepatic_vessel,
               train_organ_dice_pancreas,  train_background_dice_pancreas, train_loss_tot_pancreas,
               val_organ_dice_pancreas, val_background_dice_pancreas, val_loss_tot_pancreas,
               train_tumour_dice_pancreas,val_tumour_dice_pancreas,
               train_organ_dice_lits, train_background_dice_lits,train_loss_tot_lits,
               val_organ_dice_lits, val_background_dice_lits, val_loss_tot_lits):

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

    plt.figure()  # brain
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_brain, 'r')
    plt.plot(x, train_background_dice_brain, 'b')
    plt.plot(x, train_loss_tot_brain, 'c')
    plt.plot(x, val_organ_dice_brain, 'y')
    plt.plot(x, val_background_dice_brain, 'k')
    plt.plot(x, val_loss_tot_brain, 'g')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice', 'Val CE Loss'],loc='upper left')
    plt.title('Brain Training & Validation Loss and Dice vs Epoch Num')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Brain Training & Validation Loss.png'))

    plt.figure()  # prostate
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_prostate, 'r')
    plt.plot(x, train_background_dice_prostate, 'b')
    plt.plot(x, train_loss_tot_prostate, 'c')
    plt.plot(x, val_organ_dice_prostate, 'y')
    plt.plot(x, val_background_dice_prostate, 'k')
    plt.plot(x, val_loss_tot_prostate, 'g')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice', 'Val CE Loss'],loc='upper left')
    plt.title('Prostate Training & Validation Loss and Dice vs Epoch Num')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Prostate Training & Validation Loss.png'))

    plt.figure()  # left atrial
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_left_atrial, 'r')
    plt.plot(x, train_background_dice_left_atrial, 'b')
    plt.plot(x, train_loss_tot_left_atrial, 'c')
    plt.plot(x, val_organ_dice_left_atrial, 'y')
    plt.plot(x, val_background_dice_left_atrial, 'k')
    plt.plot(x, val_loss_tot_left_atrial, 'g')
    plt.title('Left Atrial Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice','Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Left Atrial Training & Validation Loss.png'))

    plt.figure()  # hepatic vessel
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_hepatic_vessel, 'r')
    plt.plot(x, train_background_dice_hepatic_vessel, 'b')
    plt.plot(x, train_loss_tot_hepatic_vessel, 'c')
    plt.plot(x, val_organ_dice_hepatic_vessel, 'y')
    plt.plot(x, val_background_dice_hepatic_vessel, 'k')
    plt.plot(x, val_loss_tot_hepatic_vessel, 'g')
    plt.title('Hepatic Vessel Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice','Val CE Loss'],loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir, 'Hepatic Vessel Training & Validation Loss.png'))

    plt.figure()  # pancreas
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, train_organ_dice_pancreas, 'r')
    plt.plot(x, train_background_dice_pancreas, 'b')
    plt.plot(x, train_loss_tot_pancreas, 'c')
    plt.plot(x, train_tumour_dice_pancreas, 'coral')
    plt.plot(x, val_organ_dice_pancreas, 'y')
    plt.plot(x, val_background_dice_pancreas, 'k')
    plt.plot(x, val_loss_tot_pancreas, 'g')
    plt.plot(x, val_tumour_dice_pancreas, 'm')
    plt.title('Pancreas Training & Validation Loss and Dice vs Epoch Num')
    plt.legend(['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Train Tumour Loss','Val Organ Dice', 'Val Background Dice','Val CE Loss','Val Tumour Loss'],loc='upper left')
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

    return

def dataloader_8(settings, batch_size):
    train_dataset_spleen = Seg_Dataset('spleen', settings.data_dir_spleen + '/Training',
                                       settings.data_dir_spleen + '/Training_Labels', 2)
    val_dataset_spleen = Seg_Dataset('spleen', settings.data_dir_spleen + '/Validation',
                                     settings.data_dir_spleen + '/Validation_Labels', 2)
    train_dataset_prostate = Seg_Dataset('prostate', settings.data_dir_prostate + '/Training',
                                         settings.data_dir_prostate + '/Training_Labels', 2)
    val_dataset_prostate = Seg_Dataset('prostate', settings.data_dir_prostate + '/Validation',
                                       settings.data_dir_prostate + '/Validation_Labels', 2)
    train_dataset_pancreas = Seg_Dataset('pancreas', settings.data_dir_pancreas + '/Training',
                                         settings.data_dir_pancreas + '/Training_Labels', 3)
    val_dataset_pancreas = Seg_Dataset('pancreas', settings.data_dir_pancreas + '/Validation',
                                       settings.data_dir_pancreas + '/Validation_Labels', 3)
    train_dataset_lits = Seg_Dataset('lits', settings.data_dir_lits + '/Training',
                                     settings.data_dir_lits + '/Training_Labels', 2)
    val_dataset_lits = Seg_Dataset('lits', settings.data_dir_lits + '/Validation',
                                   settings.data_dir_lits + '/Validation_Labels', 2)
    train_dataset_left_atrial = Seg_Dataset('left_atrial', settings.data_dir_left_atrial + '/Training',
                                            settings.data_dir_left_atrial + '/Training_Labels', 2)
    val_dataset_left_atrial = Seg_Dataset('left_atrial', settings.data_dir_left_atrial + '/Validation',
                                          settings.data_dir_left_atrial + '/Validation_Labels', 2)
    train_dataset_hepatic_vessel = Seg_Dataset('hepatic_vessel', settings.data_dir_hepatic_vessel + '/Training',
                                               settings.data_dir_hepatic_vessel + '/Training_Labels', 2)
    val_dataset_hepatic_vessel = Seg_Dataset('spleen', settings.data_dir_hepatic_vessel + '/Validation',
                                             settings.data_dir_hepatic_vessel + '/Validation_Labels', 2)
    train_dataset_brain = Seg_Dataset('brain', settings.data_dir_brain + '/Training',
                                      settings.data_dir_brain + '/Training_Labels', 2)
    val_dataset_brain = Seg_Dataset('brain', settings.data_dir_brain + '/Validation',
                                    settings.data_dir_brain + '/Validation_Labels', 2)
    train_dataset_list = [train_dataset_brain, train_dataset_spleen, train_dataset_prostate, train_dataset_lits,
                          train_dataset_pancreas, train_dataset_left_atrial, train_dataset_hepatic_vessel]
    val_dataset_list = [val_dataset_brain, val_dataset_spleen, val_dataset_prostate, val_dataset_lits,
                        val_dataset_pancreas, val_dataset_left_atrial, val_dataset_hepatic_vessel]

    #total_dataset = Subset(train_dataset_spleen, list(range(0, batch_size)))
    # create lists of indices one for each dataset
    # indices1 = list(range(0, len(train_dataset_spleen) - 1))
    # indices2 = list(range(0, len(train_dataset_prostate) - 1))
    # indices = ([indices1, indices2])
    # train_dataset = generate_batched_dataset(dataset_list,indices,total_dataset,batch_size)

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader,val_dataloader

def train(setting_dict, exp_ind):
    global settings
    settings = SegSettings(setting_dict, write_logger=True)
    wandb.init(project="my-project")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.initial_learning_rate)

    ## initialization of all variables for plots
    train_loss_tot = [] ##cross entropy
    train_loss_tot_spleen = []
    train_loss_tot_prostate = []
    train_loss_tot_lits = []
    train_loss_tot_brain = []
    train_loss_tot_pancreas = []
    train_loss_tot_hepatic_vessel = []
    train_loss_tot_left_atrial = []

    train_organ_dice_tot = []
    train_organ_dice_spleen = []
    train_organ_dice_prostate = []
    train_organ_dice_lits = []
    train_organ_dice_brain = []
    train_organ_dice_pancreas = []
    train_organ_dice_hepatic_vessel = []
    train_organ_dice_left_atrial = []

    train_background_dice_tot = []
    train_background_dice_spleen = []
    train_background_dice_prostate = []
    train_background_dice_lits = []
    train_background_dice_brain = []
    train_background_dice_pancreas = []
    train_background_dice_hepatic_vessel = []
    train_background_dice_left_atrial = []

    train_tumour_dice_pancreas=[]

    val_loss_tot = []  ##cross entropy
    val_loss_tot_spleen = []
    val_loss_tot_prostate = []
    val_loss_tot_lits = []
    val_loss_tot_brain = []
    val_loss_tot_pancreas = []
    val_loss_tot_hepatic_vessel = []
    val_loss_tot_left_atrial = []

    val_organ_dice_tot = []
    val_organ_dice_spleen = []
    val_organ_dice_prostate = []
    val_organ_dice_lits = []
    val_organ_dice_brain = []
    val_organ_dice_pancreas = []
    val_organ_dice_hepatic_vessel = []
    val_organ_dice_left_atrial = []

    val_background_dice_tot = []
    val_background_dice_spleen = []
    val_background_dice_prostate = []
    val_background_dice_lits = []
    val_background_dice_brain = []
    val_background_dice_pancreas = []
    val_background_dice_hepatic_vessel = []
    val_background_dice_left_atrial = []

    val_tumour_dice_pancreas = []
    train_total_dice_tot=[]
    val_total_dice_tot=[]

    num_epochs = settings.num_epochs
    batch_size = settings.batch_size
    train_dataloader, val_dataloader=dataloader_8(settings,batch_size)

    print('Training... ')
    num_epochs=14
    for epoch in range(0, num_epochs):
         weight_vis(model, epoch, settings)
         epoch_start_time = time.time()
         train_loss_tot_cur = []  ##cross entropy
         train_loss_tot_spleen_cur = []
         train_loss_tot_prostate_cur = []
         train_loss_tot_lits_cur = []
         train_loss_tot_brain_cur = []
         train_loss_tot_pancreas_cur = []
         train_loss_tot_hepatic_vessel_cur = []
         train_loss_tot_left_atrial_cur = []

         train_organ_dice_tot_cur = []
         train_organ_dice_spleen_cur = []
         train_organ_dice_prostate_cur = []
         train_organ_dice_lits_cur = []
         train_organ_dice_brain_cur = []
         train_organ_dice_pancreas_cur = []
         train_organ_dice_hepatic_vessel_cur = []
         train_organ_dice_left_atrial_cur = []

         train_background_dice_tot_cur = []
         train_background_dice_spleen_cur = []
         train_background_dice_prostate_cur = []
         train_background_dice_lits_cur = []
         train_background_dice_brain_cur = []
         train_background_dice_pancreas_cur = []
         train_background_dice_hepatic_vessel_cur = []
         train_background_dice_left_atrial_cur = []
         train_tumour_dice_pancreas_cur = []

         val_loss_tot_cur = []  ##cross entropy
         val_loss_tot_spleen_cur = []
         val_loss_tot_prostate_cur = []
         val_loss_tot_lits_cur = []
         val_loss_tot_brain_cur = []
         val_loss_tot_pancreas_cur = []
         val_loss_tot_hepatic_vessel_cur = []
         val_loss_tot_left_atrial_cur = []

         val_organ_dice_tot_cur = []
         val_organ_dice_spleen_cur = []
         val_organ_dice_prostate_cur = []
         val_organ_dice_lits_cur = []
         val_organ_dice_brain_cur = []
         val_organ_dice_pancreas_cur = []
         val_organ_dice_hepatic_vessel_cur = []
         val_organ_dice_left_atrial_cur = []

         val_background_dice_tot_cur = []
         val_background_dice_spleen_cur = []
         val_background_dice_prostate_cur = []
         val_background_dice_lits_cur = []
         val_background_dice_brain_cur = []
         val_background_dice_pancreas_cur = []
         val_background_dice_hepatic_vessel_cur = []
         val_background_dice_left_atrial_cur = []
         val_tumour_dice_pancreas_cur = []

         total_steps = len(train_dataloader)
         for i,sample in enumerate(train_dataloader,1):
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
             #visualize_features(model,outputs, images, sample['task'])
             loss = criterion(outputs.double(), masks[:,0,:,:])

             # Backward and optimize
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             #print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
             logger.info('current task: ' + sample['task'][0])
             logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
             wandb.log({'epoch': epoch, 'loss': loss})
             
             one_hot = torch.DoubleTensor(masks.size(0), sample['num_classes'][0], masks.size(2), masks.size(3)).zero_()
             one_hot = one_hot.to(device)
             masks_dice = one_hot.scatter_(1, masks.data, 1)
             # outputs_dice=one_hot.scatter_(1,outputs.type(torch.LongTensor),1)
             dices = dice(outputs, masks_dice, sample['num_classes'][0], settings)
             mean_dice = dices[0]
             background_dice = dices[1]
             organ_dice = dices[2]
             if len(dices) == 4:
                 tumour_dice = dices[3]

             if sample['task'][0] == 'lits':
                 train_organ_dice_lits_cur.append(organ_dice)
                 train_background_dice_lits_cur.append(background_dice)
                 train_loss_tot_lits_cur.append(loss.item())
             if sample['task'][0] == 'hepatic_vessel':
                 train_organ_dice_hepatic_vessel_cur.append(organ_dice)
                 train_background_dice_hepatic_vessel_cur.append(background_dice)
                 train_loss_tot_hepatic_vessel_cur.append(loss.item())
             if sample['task'][0] == 'brain':
                 train_organ_dice_brain_cur.append(organ_dice)
                 train_background_dice_brain_cur.append(background_dice)
                 train_loss_tot_brain_cur.append(loss.item())
             if sample['task'][0] == 'left_atrial':
                 train_organ_dice_left_atrial_cur.append(organ_dice)
                 train_background_dice_left_atrial_cur.append(background_dice)
                 train_loss_tot_left_atrial_cur.append(loss.item())
             if sample['task'][0] == 'spleen':
                 train_organ_dice_spleen_cur.append(organ_dice)
                 train_background_dice_spleen_cur.append(background_dice)
                 train_loss_tot_spleen_cur.append(loss.item())
             if sample['task'][0] == 'prostate':
                 train_organ_dice_prostate_cur.append(organ_dice)
                 train_background_dice_prostate_cur.append(background_dice)
                 train_loss_tot_prostate_cur.append(loss.item())
             if sample['task'][0] == 'pancreas':
                 train_organ_dice_pancreas_cur.append(organ_dice)
                 train_background_dice_pancreas_cur.append(background_dice)
                 train_loss_tot_pancreas_cur.append(loss.item())
                 train_tumour_dice_pancreas_cur.append(tumour_dice)

             train_organ_dice_tot_cur.append(organ_dice)
             train_background_dice_tot_cur.append(background_dice)
             train_loss_tot_cur.append(loss.item())

             if i % 30 == 0:
                 save_out = outputs.cpu().detach().numpy()
                 save_samp(sample['image'][0], sample['mask'][0], sample['task'][0], save_out[0][1], epoch, i,
                           settings.snapshot_dir, organ_dice)

             if (i + 1) % 50 == 0:
                 if len(dices) != 4:
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

                 else:
                     print(
                         'curr train loss: {}  train organ dice: {}  train background dice: {} train tumour dice: {}\t'
                         'iter: {}/{}'.format(np.mean(train_loss_tot_cur),
                                              np.mean(train_organ_dice_tot_cur),
                                              np.mean(train_background_dice_tot_cur),
                                              np.mean(train_tumour_dice_pancreas_cur),
                                              i + 1, len(train_dataloader)))
                     logger.info(
                         'curr train loss: {}  train organ dice: {}  train background dice: {} train tumour dice: {}\t'
                         'iter: {}/{}'.format(np.mean(train_loss_tot_cur),
                                              np.mean(train_organ_dice_tot_cur),
                                              np.mean(train_background_dice_tot_cur),
                                              np.mean(train_tumour_dice_pancreas_cur),
                                              i + 1, len(train_dataloader)))
             # save_samples(model, i + 1, epoch, samples_list, settings.snapshot_dir, settings)
         train_organ_dice_lits.append(np.mean(train_organ_dice_lits_cur))
         train_background_dice_lits.append(np.mean(train_background_dice_lits_cur))
         train_loss_tot_lits.append(np.mean(train_loss_tot_lits_cur))

         train_organ_dice_hepatic_vessel.append(np.mean(train_organ_dice_hepatic_vessel_cur))
         train_background_dice_hepatic_vessel.append(np.mean(train_background_dice_hepatic_vessel_cur))
         train_loss_tot_hepatic_vessel.append(np.mean(train_loss_tot_hepatic_vessel_cur))

         train_organ_dice_brain.append(np.mean(train_organ_dice_brain_cur))
         train_background_dice_brain.append(np.mean(train_background_dice_brain_cur))
         train_loss_tot_brain.append(np.mean(train_loss_tot_brain_cur))

         train_organ_dice_left_atrial.append(np.mean(train_organ_dice_left_atrial_cur))
         train_background_dice_left_atrial.append(np.mean(train_background_dice_left_atrial_cur))
         train_loss_tot_left_atrial.append(np.mean(train_loss_tot_left_atrial_cur))

         train_organ_dice_spleen.append(np.mean(train_organ_dice_spleen_cur))
         train_background_dice_spleen.append(np.mean(train_organ_dice_spleen_cur))
         train_loss_tot_spleen.append(np.mean(train_loss_tot_spleen_cur))

         train_organ_dice_prostate.append(np.mean(train_organ_dice_prostate_cur))
         train_background_dice_prostate.append(np.mean(train_background_dice_prostate_cur))
         train_loss_tot_prostate.append(np.mean(train_loss_tot_prostate_cur))

         train_organ_dice_pancreas.append(np.mean(train_organ_dice_pancreas_cur))
         train_background_dice_pancreas.append(np.mean(train_background_dice_pancreas_cur))
         train_tumour_dice_pancreas.append(np.mean(train_tumour_dice_pancreas_cur))
         train_loss_tot_pancreas.append(np.mean(train_loss_tot_pancreas_cur))

         train_organ_dice_tot.append(np.mean(train_organ_dice_tot_cur))
         train_background_dice_tot.append(np.mean(train_background_dice_tot_cur))
         train_loss_tot.append(np.mean(train_loss_tot_cur)) #cross entropy

         #train_tumour_dice.append(np.mean(train_tumour_dice))
         total_steps=len(val_dataloader)
         save_samp_val(model, epoch, settings)
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
             logger.info('current task: ' + sample['task'][0])
             logger.info(f"Validation Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )

             one_hot = torch.DoubleTensor(masks.size(0), data['num_classes'][0], masks.size(2), masks.size(3)).zero_()
             one_hot = one_hot.to(device)
             masks_dice = one_hot.scatter_(1, masks.data, 1)
             dices = dice(outputs, masks_dice, data['num_classes'][0], settings)
             mean_dice = dices[0]
             background_dice = dices[1]
             organ_dice = dices[2]
             if len(dices) == 4:
                 tumour_dice = dices[3]
             if i%30==0:
                 save_out = outputs.cpu().detach().numpy()
                 save_samp_validation(data['image'][0], data['mask'][0], data['task'][0], save_out[0][1], epoch, i,
                                     organ_dice)
             val_organ_dice_tot_cur.append(organ_dice)
             val_background_dice_tot_cur.append(background_dice)
             val_loss_tot_cur.append(loss.item())

             if data['task'][0] == 'lits':
                 val_organ_dice_lits_cur.append(organ_dice)
                 val_background_dice_lits_cur.append(background_dice)
                 val_loss_tot_lits_cur.append(loss.item())
             if data['task'][0] == 'hepatic_vessel':
                 val_organ_dice_hepatic_vessel_cur.append(organ_dice)
                 val_background_dice_hepatic_vessel_cur.append(background_dice)
                 val_loss_tot_hepatic_vessel_cur.append(loss.item())
             if data['task'][0] == 'brain':
                 val_organ_dice_brain_cur.append(organ_dice)
                 val_background_dice_brain_cur.append(background_dice)
                 val_loss_tot_brain_cur.append(loss.item())
             if data['task'][0] == 'left_atrial':
                 val_organ_dice_left_atrial_cur.append(organ_dice)
                 val_background_dice_left_atrial_cur.append(background_dice)
                 val_loss_tot_left_atrial_cur.append(loss.item())
             if data['task'][0] == 'spleen':
                 val_organ_dice_spleen_cur.append(organ_dice)
                 val_background_dice_spleen_cur.append(background_dice)
                 val_loss_tot_spleen_cur.append(loss.item())
             if data['task'][0] == 'prostate':
                 val_organ_dice_prostate_cur.append(organ_dice)
                 val_background_dice_prostate_cur.append(background_dice)
                 val_loss_tot_prostate_cur.append(loss.item())
             if data['task'][0] == 'pancreas':
                 val_organ_dice_pancreas_cur.append(organ_dice)
                 val_background_dice_pancreas_cur.append(background_dice)
                 val_loss_tot_pancreas_cur.append(loss.item())
                 val_tumour_dice_pancreas_cur.append(tumour_dice)


         val_organ_dice_lits.append(np.mean(val_organ_dice_lits_cur))
         val_background_dice_lits.append(np.mean(val_background_dice_lits_cur))
         val_loss_tot_lits.append(np.mean(val_loss_tot_lits_cur))

         val_organ_dice_hepatic_vessel.append(np.mean(val_organ_dice_hepatic_vessel_cur))
         val_background_dice_hepatic_vessel.append(np.mean(val_background_dice_hepatic_vessel_cur))
         val_loss_tot_hepatic_vessel.append(np.mean(val_loss_tot_hepatic_vessel_cur))

         val_organ_dice_brain.append(np.mean(val_organ_dice_brain_cur))
         val_background_dice_brain.append(np.mean(val_background_dice_brain_cur))
         val_loss_tot_brain.append(np.mean(val_loss_tot_brain_cur))

         val_organ_dice_left_atrial.append(np.mean(val_organ_dice_left_atrial_cur))
         val_background_dice_left_atrial.append(np.mean(val_background_dice_left_atrial_cur))
         val_loss_tot_left_atrial.append(np.mean(val_loss_tot_left_atrial_cur))

         val_organ_dice_spleen.append(np.mean(val_organ_dice_spleen_cur))
         val_background_dice_spleen.append(np.mean(val_organ_dice_spleen_cur))
         val_loss_tot_spleen.append(np.mean(val_loss_tot_spleen_cur))

         val_organ_dice_prostate.append(np.mean(val_organ_dice_prostate_cur))
         val_background_dice_prostate.append(np.mean(val_background_dice_prostate_cur))
         val_loss_tot_prostate.append(np.mean(val_loss_tot_prostate_cur))

         val_organ_dice_pancreas.append(np.mean(val_organ_dice_pancreas_cur))
         val_background_dice_pancreas.append(np.mean(val_background_dice_pancreas_cur))
         val_tumour_dice_pancreas.append(np.mean(val_tumour_dice_pancreas_cur))
         val_loss_tot_pancreas.append(np.mean(val_loss_tot_pancreas_cur))

         val_organ_dice_tot.append(np.mean(val_organ_dice_tot_cur))
         val_background_dice_tot.append(np.mean(val_background_dice_tot_cur))
         val_loss_tot.append(np.mean(val_loss_tot_cur)) #cross entropy

         print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,(time.time() - epoch_start_time) / 60))
         print('train loss: {} val_loss: {}'.format(np.mean(train_loss_tot_cur), np.mean(val_loss_tot_cur)))
         print('train organ dice: {}  train background dice: {} val organ dice: {}  val background dice: {}'.format(
            np.mean(train_organ_dice_tot_cur), np.mean(train_background_dice_tot_cur), np.mean(val_organ_dice_tot_cur),
            np.mean(val_background_dice_tot_cur)))

         # torch.save({'unet': model.state_dict()}, os.path.join(settings.checkpoint_dir, 'unet_%08d.pt' % (epoch + 1)))
         # torch.save({'unet': optimizer.state_dict()}, os.path.join(settings.checkpoint_dir, 'optimizer.pt'))

    plot_graph(num_epochs,settings,train_organ_dice_tot,train_background_dice_tot,train_loss_tot,val_organ_dice_tot, val_background_dice_tot,val_loss_tot,
               train_organ_dice_spleen, train_background_dice_spleen, train_loss_tot_spleen,val_organ_dice_spleen,val_background_dice_spleen,val_loss_tot_spleen,
               train_organ_dice_brain, train_background_dice_brain,train_loss_tot_brain,val_organ_dice_brain, val_background_dice_brain, val_loss_tot_brain,
               train_organ_dice_prostate,  train_background_dice_prostate, train_loss_tot_prostate,val_organ_dice_prostate, val_background_dice_prostate, val_loss_tot_prostate,
               train_organ_dice_left_atrial, train_background_dice_left_atrial, train_loss_tot_left_atrial,val_organ_dice_left_atrial, val_background_dice_left_atrial,val_loss_tot_left_atrial,
               train_organ_dice_hepatic_vessel, train_background_dice_hepatic_vessel,train_loss_tot_hepatic_vessel,val_organ_dice_hepatic_vessel, val_background_dice_hepatic_vessel,val_loss_tot_hepatic_vessel,
               train_organ_dice_pancreas,  train_background_dice_pancreas, train_loss_tot_pancreas,val_organ_dice_pancreas, val_background_dice_pancreas, val_loss_tot_pancreas,train_tumour_dice_pancreas,val_tumour_dice_pancreas,
               train_organ_dice_lits, train_background_dice_lits,train_loss_tot_lits,val_organ_dice_lits, val_background_dice_lits, val_loss_tot_lits)

if __name__ == '__main__':
    start_exp_ind = 19
    num_exp = 8 ##len(os.listdir(r'experiments directory path'))
    for exp_ind in range(num_exp):
        exp_ind += start_exp_ind
        print('start with experiment: {}'.format(exp_ind))
        with open(r'G:\Deep learning\Datasets_organized\small_dataset\Experiments\exp_{}\exp_{}.json'.format(
                exp_ind, exp_ind)) as json_file:
            setting_dict = json.load(json_file)


        train(setting_dict, exp_ind=exp_ind)
