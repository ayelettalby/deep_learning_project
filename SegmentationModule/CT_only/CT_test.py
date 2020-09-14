import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
from CT_settings import SegSettings
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from skimage.measure import label
import random
#from torchsummary import summary
from torchvision import transforms
import json
import time
import ayelet_shiri.SegmentationModule.CT_only.CT_models as models
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
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
    json_path = r'G:/Deep learning/Datasets_organized/small_dataset/Experiments/exp_1/exp_1.json'
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
        # if settings.augmentation==True:
        #     image, mask = create_augmentations(image, mask)

        sample={'image':image.astype('float64'), 'mask':mask.astype('float64'), 'task':self.task, 'num_classes':self.num_classes }
        return sample

    def __len__(self):
        return len(os.listdir(self.images_dir))

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

    def forward(self, pred, target,task):
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
                # plt.subplot(1, 3, 1)
                # plt.imshow(pred.cpu().detach().numpy()[0, 2, :, :], cmap="gray")
                #plt.show()

                if task[0]==('lits'):
                    pred_p = post_process(pred)
                    pred=torch.Tensor(pred_p).to(device)
                    pred=pred.unsqueeze(0)
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
        # plt.subplot(1, 3, 2)
        # plt.imshow(target.cpu().detach().numpy()[0, 2, :,:], cmap="gray")
        confusion_vector = pred / target.float()
        # plt.subplot(1, 3, 3)
        # plt.imshow(confusion_vector.cpu().detach().numpy()[0, 2, :, :], cmap="gray")
        # plt.show()
        # print (pred[0, 2, 150:160, 150:160])
        # print (target[0, 2, 150:160, 150:160])
        # print (confusion_vector[0, 2, 150:160, 150:160])
        TP = torch.sum(confusion_vector[0, 1, :, :] == 1).item()
        FP = torch.sum(confusion_vector[0, 1, :, :] == float('inf')).item()
        TN = torch.sum(torch.isnan(confusion_vector[0, 1, :, :])).item()
        FN = torch.sum(confusion_vector[0, 1, :, :] == 0).item()
        eps=0.000001
        sensitivity=TP/(TP+FN+eps)
        specificity=TN/(TN+FP+eps)
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
                return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),sensitivity,specificity,tumour_dice.mean().item()],pred

        if self.is_metric:
            return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),sensitivity,specificity],pred
        else:
            return -mean_dice_val

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

def save_samp(image,mask,task,prediction,epoch,iter,snapshot_dir,loss):
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    prediction = np.squeeze(prediction)

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Loss: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir,task,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
    plt.close('all')

def save_samp_test(image,mask,task,prediction,iter,loss):
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    prediction = np.squeeze(prediction)
    plt.subplot(1, 3, 3)
    plt.imshow(np.fliplr(np.rot90(prediction,3)), cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+  ' Iteration: ' + str(iter) + ' Loss: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(settings.test_dir,task,'test_it_{}.{}'.format(iter, 'png')))
    plt.close('all')

def sensitivity(pred, target, num_classes,settings):
    pred = torch.argmax(pred, dim=1).to(device)
    pred.double()
    target.double()
    TN=0
    if num_classes==2:
        TP=np.sum(target*pred)
        #TN=np.sum(target+pred==0,(0,2,3))
    return TN,TP

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

def dice(pred, target, num_classes,settings,task):
    if num_classes==2:
        mask_labels={'background': 0,  'organ': 1}
        loss_weights= {'background': 1, 'organ': 1}
    elif num_classes==3: ## pancreas only
        mask_labels = {'background': 0, 'organ': 1, 'tumour':2} ##check if this is true
        loss_weights = {'background': 1, 'organ': 10,'tumour':20}
    dice_measurement = DiceLoss(classes=num_classes,
                               dimension=settings.dimension,
                               mask_labels_numeric=mask_labels,
                               mask_class_weights_dict=loss_weights,
                               is_metric=True)
    [*dices] = dice_measurement(pred, target,task)
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

def plot_graph(nnum_epochs,settings,test_organ_dice_tot,test_background_dice_tot,test_loss_tot,
               test_organ_dice_spleen, test_background_dice_spleen, test_loss_tot_spleen,
               test_organ_dice_brain, test_background_dice_brain,test_loss_tot_brain,
               test_organ_dice_prostate,  test_background_dice_prostate, test_loss_tot_prostate,
               test_organ_dice_left_atrial, test_background_dice_left_atrial, test_loss_tot_left_atrial,
               test_organ_dice_hepatic_vessel, test_background_dice_hepatic_vessel,test_loss_tot_hepatic_vessel,
               test_organ_dice_pancreas,  test_background_dice_pancreas, test_loss_tot_pancreas,
               test_organ_dice_lits, test_background_dice_lits,test_loss_tot_lits):

    plt.figure()  # Total dice losses
    x = np.arange(0, num_epochs, 1)
    plt.plot(x, test_organ_dice_tot, 'r')
    plt.plot(x, test_background_dice_tot, 'b')
    plt.plot(x, test_loss_tot, 'c')
    plt.title('Total Test Loss and Dice')
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

def dataloader_CT(test_folder,batch_size):
    test_dataset_spleen = Seg_Dataset('spleen', test_folder + '/Spleen/Test',
                                       test_folder + '/Spleen/Test_Labels', 2)


    test_dataset_pancreas = Seg_Dataset('pancreas', test_folder  + '/Pancreas/Test',
                                         test_folder  + '/Pancreas/Test_Labels', 2)

    test_dataset_lits = Seg_Dataset('lits', test_folder + '/Lits/Test',
                                      test_folder + '/Lits/Test_Labels', 2)

    test_dataset_kits = Seg_Dataset('kits', test_folder + '/Kits/Test',
                                    test_folder + '/Kits/Test_Labels', 2)



    test_dataset_list = [test_dataset_lits, test_dataset_spleen, test_dataset_kits, test_dataset_pancreas]

    test_dataset = torch.utils.data.ConcatDataset(test_dataset_list)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=0)

    return test_dataloader

def post_process(seg):
    seg_morph =seg[:,1,:,:].clone().detach().cpu()
    organ = seg_morph>0
    kernel_morph = np.ones((1, 7, 7))
    organ=binary_erosion(organ, structure=kernel_morph, iterations=1)
    organ = binary_dilation(organ, structure=kernel_morph, iterations=1)
    organ = binary_fill_holes(organ)
    seg = np.multiply(seg.detach().cpu(), organ)

    bg=np.ones(organ.shape)
    bg=bg-organ
    bg=torch.Tensor(bg)
    organ=torch.Tensor(organ)
    final=torch.cat((bg,organ),0)
    print (final.shape)
    return final

def test(setting_dict,test_folder, exp_ind,model):
    model.to(device)
    model = model.double()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,10]).double().to(device))
    print ('model loaded')
    num_epochs = settings.num_epochs
    batch_size = settings.batch_size
    test_dataloader=dataloader_CT(test_folder,batch_size)
    print ('Data ready')

    test_loss_tot_cur = []  ##cross entropy
    test_loss_tot_spleen_cur = []
    test_loss_tot_lits_cur = []
    test_loss_tot_kits_cur = []
    test_loss_tot_pancreas_cur = []

    test_organ_dice_tot_cur = []
    test_organ_dice_spleen_cur = []
    test_organ_dice_lits_cur = []
    test_organ_dice_kits_cur = []
    test_organ_dice_pancreas_cur = []

    test_background_dice_tot_cur = []
    test_background_dice_spleen_cur = []
    test_background_dice_lits_cur = []
    test_background_dice_kits_cur = []
    test_background_dice_pancreas_cur = []

    test_mean_dice_tot_cur = []
    test_mean_dice_spleen_cur = []
    test_mean_dice_lits_cur = []
    test_mean_dice_kits_cur = []
    test_mean_dice_pancreas_cur = []

    test_sensitivity_all=[]
    test_sensitivity_spleen = []
    test_sensitivity_lits = []
    test_sensitivity_kits = []
    test_sensitivity_pancreas = []

    test_specificity_all=[]
    test_specificity_spleen = []
    test_specificity_lits = []
    test_specificity_kits = []
    test_specificity_pancreas = []

    total_steps=len(test_dataloader)
    print('Testing... ')
    for i, data in enumerate(test_dataloader):

        torch.no_grad()

        images = data['image'].double()
        masks = data['mask'].type(torch.LongTensor)
        masks = masks.unsqueeze(1)
        images = images.to(device)
        masks = masks.to(device)


        outputs = model(images, data['task'])
        outputs = outputs.to(device)

        loss = criterion(outputs.double(), masks[:,0,:,:])


        one_hot = torch.DoubleTensor(masks.size(0), data['num_classes'][0], masks.size(2), masks.size(3)).zero_()
        one_hot = one_hot.to(device)
        masks_dice = one_hot.scatter_(1, masks.data, 1)
        #activation = nn.Softmax(dim=1)
        dices,pred = dice(outputs, masks_dice, data['num_classes'][0], settings,data['task'])
        #TN,TP=sensitivity(outputs, masks_dice, data['num_classes'][0], settings)
        mean_dice = dices[0]
        background_dice = dices[1]
        organ_dice = dices[2]
        sensitivity=dices[3]
        specificity=dices[4]
        if len(dices) == 6:
            tumour_dice = dices[5]
        test_organ_dice_tot_cur.append(organ_dice)
        test_background_dice_tot_cur.append(background_dice)
        test_loss_tot_cur.append(loss.item())
        test_sensitivity_all.append(sensitivity)
        test_specificity_all.append(specificity)

        if i % 20 == 0:
            save_samp_test(data['image'][0], data['mask'][0], data['task'][0], pred, i,
                                 organ_dice)

        if data['task'][0] == 'lits':
            test_organ_dice_lits_cur.append(organ_dice)
            test_background_dice_lits_cur.append(background_dice)
            test_mean_dice_lits_cur.append(mean_dice)
            test_loss_tot_lits_cur.append(loss.item())
            test_sensitivity_lits.append(sensitivity)
            test_specificity_lits.append(specificity)
        if data['task'][0] == 'kits':
            test_organ_dice_kits_cur.append(organ_dice)
            test_background_dice_kits_cur.append(background_dice)
            test_mean_dice_kits_cur.append(mean_dice)
            test_loss_tot_kits_cur.append(loss.item())
            test_sensitivity_kits.append(sensitivity)
            test_specificity_kits.append(specificity)
        if data['task'][0] == 'spleen':
            test_organ_dice_spleen_cur.append(organ_dice)
            test_background_dice_spleen_cur.append(background_dice)
            test_mean_dice_spleen_cur.append(mean_dice)
            test_loss_tot_spleen_cur.append(loss.item())
            test_sensitivity_spleen.append(sensitivity)
            test_specificity_spleen.append(specificity)
        if data['task'][0] == 'pancreas':
            test_organ_dice_pancreas_cur.append(organ_dice)
            test_background_dice_pancreas_cur.append(background_dice)
            test_mean_dice_pancreas_cur.append(mean_dice)
            test_loss_tot_pancreas_cur.append(loss.item())
            test_sensitivity_pancreas.append(sensitivity)
            test_specificity_pancreas.append(specificity)

        print('Iteration: {}/{} Test loss: {} '.format(i, total_steps,np.mean(test_loss_tot_cur)))
        print('Test organ dice: {}  Test background dice: {}'.format(
            np.mean(test_organ_dice_tot_cur), np.mean(test_background_dice_tot_cur)))

    test_sensitivity_all = np.mean(test_sensitivity_all)
    test_specificity_all = np.mean(test_specificity_all)

    spleen_test_organ_dice=np.mean(test_organ_dice_spleen_cur)
    spleen_test_bg_dice = np.mean(test_background_dice_spleen_cur)
    spleen_test_mean_dice = np.mean(test_mean_dice_spleen_cur)
    spleen_test_loss = np.mean(test_loss_tot_spleen_cur)
    test_sensitivity_spleen = np.mean(test_sensitivity_spleen)
    test_specificity_spleen = np.mean(test_specificity_spleen)

    kits_test_organ_dice = np.mean(test_organ_dice_kits_cur)
    kits_test_bg_dice = np.mean(test_background_dice_kits_cur)
    kits_test_mean_dice = np.mean(test_mean_dice_kits_cur)
    kits_test_loss = np.mean(test_loss_tot_kits_cur)
    test_sensitivity_kits = np.mean(test_sensitivity_kits)
    test_specificity_kits = np.mean(test_specificity_kits)

    lits_test_organ_dice = np.mean(test_organ_dice_lits_cur)
    lits_test_bg_dice = np.mean(test_background_dice_lits_cur)
    lits_test_mean_dice = np.mean(test_mean_dice_lits_cur)
    lits_test_loss = np.mean(test_loss_tot_lits_cur)
    test_sensitivity_lits = np.mean(test_sensitivity_lits)
    test_specificity_lits = np.mean(test_specificity_lits)



    pancreas_test_organ_dice = np.mean(test_organ_dice_pancreas_cur)
    pancreas_test_bg_dice = np.mean(test_background_dice_pancreas_cur)
    pancreas_test_mean_dice = np.mean(test_mean_dice_pancreas_cur)
    pancreas_test_loss = np.mean(test_loss_tot_pancreas_cur)
    test_sensitivity_pancreas = np.mean(test_sensitivity_pancreas)
    test_specificity_pancreas = np.mean(test_specificity_pancreas)

    print ('kits organ dice: ' ,kits_test_organ_dice, ', kits background dice: ', kits_test_bg_dice)
    print ('kits sensitivity: ', test_sensitivity_kits, ', kits specificity: ', test_specificity_kits)
    print ('kits mean dice: ', kits_test_mean_dice)
    print('spleen organ dice: ', spleen_test_organ_dice, ', spleen background dice: ', spleen_test_bg_dice)
    print('spleen sensitivity: ', test_sensitivity_spleen, ', spleen specificity: ', test_specificity_spleen)
    print('spleen mean dice: ', spleen_test_mean_dice)
    print('lits organ dice: ', lits_test_organ_dice, ', lits background dice: ', lits_test_bg_dice)
    print('lits sensitivity: ', test_sensitivity_lits, ', lits specificity: ', test_specificity_lits)
    print('lits mean dice: ', lits_test_mean_dice)


    print('pancreas organ dice: ', pancreas_test_organ_dice, ', pancreas background dice: ', pancreas_test_bg_dice)
    print('pancreas sensitivity: ', test_sensitivity_pancreas, ', pancreas specificity: ', test_specificity_pancreas)
    print('pancreas mean dice: ', pancreas_test_mean_dice)

    with open(r'G:\Deep learning\Datasets_organized\Prepared_Data\CT_experiments\exp_{}\test\test_results_exp_{}_epoch_{}.txt'.format(exp_ind,exp_ind,epoch),'w') as f:
        f.writelines('kits organ dice: ' +str(kits_test_organ_dice)+ ', kits background dice: '+ str(kits_test_bg_dice)+
                     '\nkits sensitivity: '+str(test_sensitivity_kits)+ ', kits specificity: '+
                           str(test_specificity_kits)+
        '\nspleen organ dice: '+str(spleen_test_organ_dice)+ ', spleen background dice: '+str(spleen_test_bg_dice)+
        '\nspleen sensitivity: '+str(test_sensitivity_spleen)+ ', spleen specificity: ' +str(test_specificity_spleen)+
        '\nlits organ dice: ' + str(lits_test_organ_dice) + ', lits background dice: ' + str(lits_test_bg_dice) +
        '\nlits sensitivity: ' + str(test_sensitivity_lits) + ', lits specificity: ' + str(test_specificity_lits) +
        '\npancreas organ dice: ' + str(pancreas_test_organ_dice) + ', pancreas background dice: ' + str(pancreas_test_bg_dice) +
        '\npancreas sensitivity: ' + str(test_sensitivity_pancreas) + ', pancreas specificity: ' + str(test_specificity_pancreas) )


        # plot_graph(num_epochs,settings,test_organ_dice_tot,test_background_dice_tot,test_loss_tot,
    #            test_organ_dice_spleen, test_background_dice_spleen, test_loss_tot_spleen,
    #            test_organ_dice_brain, test_background_dice_brain,test_loss_tot_brain,
    #            test_organ_dice_prostate,  test_background_dice_prostate, test_loss_tot_prostate,
    #            test_organ_dice_left_atrial, test_background_dice_left_atrial, test_loss_tot_left_atrial,
    #            test_organ_dice_hepatic_vessel, test_background_dice_hepatic_vessel,test_loss_tot_hepatic_vessel,
    #            test_organ_dice_pancreas,  test_background_dice_pancreas, test_loss_tot_pancreas,
    #            test_organ_dice_lits, test_background_dice_lits,test_loss_tot_lits)

if __name__ == '__main__':
    exp_ind=3
    epoch=9
    with open(r'G:\Deep learning\Datasets_organized\Prepared_Data\CT_experiments\exp_{}\exp_{}.json'.format(exp_ind,exp_ind)) as json_file:
        setting_dict = json.load(json_file)
        global settings
        settings = SegSettings(setting_dict, write_logger=True)
    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)
    model.load_state_dict(torch.load('G:\Deep learning\Datasets_organized\Prepared_Data\CT_experiments\exp_3\checkpoint\exp_{}_epoch_{}.pt'.format(exp_ind,exp_ind,epoch)))
    #model.eval()
    test_folder=r'G:\Deep learning\Datasets_organized\Prepared_Data'

    test(setting_dict,test_folder, exp_ind,model)
