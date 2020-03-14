import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
from SegmentationModule.SegmentationSettings import SegSettings

from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
import json
import time
import SegmentationModule.Models as models
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

x_train_dir={'lits':'','prostate':'','brain':''} ##dictionary containing all dataset images and their location, i.e. {live: 'c:/documents....}
y_train_dir={'lits':'','prostate':'','brain':''}##dictionary containing all dataset labels and their location, i.e. {live: 'c:/documents....}
x_val_dir={'lits':'','prostate':'','brain':''}
y_val_dir={'lits':'','prostate':'','brain':''}
settings = SegSettings(setting_dict, write_logger=True)

class Seg_Dataset(BaseDataset):
    def __init__(self, task, images_dir,masks_dir, num_classes: int, transforms=None):
        self.task=task
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.transforms = transforms
        #self.device="cuda"

    def __getitem__(self, idx):
        images = os.listdir(self.images_dir)
        image = np.load(self.images_dir + '/' + images[idx])

        if self.transforms:
            image = self.transforms(image)

        masks = os.listdir(self.masks_dir)
        mask = np.load(self.masks_dir + '/' + masks[idx])
        if self.transforms:
            mask = self.transforms(mask)
        sample={'image':image, 'mask':mask, 'task':self.task }
        return sample

    def __len__(self):
        return len(os.listdir(self.images_dir))

matplotlib.use('TkAgg')

cudnn.benchmark = True

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
        if self.is_metric:
            if self.classes >1:
                pred = torch.argmax(pred, dim=1)
                pred = torch.eye(self.classes)[pred]
                pred = pred.transpose(1, 3).cuda(1)
            else:
                pred_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                pred_copy[:, 1, :, :][pred[:, 0, :, :]  > 0.5] = 1
                pred_copy[:, 0, :, :][pred[:, 0, :, :] <= 0.5] = 1
                target_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                target_copy[:, 1, :, :][target[:, 0, :, :] == 1.0] = 1
                target_copy[:, 0, :, :][target[:, 0, :, :] == 0.0] = 1

                pred = pred_copy
                target = target_copy
        batch_intersection = torch.sum(pred * target.float(), dim=tuple(list(range(2, self.dimension + 2))))
        batch_union = torch.sum(pred, dim=tuple(list(range(2, self.dimension + 2)))) + torch.sum(target.float(),
                                                                                                 dim=tuple(
                                                                                                     list(range(2,
                                                                                                                self.dimension + 2))))
        background_dice = (2 * batch_intersection[:, self.mask_labels_numeric['background']] + self.eps) / (
                batch_union[:, self.mask_labels_numeric['background']] + self.eps)
        liver_dice = (2 * batch_intersection[:, self.mask_labels_numeric['liver']] + self.eps) / (
                batch_union[:, self.mask_labels_numeric['liver']] + self.eps)

        mean_dice_val = torch.mean((background_dice * self.mask_class_weights_dict['background'] +
                                    liver_dice * self.mask_class_weights_dict['liver']) * 1 / self.tot_weight, dim=0
                                   )

        if self.is_metric:
            return mean_dice_val.mean().item(), background_dice.mean().item(), liver_dice.mean().item()
        else:
            return -mean_dice_val

def clip_n_normalize(data, settings):

    # clip and normalize
    min_val = settings.min_clip_val
    max_val = settings.max_clip_val
    data = (data - min_val) / (max_val - min_val)
    data[data > 1] = 1
    data[data < 0] = 0

    return data

def save_samples(model, iter, epoch, samples_list, snapshot_dir, settings):
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
        image = tensor_transform(image).cuda()
        image = image.unsqueeze(0)
        mask = np.load(seg_path).astype('uint8')
        mask[mask == 2] = 1
        mask = np.eye(2)[mask]
        mask = tensor_transform(mask)
        mask = torch.argmax(mask, dim=0).cuda()
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

def dice(pred, target, settings):
    dice_measurment = DiceLoss(classes=settings.classes,
                               dimension=settings.dimension,
                               mask_labels_numeric=settings.mask_labels_numeric,
                               mask_class_weights_dict=settings.loss_weights,
                               is_metric=True)
    mean_dice, background_dice, liver_dice = dice_measurment(pred, target)
    return mean_dice, background_dice, liver_dice


def train_liver_segmentation(settings, exp_ind):
    train_p = dataloader.LiTSDatasetTrain(settings, partition_set='train', transforms=None)
    val_p = dataloader.LiTSDatasetTrain(settings, partition_set='validation', transforms=None)

    train_loader_p = DataLoader(train_p, batch_size=settings.batch_size, shuffle=True)
    val_loader_p = DataLoader(val_p, batch_size=1, shuffle=False)

    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)

    model.cuda()
    summary(model, tuple(settings.input_size))

    criterion_vanilla = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.initial_learning_rate)

    train_loss_tot = []
    train_liver_dice_tot = []
    train_background_dice_tot = []
    val_loss_tot = []
    val_background_dice_tot = []
    val_liver_dice_tot = []
    num_epochs = settings.num_epochs

    print('starts training liver')

    samples_list = ['ct_122_268_0.4234.npy',
                    'ct_122_350_0.5529.npy',
                    'ct_122_365_0.5766.npy',
                    'ct_122_383_0.6051.npy']

    for epoch in range(0, num_epochs):
        epoch_start_time = time.time()
        train_loss = []
        train_liver_dice = []
        train_background_dice = []
        val_loss = []
        val_background_dice = []
        val_liver_dice = []

        for i, data in enumerate(train_loader_p):
            x_data, y_data = data['image'].cuda(), data['mask'].cuda()
            y_data = y_data.view((y_data.size(0), 1, y_data.size(1), y_data.size(2)))
            optimizer.zero_grad()
            pred = model(x_data)
            loss = criterion_vanilla(pred, y_data.float())
            loss.backward()
            optimizer.step()

            mean_dice, background_dice, liver_dice = dice(pred, y_data, settings)

            train_liver_dice.append(liver_dice)
            train_background_dice.append(background_dice)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print('curr train loss: {}  train liver dice: {}  train background dice: {} \t'
                      'iter: {}/{}'.format(np.mean(train_loss),
                                           np.mean(train_liver_dice),
                                           np.mean(train_background_dice),
                                           i + 1, len(train_loader_p)))
                save_samples(model, i + 1, epoch, samples_list, settings.snapshot_dir, settings)

        train_loss_tot.append(np.mean(train_loss))
        train_background_dice_tot.append(np.mean(train_background_dice))
        train_liver_dice_tot.append(np.mean(train_liver_dice))

        for i, data in enumerate(val_loader_p):
            model.eval()
            x_data, y_data = data['image'].cuda(), data['mask'].cuda()
            y_data = y_data.view((y_data.size(0), 1, y_data.size(1), y_data.size(2)))
            pred = model(x_data)
            loss = criterion_vanilla(pred, y_data)

            mean_dice, background_dice, liver_dice = dice(pred, y_data, settings)
            val_loss.append(loss.item())
            val_background_dice.append(background_dice)
            val_liver_dice.append(liver_dice)

        val_loss_tot.append(np.mean(val_loss))
        val_background_dice_tot.append(np.mean(val_background_dice))
        val_liver_dice_tot.append(np.mean(val_liver_dice))

        print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,
                                                                  (time.time() - epoch_start_time) / 60))
        print('train loss: {} val_loss: {}'.format(np.mean(train_loss), np.mean(val_loss)))
        print('train liver dice: {}  train background dice: {} val liver dice: {}  val background dice: {}'.format(
            np.mean(train_liver_dice), np.mean(train_background_dice), np.mean(val_liver_dice),
            np.mean(val_background_dice)
        ))

        torch.save({'unet': model.state_dict()}, os.path.join(settings.checkpoint_dir, 'unet_%08d.pt' % (epoch + 1)))
        torch.save({'unet': optimizer.state_dict()}, os.path.join(settings.checkpoint_dir, 'optimizer.pt'))

    x = np.arange(0, num_epochs, 1)
    matplotlib.pyplot.plot(x, train_loss_tot, 'r')
    matplotlib.pyplot.plot(x, val_loss_tot, 'b')
    matplotlib.pyplot.title('Training & Validation loss vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation loss vs num of epochs.png'))
    matplotlib.pyplot.plot(x, train_liver_dice_tot, 'r')
    matplotlib.pyplot.plot(x, val_liver_dice_tot, 'b')
    matplotlib.pyplot.title('Training & Validation liver Dice vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation liver Dice vs num of epochs.png'))
    matplotlib.pyplot.plot(x, train_background_dice_tot, 'r')
    matplotlib.pyplot.plot(x, val_background_dice_tot, 'b')
    matplotlib.pyplot.title('Training & Validation  background Dice vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation  background Dice vs num of epochs.png'))

def train_kidney_segmentation(settings, exp_ind):
    train_k = dataloader.KiTSDatasetTrain(settings, partition_set='train', transforms=None)
    val_k = dataloader.KiTSDatasetTrain(settings, partition_set='validation', transforms=None)

    train_loader_k = DataLoader(train_k, batch_size=settings.batch_size, shuffle=True)
    val_loader_k = DataLoader(val_k, batch_size=1, shuffle=False)

    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)

    model.cuda()
    summary(model, tuple(settings.input_size))

    criterion_vanilla = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.initial_learning_rate)

    train_loss_tot = []
    train_kidney_dice_tot = []
    train_background_dice_tot = []
    val_loss_tot = []
    val_background_dice_tot = []
    val_kidney_dice_tot = []
    num_epochs = settings.num_epochs

    print('starts training kidney')

    samples_list = ['ct_00045_113_0.6075.npy',
                    'ct_00045_104_0.5591.npy',
                    'ct_00045_74_0.3978.npy',
                    'ct_00045_97_0.5215.npy']

    for epoch in range(0, num_epochs):
        epoch_start_time = time.time()
        train_loss = []
        train_kidney_dice = []
        train_background_dice = []
        val_loss = []
        val_background_dice = []
        val_kidney_dice = []

        for i, data in enumerate(train_loader_k):
            x_data, y_data = data['image'].cuda(), data['mask'].cuda()
            y_data = y_data.view((y_data.size(0), 1, y_data.size(1), y_data.size(2)))
            optimizer.zero_grad()
            pred = model(x_data)
            loss = criterion_vanilla(pred, y_data.float())
            loss.backward()
            optimizer.step()

            mean_dice, background_dice, kidney_dice = dice(pred, y_data, settings)

            train_kidney_dice.append(kidney_dice)
            train_background_dice.append(background_dice)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print('curr train loss: {}  train kidney dice: {}  train background dice: {} \t'
                      'iter: {}/{}'.format(np.mean(train_loss),
                                           np.mean(train_kidney_dice),
                                           np.mean(train_background_dice),
                                           i + 1, len(train_loader_k)))
                save_samples(model, i + 1, epoch, samples_list, settings.snapshot_dir, settings)

        train_loss_tot.append(np.mean(train_loss))
        train_background_dice_tot.append(np.mean(train_background_dice))
        train_kidney_dice_tot.append(np.mean(train_kidney_dice))

        for i, data in enumerate(val_loader_k):
            model.eval()
            x_data, y_data = data['image'].cuda(), data['mask'].cuda()
            y_data = y_data.view((y_data.size(0), 1, y_data.size(1), y_data.size(2)))
            pred = model(x_data)
            loss = criterion_vanilla(pred, y_data)

            mean_dice, background_dice, kidney_dice = dice(pred, y_data, settings)
            val_loss.append(loss.item())
            val_background_dice.append(background_dice)
            val_kidney_dice.append(kidney_dice)

        val_loss_tot.append(np.mean(val_loss))
        val_background_dice_tot.append(np.mean(val_background_dice))
        val_kidney_dice_tot.append(np.mean(val_kidney_dice))

        print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,
                                                                  (time.time() - epoch_start_time) / 60))
        print('train loss: {} val_loss: {}'.format(np.mean(train_loss), np.mean(val_loss)))
        print('train liver dice: {}  train background dice: {} val kidney dice: {}  val background dice: {}'.format(
            np.mean(train_kidney_dice), np.mean(train_background_dice), np.mean(val_kidney_dice),
            np.mean(val_background_dice)
        ))

        torch.save({'unet': model.state_dict()}, os.path.join(settings.checkpoint_dir, 'unet_%08d.pt' % (epoch + 1)))
        torch.save({'unet': optimizer.state_dict()}, os.path.join(settings.checkpoint_dir, 'optimizer.pt'))

    x = np.arange(0, num_epochs, 1)
    matplotlib.pyplot.plot(x, train_loss_tot, 'r')
    matplotlib.pyplot.plot(x, val_loss_tot, 'b')
    matplotlib.pyplot.title('Training & Validation loss vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation loss vs num of epochs.png'))
    matplotlib.pyplot.plot(x, train_kidney_dice_tot, 'r')
    matplotlib.pyplot.plot(x, val_kidney_dice_tot, 'b')
    matplotlib.pyplot.title('Training & Validation kidney Dice vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation liver Dice vs num of epochs.png'))
    matplotlib.pyplot.plot(x, train_background_dice_tot, 'r')
    matplotlib.pyplot.plot(x, val_background_dice_tot, 'b')
    matplotlib.pyplot.title('Training & Validation  background Dice vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation  background Dice vs num of epochs.png'))

def train(setting_dict, exp_ind):
    settings = SegSettings(setting_dict, write_logger=True)
    train_dataset_lits = Seg_Dataset('lits',settings.data_dir_lits + '/Training' , settings.data_dir_lits + '/Training_Labels', 2)
    val_dataset_lits = Seg_Dataset('lits',settings.data_dir_lits + '/Validation', settings.data_dir_lits + '/Validation_Labels', 2)
    train_dataset_prostate = Seg_Dataset('prostate',settings.data_dir_prostate + '/Training' , settings.data_dir_prostate + '/Training_Labels', 2)
    val_dataset_prostate =  Seg_Dataset('prostate',settings.data_dir_prostate + '/Validation' , settings.data_dir_prostate + '/Validation_Labels', 2)
    train_dataset_brain = Seg_Dataset('brain',settings.data_dir_brain + '/Training' , settings.data_dir_prostate + '/Training_Labels', 2)
    val_dataset_brain = Seg_Dataset('brain',settings.data_dir_brain + '/Validation' , settings.data_dir_prostate + '/Validation_Labels', 2)
    train_dataset=torch.utils.data.ConcatDataset([train_dataset_lits, train_dataset_prostate, train_dataset_brain])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_lits, val_dataset_prostate, val_dataset_brain])
    batchsize = 4
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    if settings.organ_to_seg=='liver':
        train_liver_segmentation(settings, exp_ind)
    elif settings.organ_to_seg=='kidney':
        train_kidney_segmentation(settings, exp_ind)



if __name__ == '__main__':
    start_exp_ind = 7

    num_exp = len(os.listdir(r'experiments directory path'))
    for exp_ind in range(num_exp):
        exp_ind += start_exp_ind
        print('start with experiment: {}'.format(exp_ind))
        with open(r'experiments directory path\exp_{}\exp_{}.json'.format(
                exp_ind, exp_ind)) as json_file:
            setting_dict = json.load(json_file)

        train(setting_dict, exp_ind=exp_ind)
