import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
from SegmentationSettings import SegSettings
import nibabel as nb
import segmentation_models_pytorch as smp
import random
from PIL import Image
# from torchsummary import summary
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

user = 'remote'
if user == 'ayelet':
    json_path = r'C:\Users\Ayelet\Desktop\school\fourth_year\deep_learning_project\ayelet_shiri\sample_Data\exp_1\exp_1.json'
elif user == 'remote':
    json_path = r'G:/Deep learning/Datasets_organized/Prepared_Data/exp_1/exp_1.json'
elif user == 'shiri':
    json_path = r'F:/Prepared Data/exp_1/exp_1.json'

with open(json_path) as f:
    setting_dict = json.load(f)
settings = SegSettings(setting_dict, write_logger=True)


class Seg_Dataset(BaseDataset):
    def __init__(self, task, images_dir, masks_dir, num_classes, transforms=None):
        self.task = task
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        # self.device="cuda"
        self.num_classes = num_classes

    def __getitem__(self, idx):
        images = os.listdir(self.images_dir)
        image = np.load(self.images_dir + '/' + images[idx])

        masks = os.listdir(self.masks_dir)
        mask = np.load(self.masks_dir + '/' + masks[idx])
        global settings
        if settings.pre_process == True:
            image = pre_processing(image, self.task, settings)
        new_image, new_mask = create_augmentations(image, mask)

        # if self.transforms:
        #     image = self.transforms(image)
        #     mask = self.transforms(mask)

        sample = {'image': new_image, 'mask': new_mask, 'task': self.task, 'num_classes': self.num_classes}
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
            if self.classes > 1:
                pred = torch.argmax(pred, dim=1)
                pred = torch.eye(self.classes)[pred]
                pred = pred.transpose(1, 3)
                # pred = pred.transpose(1, 3).cuda(1)
            else:
                pred_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                pred_copy[:, 1, :, :][pred[:, 0, :, :] > 0.5] = 1
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
        tumour_dice = None
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
                                        organ_dice * self.mask_class_weights_dict['organ'] + tumour_dice *
                                        self.mask_class_weights_dict['tumour']) * 1 / self.tot_weight,
                                       dim=0)
            if self.is_metric:
                return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),
                        tumour_dice.mean().item()]

        if self.is_metric:
            return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item()]
        else:
            return -mean_dice_val


def create_augmentations(image, mask):
    p = random.choice([0, 1, 2])
    if p == 1:
        new_image = np.zeros(image.shape)
        new_mask = np.zeros(mask.shape)
        k = random.choice([0, 1, 2, 3])
        if k == 0:
            augmentation = np.rot90
        if k == 1:
            augmentation = np.fliplr
        if k == 2:
            augmentation = np.flipud
        if k == 3:
            augmentation = np.rot90
        if k == 3:
            new_mask = augmentation(mask, axes=(1, 0))
            for i in range(image.shape[0]):
                new_image[i, :, :] = augmentation(image[i, :, :], axes=(1, 0))

        else:
            new_mask = augmentation(mask)
            for i in range(image.shape[0]):
                new_image[i, :, :] = augmentation(image[i, :, :])

        return (new_image.copy(), new_mask.copy())
    else:
        return (image, mask)


def pre_processing(input_image, task, settings):
    if task == ('spleen' or 'lits' or 'pancreas' or 'hepatic vessel'):  # CT, clipping, Z_Score, normalization btw0-1
        clipped_image = clip(input_image, settings)
        c_n_image = zscore_normalize(clipped_image)
        min_val = np.amin(c_n_image)
        max_val = np.amax(c_n_image)
        final = (c_n_image - min_val) / (max_val - min_val)
        final[final > 1] = 1
        final[final < 0] = 0

    else:  # MRI, Z_score, normalization btw 0-1
        norm_image = zscore_normalize(input_image)
        min_val = np.amin(norm_image)
        max_val = np.amax(norm_image)
        final = (norm_image - min_val) / (max_val - min_val)
        final[final > 1] = 1
        final[final < 0] = 0
        final = norm_image

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
    mean = img.mean()
    std = img.std()
    normalized = (img - mean) / std
    return normalized


def save_samples(model, iter, epoch, samples_list, snapshot_dir, settings):
    samples_imgs = samples_list
    # samples_imgs = ['ct_122_268_0.4234.npy', 'ct_122_350_0.5529.npy', 'ct_122_365_0.5766.npy', 'ct_122_383_0.6051.npy']
    fig = plt.figure()
    k = 0
    organ_to_seg = settings.organ_to_seg
    if organ_to_seg == 'liver':
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

        pred = pred.cpu().detach().numpy()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        plt.subplot(len(samples_imgs), 3, 3 * k + 1)
        plt.imshow(image.cpu().numpy()[0, 1, :, :], cmap='gray')
        plt.subplot(len(samples_imgs), 3, 3 * k + 2)
        plt.imshow(pred[0, 0, :, :], cmap='gray')
        plt.title('pred iter: {} epoch: {} {} dice: {}'.format(iter, epoch, organ_to_seg, liver_dice))
        plt.subplot(len(samples_imgs), 3, 3 * k + 3)
        plt.imshow(mask.cpu().detach().numpy()[0, 0, :, :], cmap='gray')

        k += 1
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir, 'pred_{}_{}.{}'.format(iter, epoch, 'png')))


def dice(pred, target, num_classes, settings):
    if num_classes == 2:
        mask_labels = {'background': 0, 'organ': 1}
        loss_weights = {'background': 1, 'organ': 10}
    elif num_classes == 3:  ## pancreas only
        mask_labels = {'background': 0, 'organ': 1, 'tumour': 2}  ##check if this is true
        loss_weights = {'background': 1, 'organ': 10, 'tumour': 20}
    dice_measurement = DiceLoss(classes=num_classes,
                                dimension=settings.dimension,
                                mask_labels_numeric=mask_labels,
                                mask_class_weights_dict=loss_weights,
                                is_metric=True)
    # mean_dice, background_dice, organ_dice = dice_measurement(pred, target)
    [*dices] = dice_measurement(pred, target)
    return dices


def make_one_hot(labels, batch_size, num_classes, image_shape_0, image_shape_1):
    one_hot = torch.zeros([batch_size, num_classes, image_shape_0, image_shape_1], dtype=torch.float64)
    # one_hot = one_hot.to("cuda")
    labels = labels.unsqueeze(1)
    result = one_hot.scatter_(1, labels.data, 1)
    return result


def generate_batched_dataset(dataset_list, indices, total_dataset, batch_size):
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


def visualize_features(model, outputs, image, task):
    modules = list(model.children())
    mod = nn.Sequential(*modules)[0]
    print(mod)
    for p in mod.parameters():
        p.requires_grad = False

    mod = mod.double()
    out = mod(image)
    feature = out[0].numpy()
    print(feature.shape)
    plt.imshow(feature[0, 0, :, :], cmap="gray")
    plt.show()


def weight_vis(model):
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights = m.weight.data
            break

    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(weights[i, 0, :, :], cmap="gray")

    plt.show()

    # show only the first layer


def train(setting_dict, exp_ind):
    settings = SegSettings(setting_dict, write_logger=True)
    logging.basicConfig(
        filename=settings.simulation_folder + '\logger',
        filemode='a',
        format='%(asctime)s, %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)

    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)

    # model.cuda()
    model = model.double()
    # summary(model, tuple(settings.input_size))

    criterion = smp.utils.losses.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.initial_learning_rate)

    train_loss_tot = []
    train_organ_dice_tot = []
    train_background_dice_tot = []
    val_loss_tot = []
    val_organ_dice_tot = []
    val_background_dice_tot = []

    num_epochs = settings.num_epochs
    batch_size = settings.batch_size

    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(),
    # ])

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

    dataset_list = [train_dataset_spleen, train_dataset_prostate, train_dataset_pancreas]
    # generate mixed dataset
    batch_size = 2

    total_dataset = Subset(train_dataset_spleen, list(range(0, batch_size)))

    # create lists of indices one for each dataset
    indices1 = list(range(0, len(train_dataset_spleen) - 1))
    indices2 = list(range(0, len(train_dataset_prostate) - 1))
    indices3 = list(range(0, len(train_dataset_pancreas) - 1))
    indices = ([indices1, indices2, indices3])

    train_dataset = generate_batched_dataset(dataset_list, indices, total_dataset, batch_size)
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_spleen, val_dataset_prostate, val_dataset_pancreas])

    sampler = SequentialSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                                   shuffle=False, num_workers=0, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    print('Training... ')

    for epoch in range(0, num_epochs):
        epoch_start_time = time.time()
        train_loss = []
        train_organ_dice = []
        train_background_dice = []
        train_tumour_dice = []
        val_loss = []
        val_background_dice = []
        val_organ_dice = []
        total_steps = len(train_dataloader)
        for i, sample in enumerate(train_dataloader, 1):
            # weight_vis(model)
            print(sample['task'])
            images = sample['image'].double()

            masks = sample['mask'].type(torch.LongTensor)
            masks = masks.unsqueeze(1)
            # masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
            # masks = masks.to("cuda")

            one_hot = torch.DoubleTensor(masks.size(0), sample['num_classes'][0], masks.size(2), masks.size(3)).zero_()
            # one_hot = one_hot.to("cuda")
            masks = one_hot.scatter_(1, masks.data, 1)
            masks = masks.double()

            # Forward pass
            if sample['task'][0] == sample['task'][
                batch_size - 1]:  # make sure all samples of the batch are  of the same task
                outputs = model(images, sample['task'])
                # visualize_features(model,outputs, images, sample['task'])
                if masks.shape[0] == batch_size:
                    loss = criterion(outputs.double(), masks)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
                    logging.info('current task: ' + sample['task'][0])
                    logging.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )

            dices = dice(outputs, masks, sample['num_classes'][0], settings)
            mean_dice = dices[0]
            background_dice = dices[1]
            organ_dice = dices[2]
            if len(dices) == 4:
                tumour_dice = dices[3]
                train_tumour_dice.append(tumour_dice)

            train_organ_dice.append(organ_dice)
            train_background_dice.append(background_dice)
            train_loss.append(loss.item())

            if (i + 1) % 2 == 0:
                if len(dices) != 4:
                    print('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                          'iter: {}/{}'.format(np.mean(train_loss),
                                               np.mean(train_organ_dice),
                                               np.mean(train_background_dice),
                                               i + 1, len(train_dataloader)))
                    logging.info('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                                 'iter: {}/{}'.format(np.mean(train_loss),
                                                      np.mean(train_organ_dice),
                                                      np.mean(train_background_dice),
                                                      i + 1, len(train_dataloader)))

                else:
                    print('curr train loss: {}  train organ dice: {}  train background dice: {} train tumour dice: {}\t'
                          'iter: {}/{}'.format(np.mean(train_loss),
                                               np.mean(train_organ_dice),
                                               np.mean(train_background_dice),
                                               np.mean(train_tumour_dice),
                                               i + 1, len(train_dataloader)))
                    logging.info(
                        'curr train loss: {}  train organ dice: {}  train background dice: {} train tumour dice: {}\t'
                        'iter: {}/{}'.format(np.mean(train_loss),
                                             np.mean(train_organ_dice),
                                             np.mean(train_background_dice),
                                             np.mean(train_tumour_dice),
                                             i + 1, len(train_dataloader)))
            # save_samples(model, i + 1, epoch, samples_list, settings.snapshot_dir, settings)

        train_loss_tot.append(np.mean(train_loss))
        train_background_dice_tot.append(np.mean(train_background_dice))
        train_organ_dice_tot.append(np.mean(train_organ_dice))

        for i, data in enumerate(val_loader):
            model.eval()
            images, masks = data['image'].cuda(), data['mask'].cuda()
            masks = masks.view((masks.size(0), 1, masks.size(1), masks.size(2)))
            outputs = model(images)
            loss = criterion(outputs, masks)

            mean_dice, background_dice, organ_dice = dice(outputs, masks, settings)
            val_loss.append(loss.item())
            val_background_dice.append(background_dice)
            val_organ_dice.append(organ_dice)

        val_loss_tot.append(np.mean(val_loss))
        val_background_dice_tot.append(np.mean(val_background_dice))
        val_organ_dice_tot.append(np.mean(val_organ_dice))

        print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,
                                                                  (time.time() - epoch_start_time) / 60))
        logging.info('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,
                                                                         (time.time() - epoch_start_time) / 60))
        print('train loss: {} val_loss: {}'.format(np.mean(train_loss), np.mean(val_loss)))
        logging.info('train loss: {} val_loss: {}'.format(np.mean(train_loss), np.mean(val_loss)))
        print('train liver dice: {}  train background dice: {} val liver dice: {}  val background dice: {}'.format(
            np.mean(train_organ_dice), np.mean(train_background_dice), np.mean(val_organ_dice),
            np.mean(val_background_dice)
        ))
        logging.info(
            'train liver dice: {}  train background dice: {} val liver dice: {}  val background dice: {}'.format(
                np.mean(train_organ_dice), np.mean(train_background_dice), np.mean(val_organ_dice),
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
    matplotlib.pyplot.plot(x, train_organ_dice_tot, 'r')
    matplotlib.pyplot.plot(x, val_organ_dice_tot, 'b')
    matplotlib.pyplot.title('Training & Validation organ Dice vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation organ Dice vs num of epochs.png'))
    matplotlib.pyplot.plot(x, train_background_dice_tot, 'r')
    matplotlib.pyplot.plot(x, val_background_dice_tot, 'b')
    matplotlib.pyplot.title('Training & Validation  background Dice vs num of epochs')
    matplotlib.pyplot.show()
    plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation  background Dice vs num of epochs.png'))


if __name__ == '__main__':
    if user == 'ayelet':
        path = r'C:\Users\Ayelet\Desktop\school\fourth_year\deep_learning_project\ayelet_shiri\sample_Data'
    elif user == 'remote':
        path = r'G:/Deep learning/Datasets_organized/Prepared_Data'
    elif user == 'shiri':
        path = r'F:/Prepared Data'
    start_exp_ind = 1
    num_exp = 8  ##len(os.listdir(r'experiments directory path'))
    for exp_ind in range(num_exp):
        exp_ind += start_exp_ind
        print('start with experiment: {}'.format(exp_ind))
        with open(path + '\exp_{}\exp_{}.json'.format(
                exp_ind, exp_ind)) as json_file:
            setting_dict = json.load(json_file)

        train(setting_dict, exp_ind=exp_ind)


