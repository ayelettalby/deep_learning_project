import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.base import Activation


class SA_diceloss(base.Loss):
    def __init__(self, activation="softmax2d", ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.device = "cuda"

    def new_diceloss(self, outputs, masks, batch_size, num_classes):
        batch_intersection = torch.sum(masks * outputs, (0, 2, 3))  # TP
        fp = torch.sum(outputs, (0, 2, 3)) - batch_intersection
        fn = torch.sum(masks, (0, 2, 3)) - batch_intersection
        beta = 1
        eps = 1e-10
        result = ((1 + beta ** 2) * batch_intersection + eps) / (
                    (1 + beta ** 2) * batch_intersection + beta ** 2 * fn + fp + eps)
        return 1 - (0.9 * result[1] + 0.1 * result[0])

    def old_diceloss(self, outputs, masks, batch_size, num_classes):
        eps = 1e-10
        # values, indices = torch.max(outputs, 1)
        # y_pred=make_one_hot(indices,batch_size,num_classes,indices.size(1),indices.size(2))
        batch_intersection = torch.sum(masks * outputs, (0, 2, 3))
        fp = torch.sum(outputs, (0, 2, 3)) - batch_intersection
        fn = torch.sum(masks, (0, 2, 3)) - batch_intersection
        batch_union = torch.sum(outputs, (0, 2, 3)) + torch.sum(masks, (0, 2, 3))
        # loss = (2 * batch_intersection + eps) / (batch_union + eps)
        loss = (2 * batch_intersection + eps) / (2 * batch_intersection + 2 * fn + fp + eps)
        bg = loss[0]
        t = loss[1]
        total_loss = (bg * 0.1 + t * 0.9)
        result = (1 - total_loss)
        return result

    def forward(self, y_pr, y_gt, batch_size, class_num):
        y_pr = self.activation(y_pr)
        return self.new_diceloss(y_pr, y_gt, batch_size, class_num)