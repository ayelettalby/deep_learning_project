import numpy as np
import torch

def make_one_hot(labels,batch_size,num_classes,image_shape_0, image_shape_1):
    one_hot=torch.zeros([batch_size,num_classes,image_shape_0,image_shape_1],dtype=torch.float64)
    labels=labels.unsqueeze(1)
    result = one_hot.scatter_(1,labels.data,1)
    return result

def diceloss(masks,outputs,batch_size,num_classes):
    eps = 1e-10
    values, indices = torch.max(outputs, 1)
    y_pred=make_one_hot(indices,batch_size,num_classes,indices.size(1),indices.size(2))
    batch_intersection=torch.sum(masks*y_pred,(0,2,3))
    batch_union=torch.sum(y_pred,(0,2,3))+torch.sum(masks,(0,2,3))
    loss=(2*batch_intersection+eps)/(batch_union+eps)
    bg=loss[0].item()
    t=loss[1].item()
    total_loss=(bg*0.2+t*0.8)
    return (1-bg),(1-t),(1-total_loss)
