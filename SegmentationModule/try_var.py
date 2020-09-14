import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
from SegmentationSettings import SegSettings
import segmentation_models_pytorch as smp
import random

from torch.utils.data import DataLoader
from torchsummary import summary
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
class MySampler(RandomSampler):
    def __init__(self,data_source,batch_size):
        super().__init__(data_source, replacement=False, num_samples=None)
        self.n=-1
        self.data_source=data_source
        self.batch_size=batch_size
        self.a=''

    def __iter__(self):
        self.n+=1
        print (self.n)
        n = len(self.data_source)
        # if self.replacement:
        #     return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    # def __next__(self):
    #     # if self.n % self.batch_size == 0:
    #
    #
    #     # self.n+=1
    #     # if self.n%self.batch_size==0:
    #     #     a=iter(range(len(self.data_source)))
    #     # else:
    #     #     a=iter(range(self.n))
    #     # print (a)
    #     # return a
    #
    # def __len__(self):
    #     return len(self.data_source)


