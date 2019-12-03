import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import torch
import torch.utils.data as utils

path='D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Prepared_Data/Spleen/Validation/spleen_6.nii.gz'
content=os.listdir(path)

my_data=[]
for i in content:
    slice=np.load(path+'/'+i)
    my_data.append(slice)
tensor_x = torch.stack([torch.Tensor(i) for i in my_data])
print (tensor_x.shape)

#create data loade
my_dataset = utils.TensorDataset(tensor_x)
my_dataloader = utils.DataLoader(my_dataset)