import torch
from unet_2d import Unet_2D
import numpy as np
import matplotlib.pyplot as plt

batch_size=4
model=Unet_2D(encoder_name="resnet18",
                encoder_depth=5,
                encoder_weights="imagenet",
                decoder_use_batchnorm="True",
                decoder_channels=[256, 128, 64, 32, 16],
                in_channels=3,
                classes=2,
                activation='softmax')

model.load_state_dict(torch.load('D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Segmentation network/model_weights/model_weights.pth',map_location=torch.device('cpu')))

sample_image1=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test/spleen_47.nii.gz_slice_72.npy')
sample_image2=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test/spleen_47.nii.gz_slice_73.npy')
sample_image3=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test/spleen_47.nii.gz_slice_74.npy')
sample_image4=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test/spleen_47.nii.gz_slice_75.npy')
label0=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test_Labels/spleen_47.nii.gz_slice_72.npy')
label1=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test_Labels/spleen_47.nii.gz_slice_73.npy')
label2=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test_Labels/spleen_47.nii.gz_slice_74.npy')
label3=np.load('D:/Documents/ASchool/year 4/prepared/Spleen/Test_Labels/spleen_47.nii.gz_slice_75.npy')
labels=[label0,label1,label2,label3]
sample_image=np.empty((batch_size,3,384,384))
sample_image[0,:,:,:]=sample_image1
sample_image[1,:,:,:]=sample_image2
sample_image[2,:,:,:]=sample_image3
sample_image[3,:,:,:]=sample_image4


model=model.double()
output=model(torch.from_numpy(sample_image))
print (output.shape)
plt.figure()
for i in range(batch_size):
    print (type('label'+str(i)))

    plt.subplot(batch_size+1,3,i*3+1)
    plt.imshow(sample_image[i,1,:,:],cmap='gray')
    plt.subplot(batch_size+1,3,i*3+2)
    plt.imshow(labels[i],cmap='gray')
    plt.subplot(batch_size+1,3,i*3+3)
    plt.imshow(output[i,0,:,:].detach().numpy(),cmap='gray')
plt.show()