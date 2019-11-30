import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

path='spleen_2.nii.gz'
img=nb.load(path)
data=img.get_data()
fig=plt.figure(figsize=(9,10))
num_slices=data.shape[2]
output=[]
slices=[]
for i in range(0,num_slices):
    image=data[:,:,i]
    slices.append(image)


output.append([slices[0],slices[0],slices[1]])
for i in range(1,num_slices-1):
    output.append([slices[i-1],slices[i],slices[i+1]])
output.append([slices[num_slices-2],slices[num_slices-1],slices[num_slices-1]])

n=0
for i in range(0,5):
    for j in range(0,3):
        if j==0:
            pos="Top"
        elif j==1:
            pos="Middle"
        else:
            pos="Bottom"
        a=fig.add_subplot(5,3,n+1)
        plt.imshow(output[i][j],cmap="gray")
        a.set_title("Slice: %d, Pos: %s" %(i,pos))
        plt.subplots_adjust(hspace=1)
        n+=1
fig.suptitle("Sample slices of spleen CT, 2.5D",fontsize=20)
plt.show()
