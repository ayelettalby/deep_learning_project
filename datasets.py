import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# import a single image
path='spleen_2.nii.gz'
img=nb.load(path)
data=img.get_data()
num_slices=data.shape[2]
output=[]
slices=[]

#create a list of all the slices
for i in range(0,num_slices):
    image=data[:,:,i]
    slices.append(image)

#create a list of our "2.5D slices", each containing 3 slices
output.append([slices[0],slices[0],slices[1]])
for i in range(1,num_slices-1):
    output.append([slices[i-1],slices[i],slices[i+1]])
output.append([slices[num_slices-2],slices[num_slices-1],slices[num_slices-1]])

#plot sample 2.5D slices
n=0
fig=plt.figure()
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
