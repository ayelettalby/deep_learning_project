import numpy as np
import matplotlib.pyplot as plt
image=np.load('E:/Deep learning/Datasets_organized/Prepared_Data/BRATS/Training/Brats18_2013_0_1/slice30.npy')
#image=np.load('E:/Deep learning/Datasets_organized/Prepared_Data/Spleen/Training/spleen_2.nii.gz/slice16.npy')
image1=image[:,:,0]
image2=image[:,:,1]
image3=image[:,:,2]
print (image.shape, image1.shape)
fig=plt.figure()
plt.subplot(1,3,1)
plt.imshow(image1,cmap='gray')
plt.title('T2')
plt.subplot(1,3,2)
plt.imshow(image2,cmap="gray")
plt.title('T1ce')
plt.subplot(1,3,3)
plt.imshow(image3,cmap="gray")
plt.title('T1')
plt.show()

# bottom_thresh=-800
# top_thresh=800
#
# new_image=np.copy(image)
# new_image[image<=bottom_thresh]=bottom_thresh
# new_image[image>=top_thresh]=top_thresh
#
#
# fig=plt.figure()
# #fig=plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(image, cmap="gray")
# plt.title('old image')
# plt.subplot(1,2,2)
# plt.title('new image')
# plt.imshow(new_image, cmap="gray")
# plt.show()
