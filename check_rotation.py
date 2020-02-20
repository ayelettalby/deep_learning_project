import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math
plt.figure()
for i in range(30):
    path=('E:/Deep learning/Datasets_organized/Prepared_Data/Hepatic Vessel/Training_Labels/%d_slice_20.npy' % (i))
    #path='D:/Documents/ASchool/year 4/my_seg_try/imgs/Slice0.npy'
    image=np.load(path)

    #print (image.shape)
    image=image[:,:]
    #scipy.misc.imsave('samplespl.jpg', image)

    plt.subplot(5,6,i+1)
    #plt.subplot(i%10+1,math.floor(i/5)+1,i+1)
    plt.imshow(image,cmap='gray')
plt.show()