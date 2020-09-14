import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math
plt.figure()
for i in range(100):
    path=('G:/Deep learning/Datasets_organized/Prepared_Data/kits/Training_Labels/{}_slice_24.npy' .format(i))
    #path='D:/Documents/ASchool/year 4/my_seg_try/imgs/Slice0.npy'
    image=np.load(path)

    #print (image.shape)
    image=image[:,:]
    #scipy.misc.imsave('samplespl.jpg', image)

    plt.subplot(10,10,i+1)
    #plt.subplot(i%10+1,math.floor(i/5)+1,i+1)
    plt.imshow(image,cmap='gray')
plt.show()