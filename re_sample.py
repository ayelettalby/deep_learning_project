import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# path='D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Seg_data_try/slice11.npy'
#
# def re_sample(slice, end_shape, order=3):
#     zoom_factor = [n / float(o) for n, o in zip(end_shape, slice.shape)]
#     if not np.all(zoom_factor == (1, 1)):
#         data = ndimage.zoom(slice, zoom=zoom_factor, order=order, mode='constant')
#     else:
#         data = slice
#     return data
#
# a = np.fromfunction(lambda x, y: x + y, (8, 8))
# print (a)
#
# b=re_sample(a, (4,4))
# print (b)

# img=np.load(path)
# print (img.shape)
# new_img=re_sample(img,(600,600))
# print (new_img.shape)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.imshow(new_img)
# plt.show()

