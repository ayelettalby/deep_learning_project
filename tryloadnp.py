import os
import numpy as np
import matplotlib.pyplot as plt

path='D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Prepared_Data/Spleen/Validation/spleen_6.nii.gz/slice1.npy'
img=np.load(path)
plt.imshow(int(img))