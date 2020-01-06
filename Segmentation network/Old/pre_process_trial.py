import numpy as np
import matplotlib.pyplot as plt
image=np.load('slice5.npy')
image=image[:,:,0]

bottom_thresh=-800
top_thresh=800

new_image=np.copy(image)
new_image[image<=bottom_thresh]=bottom_thresh
new_image[image>=top_thresh]=top_thresh


fig=plt.figure()
#fig=plt.figure()
plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title('old image')
plt.subplot(1,2,2)
plt.title('new image')
plt.imshow(new_image, cmap="gray")
plt.show()

