def scan_to_slices(path):
    import os
    import numpy as np
    import nibabel as nb
    import matplotlib.pyplot as plt

    # import a single image
    img = nb.load(path)
    data = img.get_data()
    num_slices = data.shape[2]

    data = np.dstack((data[:, :, 0], data, data[:, :, num_slices - 1])) #padding the slices
    output = np.empty((data.shape[0],data.shape[1],3), dtype=float, order='C')

    # create a list of our "2.5D slices", each containing 3 slices

    for i in range(1, num_slices+1):
        output_new=output
        output_new[:,:,1]=data[:,:,i] #main middle slice
        output_new[:,:,0]=data[:,:,i-1] #bottom slice
        output_new[:, :, 2] = data[:, :, i+1] #top slice
        np.save('slice'+str(i), output_new)

    return None