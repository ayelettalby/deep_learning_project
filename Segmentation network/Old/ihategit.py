import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import torch.utils.data as utils

path= 'D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Spleen data'

def scan_to_slices(path):
    save_path='D:/Documents/ASchool/year 4/Deep learning project/ayelet_shiri/Prepared_Data'
    os.mkdir(save_path+'/Spleen', 777)
    os.mkdir(save_path + '/Spleen/Training', 777)
    os.mkdir(save_path + '/Spleen/Validation', 777)
    os.mkdir(save_path + '/Spleen/Test', 777)
    for set in ['/Training','/Validation','/Test']:
        files=os.listdir(path+set)###
        for file in files:
            new_path = save_path + '/Spleen/' + set+'/'+file
            os.mkdir(new_path, 777)
            img = nb.load(path+'/'+set+'/'+file)
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

                np.save(new_path+'/slice'+str(i), output_new)

    return None

def main(path):
    scan_to_slices(path)

if __name__ == '__main__':
    main(path)