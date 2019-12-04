import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import torch.utils.data as utils


path= 'C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Spleen data' #change to relevant source path
task_name='Spleen'
save_path='C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Prepared_Data' #change to where you want to save data

def pre_process(slice,bottom_thresh,top_thresh): #receives a 2D image and intensity thresholds and performs windowing
    new_image = np.copy(slice)
    new_image[slice <= bottom_thresh] = bottom_thresh
    new_image[slice >= top_thresh] = top_thresh
    return(new_image)

def scan_to_slices(path):
    save_path='C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Prepared_Data'
    os.mkdir(save_path+'/'+task_name, 777)
    os.mkdir(save_path + '/'+task_name+'/Training', 777)
    os.mkdir(save_path + '/'+task_name+'/Validation', 777)
    os.mkdir(save_path + '/'+task_name+'/Test', 777)
    for set in ['/Training','/Validation','/Test']:
        files=os.listdir(path+set)###
        for file in files:
            new_path = save_path + '/'+task_name+'/' + set+'/'+file
            os.mkdir(new_path, 777)
            img = nb.load(path+'/'+set+'/'+file)
            data = img.get_data()
            num_slices = data.shape[2]

            data = np.dstack((data[:, :, 0], data, data[:, :, num_slices - 1])) #padding the slices
            output = np.empty((data.shape[0],data.shape[1],3), dtype=float, order='C')

        # create a stack of our "2.5D slices", each containing 3 slices

            for i in range(1, num_slices+1):
                output_new=output
                #create three slices from data and pre process them:
                middle_slice=pre_process(data[:,:,i],-800,800)
                bottom_slice=pre_process(data[:,:,i-1],-800,800)
                top_slice=pre_process(data[:, :, i+1],-800,800)

                #stack the three slices to form 2.5D slices
                output_new[:,:,1]=middle_slice #main middle slice
                output_new[:,:,0]=bottom_slice #bottom slice
                output_new[:, :, 2] = top_slice #top slice

                np.save(new_path+'/slice'+str(i), output_new)

    return None

def main(path):
    scan_to_slices(path)

if __name__ == '__main__':
    main(path)