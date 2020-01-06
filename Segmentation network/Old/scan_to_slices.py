import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt


##change the following based on task:
path= 'C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Spleen data' #change to relevant source path
task_name='Spleen'
save_path='C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Prepared_Data' #change to where you want to save data
##

def scan_to_slices(path):

    os.mkdir(save_path+'/'+task_name, 777) #opens a folder for the relevant task
    files=os.listdir(path)
    for file in files:
        new_path = save_path + '/'+task_name + file
        os.mkdir(new_path, 777)
        img = nb.load(path+'/'+file)
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