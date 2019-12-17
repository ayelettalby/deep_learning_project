import os
import numpy as np
import nibabel as nb
import csv
import matplotlib.pyplot as plt
import torch.utils.data as utils
from scipy import ndimage
import math



def re_sample(slice, end_shape, order=3):
    zoom_factor = [n / float(o) for n, o in zip(end_shape, slice.shape)]
    if not np.all(zoom_factor == (1, 1)):
        data = ndimage.zoom(slice, zoom=zoom_factor, order=order, mode='constant')
    else:
        data = slice
    return data

def pre_process(slice,bottom_thresh,top_thresh): #receives a 2D image and intensity thresholds and performs windowing
    new_image = np.copy(slice)
    new_image[slice <= bottom_thresh] = bottom_thresh
    new_image[slice >= top_thresh] = top_thresh
    return(new_image)

def get_truncate_index(scan,num_slices,percent): #function that takes only some slices on z axis
    top_index=num_slices-1
    bottom_index=0

    for i in range (num_slices):
        slice = scan[:,:,i]
        result = sum(sum(slice))
        if result != 0:
            bottom_index = i
            break
    for i in range (num_slices-1,-1,-1):
        slice = scan[:, :, i]
        result = sum(sum(slice))
        if result != 0:
            top_index = i
            break

    num_good_slices = top_index - bottom_index + 1
    on_top = min(num_slices - top_index - 1, math.ceil(percent*num_good_slices))
    on_bottom = min(bottom_index, math.ceil(percent*num_good_slices))
    num_slices_each_side = min(on_top, on_bottom)

    final_top = top_index + num_slices_each_side
    final_bottom = bottom_index - num_slices_each_side

    return final_bottom,final_top

def make_binary(label):
    label[label!=0] = 1
    return None

def scan_to_slices(path, truncate=False, binary=False):
    os.mkdir(save_path+'/'+task_name, 777)
    os.mkdir(save_path + '/'+task_name+'/Training', 777)
    os.mkdir(save_path + '/'+task_name+'/Validation', 777)
    os.mkdir(save_path + '/'+task_name+'/Test', 777)

    #create csv for metadata
    meta_data = open(save_path+'/'+task_name + '/' + task_name+ '_metadata.csv' , mode='w')
    wr = csv.writer(meta_data, lineterminator='\n')

    for set in ['Training','Validation','Test']:

        files=os.listdir(path + '/' + set)###
        for file in files:
            new_path = save_path + '/'+task_name+'/' + set+'/'+file
            label_path = save_path + '/'+task_name+'/' + set+'/'+'Labels_'+file
            os.mkdir(new_path, 777)
            os.mkdir(label_path,777)

            img = nb.load(path + '/' + set+'/'+file)
            label = nb.load(path + '/Labels' + '/' + file)

            data = img.get_data()
            print(data.shape)
            print(label.shape)
            label = label.get_data()

            num_slices = data.shape[2]

            if truncate==True:
                bottom_index,top_index = get_truncate_index(label,num_slices,0.2)
                data = data[:, :, bottom_index:top_index]
                label = label[:, :, bottom_index:top_index]

            if binary==True:
                make_binary(label)


            num_slices = data.shape[2]
            data = np.dstack((data[:, :, 0], data, data[:, :, num_slices - 1])) #padding the slices
            label = np.dstack((label[:, :, 0], label, label[:, :, num_slices - 1])) #padding the slices
            output = np.empty((end_shape[0],end_shape[1],3), dtype=float, order='C')

        # create a stack of our "2.5D slices", each containing 3 slices

            for i in range(1, num_slices+1):
                # adding relevant data to csv:
                # scan, number of slice, set(training/val/test), slice path, label path

                wr.writerow([file, str(i), set, new_path + '/slice' + str(i), label_path + '/slice' + str(i)])
                output_new = output

                #create three slices from data and re samples them to wanted size:

                middle_slice=re_sample(data[:,:,i], end_shape)
                bottom_slice=re_sample(data[:,:,i-1],end_shape)
                top_slice=re_sample(data[:, :, i+1],end_shape)

                label_new = re_sample(label[:,:,i], end_shape,order=1)


                #stack the three slices to form 2.5D slices
                output_new[:,:,1]=middle_slice #main middle slice
                output_new[:,:,0]=bottom_slice #bottom slice
                output_new[:, :, 2] = top_slice #top slice


                np.save(new_path+'/slice'+str(i), output_new)
                np.save(label_path+'/slice'+str(i),label_new)
    meta_data.close()
    return None
############################################
path= 'C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Prostate data' #change to relevant source path
task_name='Prostate'
save_path='C:/Users/Ayelet/Desktop/school/fourth_year/deep_learning_project/ayelet_shiri/Prepared_Data' #change to where you want to save data
end_shape= (320,320) #wanted slice shape after resampling

def main(path):
    scan_to_slices(path,truncate=False,binary=False)

if __name__ == '__main__':
    main(path)