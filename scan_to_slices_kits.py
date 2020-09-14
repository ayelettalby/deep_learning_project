import os
import numpy as np
import nibabel as nb
import csv
import matplotlib.pyplot as plt
import torch.utils.data as utils
from scipy import ndimage
import math
import imageio


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
        slice = scan[i,:,:]
        result = sum(sum(slice))
        if result != 0:
            bottom_index = i
            break
    for i in range (num_slices-1,-1,-1):
        slice = scan[i,:, :]
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

def main(path, task_name,end_shape,truncate=False, binary=False):
    os.mkdir(save_path+'/'+task_name, 777)

    #create csv for metadata
    meta_data = open(save_path+'/'+task_name + '/' + task_name+ '_metadata.csv' , mode='w')
    wr = csv.writer(meta_data, lineterminator='\n')

    for set in ['Training','Validation','Test']:
        files=os.listdir(path + '/' + set)
        new_path = save_path + '/' + task_name + '/' + set
        label_path = save_path + '/' + task_name  +'/' + set + '_Labels'
        os.mkdir(new_path, 777)
        os.mkdir(label_path, 777)

        for ind,file in enumerate(files,0):
            img = nb.load(path + '/' + set+'/'+file)
            label = nb.load(path + '/Labels' + '/' + file)

            data = img.get_data()

            header=img.header
            originalSpacing=header['pixdim'][1]
            spacingFactor=end_shape[0]/data.shape[0]
            newSpacing=originalSpacing*spacingFactor
            label = label.get_data()

            num_slices = data.shape[0]
            print (num_slices)

            if truncate==True:
                bottom_index,top_index = get_truncate_index(label,num_slices,0.2)
                data = data[bottom_index:top_index,:, :]
                label = label[bottom_index:top_index, :, :]

            if binary==True:
                make_binary(label)

            num_slices = data.shape[0]
            # data = np.dstack((data[0,:, :], data, data[num_slices - 1,:, : ])) #padding the slices
            # label = np.dstack((label[0,:, :], label, label[num_slices - 1,:, : ])) #padding the slices
            output = np.empty((3,end_shape[0],end_shape[1]), dtype=float, order='C')


            for i in range(2, num_slices-1):
                # adding relevant data to csv:
                # scan, number of slice, set(training/val/test), spacing, slice path, label path
                wr.writerow([file, str(i), set, str(newSpacing), new_path + '/slice' + str(i), label_path + '/slice' + str(i)])
                output_new=output

                output_new[1, :, :] = re_sample(data[i,:, :], end_shape)  # middle slice
                output_new[0, :, :] = re_sample(data[i - 1,:, : ], end_shape)  # bottom slice
                output_new[2, :, :] = re_sample(data[i + 1,:, : ], end_shape)  # top slice
                label_new = re_sample(label[i,:, :], end_shape, order=1)

                # if need rotating:
                # output_new[1,:,:]=np.fliplr(np.rot90(re_sample(data[:,:,i], end_shape))) #middle slice
                # output_new[0,:,:]=np.fliplr(np.rot90(re_sample(data[:,:,i-1],end_shape)))#bottom slice
                # output_new[2,:, :]=np.fliplr(np.rot90(re_sample(data[:, :, i+1],end_shape))) #top slice
                # label_new = np.fliplr(np.rot90(re_sample(label[:, :, i], end_shape, order=1)))


                np.save(new_path + '/' + str(ind) + '_slice_' + str(i-1), output_new)
                np.save(label_path + '/' + str(ind) + '_slice_' + str(i-1), label_new)

    meta_data.close()
    return None
############################################
path= r'G:\Deep learning\Datasets_organized\Kits' #change to relevant source path
task_name='Kits'
save_path='G:/Deep learning/Datasets_organized/Prepared_Data' #change to where you want to save data
end_shape= (384,384) #wanted slice shape after resampling

if __name__ == '__main__':
    main(path,task_name,end_shape,truncate=True,binary=True)