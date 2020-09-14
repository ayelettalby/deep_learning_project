import os
import shutil

path = r'G:\Deep learning\Cervix\Cervix\RawData\Training\img'
folders = os.listdir(path)

for i,folder in enumerate(folders):
    file_name = os.listdir(path + '/' + folder)
    print(file_name)
    file_path = path + '/' + folder  + '/' + file_name[0]
    print(file_path)
    shutil.copyfile(file_path, path+'/'+str(i+1)+'.nii')
