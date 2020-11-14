"""
This code is to
1. Create train & test input to Network as numpy arrays
2. Load the train & test numpy arrays
"""

import numpy as np
import os
from utils import *
from skimage import measure
import scipy.io as scio
import math
# data type to save as np array
#Prob_FD1/Pred/Testing/ProbMap_class_1_001.nii
code_path = sys.argv[1] +'Code_2D'
SEG3D_path=sys.argv[1] +'Code_3D/src/outputFiles/'
data_path = sys.argv[2]
current_fold = int(sys.argv[3])

ZMAX = int(sys.argv[4])
YMAX = int(sys.argv[5])
XMAX = int(sys.argv[6])

margin = int(sys.argv[7])
low_range = int(sys.argv[8])
high_range = int(sys.argv[9])

## Fixed parameters in 2D model
Normlize_VS=[1,1,1]
Normlize_shape=[120,140,200]
FD=4

def create_train_data(current_fold):
    """
    Crop each slice by its ground truth bounding box,
    then pad zeros to form uniform dimension,
    rescale pixel intensities to [0,1]
    """
    # get the list of image and label number of current_fold

    images_z = []
    labels_z = []
    probs_z = []

    num_patient=20
    for l in range(num_patient):
        idx=str(l+1).zfill(2)
        print(l,idx)
        dir_img = data_path + 'Train/Img/' + idx + '.nii.gz'
        dir_gs = data_path + 'Train/GroundTruth/' + idx + '_seg.nii.gz'
        dir_segg3d=SEG3D_path+ 'Prob_FD'+str(current_fold)+'/Pred/Testing/ProbMap_class_1_'+idx+'.nii'

        data_img = sitk.ReadImage(dir_img)
        IMG_ori = sitk.GetArrayFromImage(data_img)

        data_img = sitk.ReadImage(dir_gs)
        GS_ori = sitk.GetArrayFromImage(data_img)
        GS_ori[GS_ori > 0.5] = 1

        data_seg3d = sitk.ReadImage(dir_segg3d)
        SEG3D_crop = sitk.GetArrayFromImage(data_seg3d)

        Spacing_img = data_img.GetSpacing()[::-1]

        IMG_array = ThreeD_resize(IMG_ori, Normlize_VS, Spacing_img)
        GS_array = ThreeD_resize(GS_ori, Normlize_VS, Spacing_img)
        PROB_array=np.zeros(GS_array.shape)
        #print(GS_ori.shape,GS_array.shape)
        #print(SEG3D_crop.shape)
        Min_z, Max_z, Min_x, Max_x, Min_y, Max_y = Range_from_mask(GS_array)
        # print(Min_z, Max_z, Min_x, Max_x, Min_y, Max_y)
        if Max_z - Min_z < Normlize_shape[0]:
            Min_z, Max_z = Enlarge_crop_shape(Min_z, Max_z, Normlize_shape[0], IMG_array.shape[0])
        if Max_x - Min_x < Normlize_shape[1]:
            Min_x, Max_x = Enlarge_crop_shape(Min_x, Max_x, Normlize_shape[1], IMG_array.shape[1])
        if Max_y - Min_y < Normlize_shape[2]:
            Min_y, Max_y = Enlarge_crop_shape(Min_y, Max_y, Normlize_shape[2], IMG_array.shape[2])
        PROB_array[Min_z: Max_z, Min_x: Max_x, Min_y:Max_y]=SEG3D_crop
        Mask=Label_connection(PROB_array,0.5)
        PROB_array = PROB_array*Mask
        IMG_array[IMG_array>high_range]=high_range
        IMG_array[IMG_array<low_range]=low_range

        Min_z, Max_z, Min_y, Max_y, Min_x, Max_x = GetBoundingBox(Mask,margin,ZMAX,YMAX,XMAX)
        IMG_crop_padding = np.zeros([ZMAX,YMAX,XMAX])+low_range
        GS_crop_padding = np.zeros([ZMAX, YMAX, XMAX])
        PROB_crop_padding = np.zeros([ZMAX, YMAX, XMAX])
        left_z=int(ZMAX/2-(Max_z-Min_z)/2)
        left_y=int(YMAX/2-(Max_y-Min_y)/2)
        left_x=int(XMAX/2-(Max_x-Min_x)/2)

        #print(Min_z, Max_z, Min_x, Max_x, Min_y, Max_y)
        IMG_crop_padding[left_z:left_z+Max_z-Min_z,left_y:left_y+Max_y-Min_y,left_x:left_x+Max_x-Min_x]=IMG_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]
        GS_crop_padding[left_z:left_z+Max_z-Min_z,left_y:left_y+Max_y-Min_y,left_x:left_x+Max_x-Min_x]=GS_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]
        PROB_crop_padding[left_z:left_z+Max_z-Min_z,left_y:left_y+Max_y-Min_y,left_x:left_x+Max_x-Min_x]=PROB_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]

        #IMG_crop_padding=(IMG_crop_padding-low_range)/(high_range-low_range)
        M=np.mean(IMG_array[GS_array>0.5])
        STD=np.mean(IMG_array[GS_array>0.5])
        IMG_crop_padding=(IMG_crop_padding-M)/STD

        #print(IMG_array.shape,GS_array.shape,PROB_array.shape)
        ##############################
        '''
        Fusion_MASK=np.zeros([ZMAX,YMAX,XMAX])
        Fusion_MASK[GS_crop_padding+PROB_crop_padding>0.5]=1
        lesions_idx0= Find_lesion(Fusion_MASK, axis=0)
        Crop_image = IMG_crop_padding[lesions_idx0, :, :]
        Crop_label = GS_crop_padding[lesions_idx0, :, :]
        Crop_prob = PROB_crop_padding[lesions_idx0, :, :]
        '''
        Crop_image = IMG_crop_padding[left_z:left_z+Max_z-Min_z,:,:]
        Crop_label = GS_crop_padding[left_z:left_z+Max_z-Min_z,:,:]
        Crop_prob = PROB_crop_padding[left_z:left_z+Max_z-Min_z,:,:]
        #print(np.shape(Crop_image))
        images_z.append(Crop_image)
        labels_z.append(Crop_label)
        probs_z.append(Crop_prob)

    images_z = np.concatenate(images_z, axis=0).reshape(-1, YMAX, XMAX, 1)
    labels_z = np.concatenate(labels_z, axis=0).reshape(-1, YMAX, XMAX, 1)
    probs_z = np.concatenate(probs_z, axis=0).reshape(-1, YMAX, XMAX, 1)
    #scio.savemat(code_path + 'NPY_data/Trans_X_FD'+str(current_fold)+'.mat', {'images':images_z})
    #scio.savemat(code_path + 'NPY_data/Trans_y_FD'+str(current_fold)+'.mat', {'labels':labels_z})
    #scio.savemat(code_path + 'NPY_data/Trans_M_FD'+str(current_fold)+'.mat', {'masks':masks_z})
    np.save(code_path + 'NPY_data/Trans_X_FD' + str(current_fold) + '.npy', images_z)
    np.save(code_path + 'NPY_data/Trans_y_FD' + str(current_fold) + '.npy', labels_z)
    np.save(code_path + 'NPY_data/Trans_P_FD' + str(current_fold) + '.npy', probs_z)

def create_train_data_25D(current_fold):
    """
    Crop each slice by its ground truth bounding box,
    then pad zeros to form uniform dimension,
    rescale pixel intensities to [0,1]
    """
    # get the list of image and label number of current_fold
    images_Trans = []
    labels_Trans = []
    probs_Trans = []
    images_Coron = []
    labels_Coron = []
    probs_Coron = []
    images_Sagit = []
    labels_Sagit = []
    probs_Sagit = []

    num_patient = 20
    for l in range(num_patient):
        idx = str(l + 1).zfill(2)
        print(l, idx)
        dir_img = data_path + 'Train/Img/' + idx + '.nii.gz'
        dir_gs = data_path + 'Train/GroundTruth/' + idx + '_seg.nii.gz'
        dir_segg3d = SEG3D_path + 'Prob_FD' + str(current_fold) + '/Pred/Testing/ProbMap_class_1_' + idx + '.nii'

        data_img = sitk.ReadImage(dir_img)
        IMG_ori = sitk.GetArrayFromImage(data_img)

        data_img = sitk.ReadImage(dir_gs)
        GS_ori = sitk.GetArrayFromImage(data_img)
        GS_ori[GS_ori > 0.5] = 1

        data_seg3d = sitk.ReadImage(dir_segg3d)
        SEG3D_crop = sitk.GetArrayFromImage(data_seg3d)

        Spacing_img = data_img.GetSpacing()[::-1]

        IMG_array = ThreeD_resize(IMG_ori, Normlize_VS, Spacing_img)
        GS_array = ThreeD_resize(GS_ori, Normlize_VS, Spacing_img)
        PROB_array = np.zeros(GS_array.shape)
        # print(GS_ori.shape,GS_array.shape)
        # print(SEG3D_crop.shape)
        Min_z, Max_z, Min_x, Max_x, Min_y, Max_y = Range_from_mask(GS_array)
        # print(Min_z, Max_z, Min_x, Max_x, Min_y, Max_y)
        if Max_z - Min_z < Normlize_shape[0]:
            Min_z, Max_z = Enlarge_crop_shape(Min_z, Max_z, Normlize_shape[0], IMG_array.shape[0])
        if Max_x - Min_x < Normlize_shape[1]:
            Min_x, Max_x = Enlarge_crop_shape(Min_x, Max_x, Normlize_shape[1], IMG_array.shape[1])
        if Max_y - Min_y < Normlize_shape[2]:
            Min_y, Max_y = Enlarge_crop_shape(Min_y, Max_y, Normlize_shape[2], IMG_array.shape[2])
        PROB_array[Min_z: Max_z, Min_x: Max_x, Min_y:Max_y] = SEG3D_crop
        Mask = Label_connection(PROB_array, 0.5)
        PROB_array = PROB_array * Mask
        IMG_array[IMG_array > high_range] = high_range
        IMG_array[IMG_array < low_range] = low_range

        Min_z, Max_z, Min_y, Max_y, Min_x, Max_x = GetBoundingBox(Mask, margin, ZMAX, YMAX, XMAX)
        IMG_crop_padding = np.zeros([ZMAX, YMAX, XMAX]) + low_range
        GS_crop_padding = np.zeros([ZMAX, YMAX, XMAX])
        PROB_crop_padding = np.zeros([ZMAX, YMAX, XMAX])
        left_z = int(ZMAX / 2 - (Max_z - Min_z) / 2)
        left_y = int(YMAX / 2 - (Max_y - Min_y) / 2)
        left_x = int(XMAX / 2 - (Max_x - Min_x) / 2)

        # print(Min_z, Max_z, Min_x, Max_x, Min_y, Max_y)
        IMG_crop_padding[left_z:left_z + Max_z - Min_z, left_y:left_y + Max_y - Min_y,
        left_x:left_x + Max_x - Min_x] = IMG_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]
        GS_crop_padding[left_z:left_z + Max_z - Min_z, left_y:left_y + Max_y - Min_y,
        left_x:left_x + Max_x - Min_x] = GS_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]
        PROB_crop_padding[left_z:left_z + Max_z - Min_z, left_y:left_y + Max_y - Min_y,
        left_x:left_x + Max_x - Min_x] = PROB_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]

        # IMG_crop_padding=(IMG_crop_padding-low_range)/(high_range-low_range)
        M = np.mean(IMG_array[GS_array > 0.5])
        STD = np.mean(IMG_array[GS_array > 0.5])
        IMG_crop_padding = (IMG_crop_padding - M) / STD

        # print(IMG_array.shape,GS_array.shape,PROB_array.shape)
        #############################
        Crop_image = IMG_crop_padding[left_z:left_z + Max_z - Min_z, :, :]
        Crop_label = GS_crop_padding[left_z:left_z + Max_z - Min_z, :, :]
        Crop_prob = PROB_crop_padding[left_z:left_z + Max_z - Min_z, :, :]
        # print(np.shape(Crop_image))
        images_Trans.append(Crop_image)
        labels_Trans.append(Crop_label)
        probs_Trans.append(Crop_prob)

        Crop_image = IMG_crop_padding[:, left_y:left_y+Max_y-Min_y:2, :]
        Crop_label = GS_crop_padding[:, left_y:left_y+Max_y-Min_y:2, :]
        Crop_prob = PROB_crop_padding[:, left_y:left_y+Max_y-Min_y:2, :]
        Crop_image = np.transpose(Crop_image, [1, 0, 2])
        Crop_label = np.transpose(Crop_label, [1, 0, 2])
        Crop_prob = np.transpose(Crop_prob, [1, 0, 2])
        images_Coron.append(Crop_image)
        labels_Coron.append(Crop_label)
        probs_Coron.append(Crop_prob)

        Crop_image = IMG_crop_padding[:, :, left_x:left_x+Max_x-Min_x:2]
        Crop_label = GS_crop_padding[:, :, left_x:left_x+Max_x-Min_x:2]
        Crop_prob = PROB_crop_padding[:, :, left_x:left_x+Max_x-Min_x:2]
        Crop_image = np.transpose(Crop_image, [2, 1, 0])
        Crop_label = np.transpose(Crop_label, [2, 1, 0])
        Crop_prob = np.transpose(Crop_prob, [2, 1, 0])
        images_Sagit.append(Crop_image)
        labels_Sagit.append(Crop_label)
        probs_Sagit.append(Crop_prob)

    images_Trans = np.concatenate(images_Trans, axis=0).reshape(-1, YMAX, XMAX, 1)
    labels_Trans = np.concatenate(labels_Trans, axis=0).reshape(-1, YMAX, XMAX, 1)
    probs_Trans = np.concatenate(probs_Trans, axis=0).reshape(-1, YMAX, XMAX, 1)
    np.save(code_path + 'NPY_data/Trans_X_FD' + str(current_fold) + '.npy', images_Trans)
    np.save(code_path + 'NPY_data/Trans_y_FD' + str(current_fold) + '.npy', labels_Trans)
    np.save(code_path + 'NPY_data/Trans_P_FD' + str(current_fold) + '.npy', probs_Trans)

    images_Coron = np.concatenate(images_Coron, axis=0).reshape(-1, ZMAX, XMAX, 1)
    labels_Coron = np.concatenate(labels_Coron, axis=0).reshape(-1, ZMAX, XMAX, 1)
    probs_Coron = np.concatenate(probs_Coron, axis=0).reshape(-1, ZMAX, XMAX, 1)
    np.save(code_path + 'NPY_data/Coron_X_FD' + str(current_fold) + '.npy', images_Coron)
    np.save(code_path + 'NPY_data/Coron_y_FD' + str(current_fold) + '.npy', labels_Coron)
    np.save(code_path + 'NPY_data/Coron_P_FD' + str(current_fold) + '.npy', probs_Coron)

    images_Sagit = np.concatenate(images_Sagit, axis=0).reshape(-1, YMAX, ZMAX, 1)
    labels_Sagit = np.concatenate(labels_Sagit, axis=0).reshape(-1, YMAX, ZMAX, 1)
    probs_Sagit = np.concatenate(probs_Sagit, axis=0).reshape(-1, YMAX, ZMAX, 1)
    np.save(code_path + 'NPY_data/Sagit_X_FD' + str(current_fold) + '.npy', images_Sagit)
    np.save(code_path + 'NPY_data/Sagit_y_FD' + str(current_fold) + '.npy', labels_Sagit)
    np.save(code_path + 'NPY_data/Sagit_P_FD' + str(current_fold) + '.npy', probs_Sagit)


if __name__ == '__main__':
    create_train_data_25D(current_fold)
    #create_train_data(current_fold)
