# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import math
from skimage import measure
from skimage.morphology import convex_hull_image
import SimpleITK as sitk

#import cv2
#import matplotlib.pyplot as plt

data_path = sys.argv[1]

print "data_path: ", data_path

#################################################################### Gen some folders
list_path = os.path.join(data_path, 'lists')
if not os.path.exists(list_path):
    os.makedirs(list_path)

model_path = os.path.join(data_path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

log_path = os.path.join(data_path, 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)

def in_training_set(i):
    # fold_remainder = folds - total_samples % folds
    # fold_size = (total_samples - total_samples % folds) / folds
    # start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    # end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    start_index = 20
    end_index = 36
    #start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    #end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (i >= start_index and i < end_index)

# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')

# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')

# computing the DSC together with other values based on the label and prediction volumes


# ------ defining the common variables used throughout the entire flowchart ------



##################################################################### functions
def C2F_bounding_box(SEG_C):

    MM=np.zeros([4,SEG_C.shape[0]])
    R=np.zeros([4,1])

    for sli in range(SEG_C.shape[0]):
        SEG_C_=SEG_C[sli,:,:]
        #SEG_C_[SEG_C_ < 0.5] = 0
        NZ_arr = np.nonzero(SEG_C_)
        MM[0, sli] = 0 if not len(NZ_arr[0]) else min(NZ_arr[0])
        MM[1, sli] = 0 if not len(NZ_arr[0]) else max(NZ_arr[0])
        MM[2, sli] = 0 if not len(NZ_arr[1]) else min(NZ_arr[1])
        MM[3, sli] = 0 if not len(NZ_arr[1]) else max(NZ_arr[1])

    temp=np.nonzero(MM[0, :])
    R[0,0]=np.min(MM[0, temp])
    R[1,0]=np.max(MM[1, :])
    temp=np.nonzero(MM[2, :])
    R[2,0]=np.min(MM[2, temp])
    R[3,0]=np.max(MM[3, :])
    #print(R)

    return R

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)
    return imgs


def conv_mask(prob):
    for sli in range(prob.shape[0]):
        cur_prob = prob[sli]
        if np.max(cur_prob) < 0.5:
            continue
        temp = convex_hull_image(cur_prob > 0.5)
        prob[sli]=temp
    return prob


def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.logical_and(pred, label).sum()
    return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum

def Label_connection(prob_):
    area_list = []
    binary = prob_ > 0.5
    label_prob_ = measure.label(binary)
    region_label_prob_ = measure.regionprops(label_prob_)
    for region in region_label_prob_:
        area_list.append(region.area)
    idx_max = np.argmax(area_list)
    binary[label_prob_ != idx_max + 1] = 0
    temp_prob_ = binary
    temp_prob_.astype(np.int16)
    #plt.figure('Orginal Prob'), plt.imshow(temp_prob_, cmap='gray')
    #plt.show()
    return temp_prob_

def Find_center(gs_array):
    X_sum = 0
    Y_sum = 0
    Pixel_sum = 0
    # print(np.shape(gs_array))
    for iii in range(511):
        a = gs_array[iii,:]
        #print(np.max(a))
        if np.max(gs_array) ==0:
            X_center = 255
            Y_center = 255
        else:
            idx = [idx for (idx, val) in enumerate(a) if val > 0.5]
            X_sum = X_sum + iii * len(idx)
            Y_sum = Y_sum + np.sum(idx)
            Pixel_sum = Pixel_sum + len(idx)

    if Pixel_sum == 0:
        X_center = 255
        Y_center = 255
    else:
        X_center = int(X_sum / Pixel_sum)
        Y_center = int(Y_sum / Pixel_sum)

    #print(X_center)
    return X_center, Y_center

def preprocess(imgs):
    """add one more axis as tf require"""
    imgs = imgs[..., np.newaxis]
    return imgs

def preprocess_front(imgs):
    imgs = imgs[np.newaxis, ...]
    return imgs

# returning the binary label map by the organ ID (especially useful under overlapping cases)
#   label: the label matrix
#   organ_ID: the organ ID
def is_organ(label, organ_ID):
    return label == organ_ID

def pad_ZY(image, padval, xmax, ymax):
    """pad image with zeros to reach dimension as (row_max, col_max)
    """
    if np.shape(image)[0] > xmax and np.shape(image)[1] > ymax:
        image = image[0:xmax , 0:ymax ]
    elif np.shape(image)[0] > xmax:
        image = image[0:xmax, :]
    elif np.shape(image)[1] > ymax:
        image = image[:, 0:ymax]

    pad_X_0 = np.int(math.ceil((xmax - image.shape[0])/2))
    pad_X_1 = xmax - image.shape[0]-pad_X_0
    pad_Y_0 = np.int(math.ceil((ymax - image.shape[1])/2))
    pad_Y_1 = ymax - image.shape[1]-pad_Y_0

    npad = ((pad_X_0, pad_X_1), (pad_Y_0, pad_Y_1))
    padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)

    #print(np.shape(padded))
    return padded

def Test_pad_ZY(image, padval, xmax, ymax):
    """pad image with zeros to reach dimension as (row_max, col_max)
    """
    if np.shape(image)[0] > xmax and np.shape(image)[1] > ymax:
        image = image[0:xmax, 0:ymax ]
    elif np.shape(image)[0] > xmax:
        image = image[0:xmax, :]
    elif np.shape(image)[1] > ymax:
        image = image[:, 0:ymax ]

    pad_X_0 = np.int(math.ceil((xmax - image.shape[0])/2))
    pad_X_1 = xmax - image.shape[0]-pad_X_0
    pad_Y_0 = np.int(math.ceil((ymax - image.shape[1])/2))
    pad_Y_1 = ymax - image.shape[1]-pad_Y_0
    #print(type(pad_X_0))

    npad = ((pad_X_0, pad_X_1), (pad_Y_0, pad_Y_1))
    #print(npad)
    padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)

    #print(np.shape(padded))
    return padded,pad_X_0,pad_X_1,pad_Y_0,pad_Y_1

def pad_2d(image, plane, padval, xmax, ymax, zmax):
    """pad image with zeros to reach dimension as (row_max, col_max)

    Params
    -----
    image : 2D numpy array
        image to pad
    dim : char
        X / Y / Z
    padval : int
        value to pad around
    xmax, ymax, zmax : int
        dimension to reach in x/y/z axis
    """
    #print(image.shape)
    if plane == 'X':
        npad = ((0, ymax - image.shape[0]), (0, zmax - image.shape[1]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
    elif plane =='Y':
        npad = ((0, xmax - image.shape[0]), (0, zmax - image.shape[1]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
    elif plane == 'Z':
        npad = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values=padval)

    return padded


#   determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]

