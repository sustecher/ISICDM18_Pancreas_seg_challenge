# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import math
from skimage import measure
from skimage.morphology import convex_hull_image
import SimpleITK as sitk
from scipy import ndimage as nd
#import cv2
#import matplotlib.pyplot as plt


#log_path = os.path.join(data_path, 'logs')
#if not os.path.exists(log_path):
#    os.makedirs(log_path)

##################################################################### functions
def Save_AS_NII(data_img,SEG_Arrray,results_save_path,filename):
    itkimage = sitk.GetImageFromArray(SEG_Arrray, isVector=False)
    itkimage.SetSpacing(data_img.GetSpacing())
    itkimage.SetOrigin(data_img.GetOrigin())
    itkimage.SetDirection(data_img.GetDirection())
    sitk.WriteImage(itkimage, results_save_path + filename, True)

def Fixed_shape_3D_resize(imgs,shape_output):
    #order The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
    shape_input=imgs.shape
    zf0=np.float(shape_output[0])/np.float(shape_input[0])
    zf1=np.float(shape_output[1])/np.float(shape_input[1])
    zf2=np.float(shape_output[2])/np.float(shape_input[2])
    #print(zf0,zf1,zf2)
    new_imgs = nd.zoom(imgs, [zf0, zf1, zf2], order=0)
    new_imgs[new_imgs>=0.5] = 1
    new_imgs[new_imgs<0.5] = 0

    return new_imgs

def ThreeD_resize(imgs,Normlize_VS,img_vs):
    #order The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
    zf0=img_vs[0]/Normlize_VS[0]
    zf1=img_vs[1]/Normlize_VS[1]
    zf2=img_vs[2]/Normlize_VS[2]
    new_imgs = nd.zoom(imgs, [zf0, zf1, zf2], order=0)
    return new_imgs

def Range_from_mask(mask):
    arr=np.nonzero(mask)
    #print(min(arr[1]))
    #margin_xy=20
    #margin_z=5
    margin_xy=20
    margin_z=5
    Min_z = int(min(arr[0]) - margin_z)
    Max_z = int(max(arr[0]) + margin_z)
    Min_y=int(min(arr[1])-margin_xy)
    Max_y=int(max(arr[1])+margin_xy)
    Min_x=int(min(arr[2])-margin_xy)
    Max_x=int(max(arr[2])+margin_xy)

    Min_z = max(0, Min_z)
    Max_z = min(np.shape(mask)[0], Max_z)
    Min_y = max(0, Min_y)
    Max_y = min(np.shape(mask)[1], Max_y)
    Min_x = max(0, Min_x)
    Max_x = min(np.shape(mask)[2], Max_x)

    return Min_z,Max_z,Min_y,Max_y,Min_x,Max_x

def Enlarge_crop_shape(Min_xyz,Max_xyz,nor_size,bounding_size):
    center_xyz=int((Min_xyz+Max_xyz)/2)
    Min_xyz=int(center_xyz-nor_size/2)
    Min_xyz=max(0,Min_xyz)

    Max_xyz=int(center_xyz+nor_size/2)
    Max_xyz=min(Max_xyz,bounding_size)
    return Min_xyz,Max_xyz

def Find_lesion(mask,axis=0):
    if axis==0:
        max_axis0=np.zeros([mask.shape[0],1])
        for i in range(mask.shape[0]):
            #print(np.max(mask[i,:,:]))
            max_axis0[i]=np.max(mask[i,:,:])
        #print(max_axis0.shape)
        arr=np.nonzero(max_axis0)
        #print(arr)
    elif axis==1:
        max_axis1=np.zeros([mask.shape[1],1])
        for i in range(mask.shape[1]):
            #print(np.max(mask[:,i,:]))
            max_axis1[i]=np.max(mask[:,i,:])
        arr=np.nonzero(max_axis1)
    elif axis==2:
        max_axis2=np.zeros([mask.shape[2],1])
        for i in range(mask.shape[2]):
            #print(np.max(mask[:,:,i]))
            max_axis2[i]=np.max(mask[:,:,i])
        #print(max_axis0.shape)
        arr=np.nonzero(max_axis2)
        #print(arr)
    return arr[0]

def Avoid_out_of_Range(arr,margin,dimension_idx,XYZMAX,Maskshape):
    Min_ = int(min(arr[dimension_idx]) - margin)
    Max_ = int(max(arr[dimension_idx]) + margin)
    if Max_ - Min_ > XYZMAX:
        Mid_ = int((Max_ + Min_) / 2)
        Min_ = int(Mid_ - XYZMAX / 2)
        Max_ = int(Mid_ + XYZMAX / 2)

    Min_ = max(0, Min_)
    Max_ = min(Maskshape[dimension_idx], Max_)
    return Min_,Max_

def GetBoundingBox(Mask,margin,ZMAX,YMAX,XMAX):
    arr = np.nonzero(Mask)
    Min_z, Max_z = Avoid_out_of_Range(arr, margin, 0, ZMAX, Mask.shape)
    Min_y, Max_y = Avoid_out_of_Range(arr, margin, 1, YMAX, Mask.shape)
    Min_x, Max_x = Avoid_out_of_Range(arr, margin, 2, XMAX, Mask.shape)
    return Min_z, Max_z, Min_y, Max_y,Min_x, Max_x

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

def Label_connection(prob_,TH):
    area_list = []
    #binary = prob_ > 0.5
    binary = prob_ > TH
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

def preprocess(imgs):
    """add one more axis as tf require"""
    imgs = imgs[..., np.newaxis]
    return imgs

def preprocess_front(imgs):
    imgs = imgs[np.newaxis, ...]
    return imgs

#   determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]

