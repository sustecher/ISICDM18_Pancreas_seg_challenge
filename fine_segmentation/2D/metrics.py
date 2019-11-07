from __future__ import division, print_function

from keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
from scipy.ndimage import morphology

import SimpleITK as sitk
from skimage.exposure import equalize_adapthist, equalize_hist
import cv2
import math
import random
from skimage import measure
from scipy import ndimage

smooth=1.0

def Avoid_out_of_range(Center_0,size_0,leng_0):
    if Center_0-size_0/2<0:
        Center_0=size_0/2+1

    if Center_0+size_0/2>leng_0:
        Center_0=leng_0-size_0/2-1
    new_center=int(Center_0)
    return new_center

def Stage3_Crop_with_size(image_array,label_array,sx,sy):
    leng_x=np.shape(label_array)[0]
    leng_y=np.shape(label_array)[1]
    #print(label_array.shape)
    arr = np.nonzero(label_array)
    mask1,mask2,kideney_num=Seperate_two_Kidey(label_array)
    #print(kideney_num)

    if kideney_num<2:
       centroid_x1, centroid_y1,centroid_z1 = ndimage.measurements.center_of_mass(mask1)
       C_x1=int(centroid_x1)+random.randint(-20,20)
       C_y1=int(centroid_y1)+random.randint(-20,20)
       Min_z1=int(min(arr[2]))
       Max_z1=int(max(arr[2]))
       # print(C_x1,C_y1)
       C_x1=Avoid_out_of_range(C_x1,sx,leng_x)
       C_y1=Avoid_out_of_range(C_y1,sy,leng_y)
       # print(C_x1,C_y1)
       crop_image1=np.zeros([sx,sy,Max_z1-Min_z1])
       crop_label1=np.zeros([sx,sy,Max_z1-Min_z1])
       crop_image = image_array[C_x1-int(sx/2):C_x1+int(sx/2),C_y1-int(sy/2):C_y1+int(sy/2),Min_z1:Max_z1]
       crop_label = label_array[C_x1-int(sx/2):C_x1+int(sx/2),C_y1-int(sy/2):C_y1+int(sy/2),Min_z1:Max_z1]
       crop_image = (crop_image - np.mean(crop_image)) /np.std(crop_image)
       crop_image=np.transpose(crop_image,(2,0,1)) 
       crop_label=np.transpose(crop_label,(2,0,1)) 
    else:
        # 要有两个
        for i in range(2):
            if i==0:
               mask=mask1
            else:
               mask=mask2
            centroid_x1, centroid_y1,centroid_z1 = ndimage.measurements.center_of_mass(mask)
            #print(centroid_x1, centroid_y1,centroid_z1)
            C_x1=int(centroid_x1)+random.randint(-20,20)
            C_y1=int(centroid_y1)+random.randint(-20,20)
            Min_z1=int(min(arr[2]))
            Max_z1=int(max(arr[2]))
           # print(C_x1,C_y1)
            C_x1=Avoid_out_of_range(C_x1,sx,leng_x)
            C_y1=Avoid_out_of_range(C_y1,sy,leng_y)
           # print(C_x1,C_y1)

            crop_image1=np.zeros([sx,sy,Max_z1-Min_z1])
            crop_label1=np.zeros([sx,sy,Max_z1-Min_z1])
            crop_image = image_array[C_x1-int(sx/2):C_x1+int(sx/2),C_y1-int(sy/2):C_y1+int(sy/2),Min_z1:Max_z1]
            crop_label = label_array[C_x1-int(sx/2):C_x1+int(sx/2),C_y1-int(sy/2):C_y1+int(sy/2),Min_z1:Max_z1]
            crop_image = (crop_image - np.mean(crop_image)) /np.std(crop_image)
            crop_image=np.transpose(crop_image,(2,0,1)) 
            crop_label=np.transpose(crop_label,(2,0,1)) 
            if i==0:
               crop_image_temp=crop_image
               crop_label_temp=crop_label
            else:
               crop_image=np.concatenate([crop_image_temp,crop_image],axis=0)
               crop_label=np.concatenate([crop_label_temp,crop_label],axis=0)
            #print(np.shape(crop_image))
            #print(np.shape(crop_label))
    return crop_image,crop_label

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

def preprocess(imgs):
    """add one more axis as tf require"""
    imgs = imgs[..., np.newaxis]
    return imgs

def preprocess_front(imgs):
    imgs = imgs[np.newaxis, ...]
    return imgs

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def pad_2D(image, padval, xmax, ymax):
    """pad image with zeros to reach dimension as (row_max, col_max)
    """
    lx_std=int(xmax/2)
    ly_std=int(ymax/2)
    cx_i = int(np.shape(image)[0]/2)
    cy_i = int(np.shape(image)[1]/2)

    if np.shape(image)[0] > xmax and np.shape(image)[1] > ymax:
        image = image[cx_i-lx_std: cx_i+lx_std,cy_i-ly_std: cy_i+ly_std ]
    elif np.shape(image)[0] > xmax:
        image = image[cx_i-lx_std: cx_i+lx_std, :]
    elif np.shape(image)[1] > ymax:
        image = image[:, cy_i-ly_std: cy_i+ly_std]

    pad_X_0 = np.int(math.ceil((xmax - image.shape[0])/2))
    pad_X_1 = xmax - image.shape[0]-pad_X_0
    pad_Y_0 = np.int(math.ceil((ymax - image.shape[1])/2))
    pad_Y_1 = ymax - image.shape[1]-pad_Y_0

    npad = ((pad_X_0, pad_X_1), (pad_Y_0, pad_Y_1))
   # print(npad)
    #print(image.shape)
    padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)

    #print(np.shape(padded))
    return padded

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

def Preprocee_2D(imgs_2D_Z,ref_2D_Z,margin):
    ref_2D_Z[ref_2D_Z < 0.5] = 0
    arr = np.nonzero(ref_2D_Z)
    minA = 0 if not len(arr[0]) else min(arr[0])
    maxA = 0 if not len(arr[0]) else max(arr[0])
    minB = 0 if not len(arr[1]) else min(arr[1])
    maxB = 0 if not len(arr[1]) else max(arr[1])
    #print(minA)

    width = imgs_2D_Z.shape[0]
    height = imgs_2D_Z.shape[1]
    # with margin
    cropped_im = imgs_2D_Z[max(minA - margin, 0): min(maxA + margin + 1, width), \
                 max(minB - margin, 0): min(maxB + margin + 1, height)]

    cropped_ref= ref_2D_Z[max(minA - margin, 0): min(maxA + margin + 1, width), \
                   max(minB - margin, 0): min(maxB + margin + 1, height)]

    return cropped_im,cropped_ref

def cut_with_ref_and_pading(imgs,refs,row_std):
    total_Z = imgs.shape[0]
    new_imgs = np.zeros([len(imgs),row_std,row_std])
    new_refs = np.zeros([len(imgs),row_std,row_std])
    margin = int(imgs.shape[1]*0.1)

    for j in range(total_Z):
        imgs_2D_Z = imgs[j, :, :]
        ref_2D_Z = refs[j, :, :]
        #print(np.shape(ref_2D_Z))
        cropped_im, cropped_ref = Preprocee_2D(imgs_2D_Z, ref_2D_Z, margin)
        #print(np.shape(pad_2D(cropped_im, 0, row_std, row_std)))

        new_imgs[j,::]=pad_2D(cropped_im, 0, row_std, row_std)
        new_refs[j,::]=pad_2D(cropped_ref, 0, row_std, row_std)

    #print(np.max(new_refs))

    return new_imgs,new_refs

def imgs_adapthist(imgs):
    new_imgs = np.zeros([len(imgs),imgs.shape[1],imgs.shape[2]])
    for mm, img in enumerate(imgs):
        img = equalize_adapthist( img, clip_limit=0.05 )
        new_imgs[mm] = img

    return new_imgs

def Find_rangeZ(GS_3D):
    max_Z = np.zeros([GS_3D.shape[0]])
    for i in range(GS_3D.shape[0]):
        max_Z[i]=np.max(GS_3D[i,:,:])
        #print(max_Z[i])
    arr = np.nonzero(max_Z)
    start_idx = min(arr[0])
    end_idx = max(arr[0])+1
    return start_idx, end_idx

def itk_to_array(image_filename):
    image_itk = sitk.ReadImage(image_filename)
    image_array = sitk.GetArrayFromImage(image_itk)
    return image_array

def Label_connection(prob_):
    area_list = []
    binary = np.zeros(prob_.shape,dtype=np.uint8)
    binary[prob_>0.5]=1
    #binary = prob_ > 0.5
    label_prob_ = measure.label(binary)
    region_label_prob_ = measure.regionprops(label_prob_)
    for region in region_label_prob_:
        area_list.append(region.area)
    idx_max = np.argmax(area_list)
    binary[label_prob_ != idx_max + 1] = 0
    temp_prob_ = binary
    #temp_prob_.astype(np.uint8)
    #plt.figure('Orginal Prob'), plt.imshow(temp_prob_, cmap='gray')
    #plt.show()
    return temp_prob_
'''
def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
'''

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

#focal loss
def focal_loss(y_true, y_pred, gamma=2.0):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    loss = -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred), axis=-1)
    return loss

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )


def rel_abs_vol_diff(y_true, y_pred):

    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)


def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)



def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)


    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds


def Seperate_two_Kidey(array_gs):
    mask1=np.zeros(np.shape(array_gs))
    mask2=np.zeros(np.shape(array_gs))
    binary =array_gs> 0.5
    area_list = []

    mask012 = measure.label(binary)
    region_mask = measure.regionprops(mask012)
    kideney_num=0
    for region in region_mask:
        area_list.append(region.area)
        if region.area>200:
            kideney_num+=1
    #print(kideney_num)
    if kideney_num>1:
        ############################################################ 2 kidey
        idx_1 = np.argmax(area_list) 
        area_list[idx_1]=0
        idx_2 = np.argmax(area_list)

        region_1=region_mask[idx_1] # the largeset region
        region_2=region_mask[idx_2] # the second largeset region
        ##
        if region_1.centroid[0]<region_2.centroid[0]:
            mask1[mask012 ==idx_1+1]=1
            mask2[mask012 ==idx_2+1]=1
        else:
            mask1[mask012 ==idx_2+1]=1
            mask2[mask012 ==idx_1+1]=1
        ############################################################ Only 1 kidey 
    else:
        idx_1 = np.argmax(area_list) 
        region_1=region_mask[idx_1]
        mask1[mask012 ==idx_1+1]=1
        mask2 = mask1
    return mask1,mask2,kideney_num

def Range_from_mask(mask):
    #print(np.shape(mask))
    arr=np.nonzero(mask)
    margin_xy=5
    margin_z=3
    Min_z=int(min(arr[0])-margin_z)
    Max_z=int(max(arr[0])+margin_z)

    Min_x=int(min(arr[1])-margin_xy)
    Max_x=int(max(arr[1])+margin_xy)
    Min_y=int(min(arr[2])-margin_xy)
    Max_y=int(max(arr[2])+margin_xy)
    
    Min_z=max(0,Min_z)
    Max_z=min(np.shape(mask)[0],Max_z)

    return Min_z,Max_z,Min_x,Max_x,Min_y,Max_y