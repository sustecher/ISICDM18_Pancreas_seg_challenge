#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function
import numpy as np
import os,sys
import scipy.io as scio
from skimage import measure
from scipy import ndimage as nd
import SimpleITK as sitk
import xlwt

root_path='/home/bisp/dir/Pancreas_ISICDM_2020/Code_3D/'
data_path = '/home/bisp/dir/raw_data/ISICDM_2018/Train/'
save_path='/home/bisp/dir/Pancreas_ISICDM_2020/Code_3D/Train/'

Normlize_VS=[1,1,1]
Normlize_shape=[120,140,200]
LR=int(-100)
HR=int(240)

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
    margin_xy=20
    margin_z=5
    Min_x=int(min(arr[1])-margin_xy)
    Max_x=int(max(arr[1])+margin_xy)
    Min_y=int(min(arr[2])-margin_xy)
    Max_y=int(max(arr[2])+margin_xy)
    Min_z = int(min(arr[0]) - margin_z)
    Max_z = int(max(arr[0]) + margin_z)

    Min_x = max(0, Min_x)
    Max_x = min(np.shape(mask)[1], Max_x)
    Min_y = max(0, Min_y)
    Max_y = min(np.shape(mask)[2], Max_y)
    Min_z = max(0, Min_z)
    Max_z = min(np.shape(mask)[0], Max_z)

    return Min_z,Max_z,Min_x,Max_x,Min_y,Max_y

def Enlarge_crop_shape(Min_xyz,Max_xyz,nor_size,bounding_size):
    center_xyz=int((Min_xyz+Max_xyz)/2)
    Min_xyz=int(center_xyz-nor_size/2)
    Min_xyz=max(0,Min_xyz)

    Max_xyz=int(center_xyz+nor_size/2)
    Max_xyz=min(Max_xyz,bounding_size)
    return Min_xyz,Max_xyz

def Demo_Test():
    num_patient=20
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    row_num=0
    sheet.write(row_num, 0, 'idx')
    sheet.write(row_num, 1, 'Min dim 1')
    sheet.write(row_num, 2, 'Min dim 2')
    sheet.write(row_num, 3, 'Min dim 3')
    sheet.write(row_num, 4, 'Len dim 1')
    sheet.write(row_num, 5, 'Len dim 2')
    sheet.write(row_num, 6, 'Len dim 3')
    
    for l in range(num_patient):
        filename=str(l+1).zfill(2)
        idx=filename
        print('Pre Processng for case:',l,filename)
        dir_img = data_path +'Img/'+filename +'.nii.gz'
        dir_gs = data_path + 'GroundTruth/'+filename +'_seg.nii.gz'

        data_img = sitk.ReadImage(dir_img)
        array_img = sitk.GetArrayFromImage(data_img)
        print(np.max(array_img))

        data_Pancreas = sitk.ReadImage(dir_gs)
        array_Pancreas = sitk.GetArrayFromImage(data_Pancreas)
        array_Pancreas[array_Pancreas>0.5]=1

        Spacing_img = data_img.GetSpacing()[::-1]
        #print(array_Pancreas.shape)

        Img_array = ThreeD_resize(array_img, Normlize_VS, Spacing_img)
        Mask_array = ThreeD_resize(array_Pancreas, Normlize_VS, Spacing_img)
        #print(Mask_array.shape)
        Min_z, Max_z, Min_x, Max_x, Min_y, Max_y=Range_from_mask(Mask_array)
        #print(Min_z, Max_z, Min_x, Max_x, Min_y, Max_y)
        if Max_z -Min_z<Normlize_shape[0]:      
            Min_z, Max_z=Enlarge_crop_shape(Min_z, Max_z,Normlize_shape[0],Mask_array.shape[0])
        if Max_x -Min_x<Normlize_shape[1]:      
            Min_x, Max_x=Enlarge_crop_shape(Min_x, Max_x,Normlize_shape[1],Mask_array.shape[1])
        if Max_y -Min_y<Normlize_shape[2]:      
            Min_y, Max_y=Enlarge_crop_shape(Min_y, Max_y,Normlize_shape[2],Mask_array.shape[2])
        
        Crop_Img_array = Img_array[Min_z:Max_z, Min_x:Max_x, Min_y:Max_y]
        Crop_GS_array = Mask_array[Min_z:Max_z, Min_x:Max_x, Min_y:Max_y]
        row_num+=1
        sheet.write(row_num, 0, idx)
        sheet.write(row_num, 1, Min_z)
        sheet.write(row_num, 2, Min_x)
        sheet.write(row_num, 3, Min_y)
        sheet.write(row_num, 4, Max_z-Min_z)
        sheet.write(row_num, 5, Max_x-Min_x)
        sheet.write(row_num, 6, Max_y-Min_y)
        
        #print(Crop_GS_array.shape)
        #save_path_3D_images =save_path
        #save_path_3D_labels =save_path
        save_path_3D_images = save_path+'Images/'
        if not os.path.exists(save_path_3D_images):
            os.makedirs(save_path_3D_images)
        save_path_3D_labels = save_path+'Labels/'
        if not os.path.exists(save_path_3D_labels):
            os.makedirs(save_path_3D_labels)

        Crop_Img_array[Crop_Img_array<LR]=LR
        Crop_Img_array[Crop_Img_array>HR] = HR
        Crop_Img_array=(Crop_Img_array-LR)/float(HR-LR)

        IMG_ni = sitk.GetImageFromArray(Crop_Img_array)
        #IMG_ni.SetOrigin(data_img.GetOrigin())
        #IMG_ni.SetDirection(data_img.GetDirection())
        sitk.WriteImage(IMG_ni,save_path_3D_images+ idx + '.nii')
        SEG_ni = sitk.GetImageFromArray(Crop_GS_array)
        SEG_ni.SetOrigin(data_img.GetOrigin())
        SEG_ni.SetDirection(data_img.GetDirection())
        sitk.WriteImage(SEG_ni,save_path_3D_labels + idx + '_SEG.nii')
    book.save(save_path+'CropParameters.xls')

if __name__=='__main__':

    import time

    start = time.time()

    Demo_Test()
    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2 ) )
