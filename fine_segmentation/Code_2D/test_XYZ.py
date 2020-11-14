"""
This code is to test NN model and visualize output
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import scipy.io as scio
from skimage.morphology import convex_hull_image
import xlwt

from utils import *

K.set_image_data_format('channels_last')  # Tensorflow dimension ordering
root_path  = sys.argv[1]
code_path = sys.argv[1] +'Code_2D/'
SEG3D_path= sys.argv[1] + 'Code_3D/src_3DUnet/outputFiles/'

data_path = sys.argv[2]
model_path = code_path + "models/"

# dir for storing results that contains
results_path_SEG3D = os.path.join(data_path, 'Test/SEG3D/')
if not os.path.exists(results_path_SEG3D):
    os.makedirs(results_path_SEG3D)

results_path_SEG3D_LC = os.path.join(data_path, 'Test/SEG3D_LC/')
if not os.path.exists(results_path_SEG3D_LC):
    os.makedirs(results_path_SEG3D_LC)

results_path_SEGTrans= os.path.join(data_path, 'Test/SEGTrans/')
if not os.path.exists(results_path_SEGTrans):
    os.makedirs(results_path_SEGTrans)

results_path_SEGCoron= os.path.join(data_path, 'Test/SEGCoron/')
if not os.path.exists(results_path_SEGCoron):
    os.makedirs(results_path_SEGCoron)

results_path_SEGSagit= os.path.join(data_path, 'Test/SEGSagit/')
if not os.path.exists(results_path_SEGSagit):
    os.makedirs(results_path_SEGSagit)

results_path_SEG25D= os.path.join(data_path, 'Test/SEG25D/')
if not os.path.exists(results_path_SEG25D):
    os.makedirs(results_path_SEG25D)

cur_fold = sys.argv[3]
ZMAX = int(sys.argv[4])
YMAX = int(sys.argv[5])
XMAX = int(sys.argv[6])
high_range = float(sys.argv[7])
low_range = float(sys.argv[8])
margin = int(sys.argv[9])

## Fixed parameters in 2D model
model_name_Trans = sys.argv[10]+'_' + 'Trans' +'FD_' +cur_fold
model_name_Coron = sys.argv[10]+'_' + 'Coron' +'FD_' +cur_fold
model_name_Sagit = sys.argv[10]+'_' + 'Sagit' +'FD_' +cur_fold

Normlize_VS=[1,1,1]

# prediction of trained model

"""
Dice Ceofficient and Cost functions for training
"""
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return  -dice_coef(y_true, y_pred)

def test():
    model_Trans = load_model(model_path + model_name_Trans + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model_Sagit = load_model(model_path + model_name_Sagit + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model_Coron = load_model(model_path + model_name_Coron + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    row_num = 0

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    sheet.write(row_num, 0, 'CaseName')
    sheet.write(row_num, 1, 'SEG3D')
    sheet.write(row_num, 2, 'SEG3D LC')
    sheet.write(row_num, 3, 'Coron')
    sheet.write(row_num, 4, 'Sagit')
    sheet.write(row_num, 5, 'Axial')
    sheet.write(row_num, 6, '25D')
    sheet.write(row_num, 7, 'LC')
    MAT_content = scio.loadmat(root_path + 'Crop_Range_ISICDM.mat')
    Crop_Range = MAT_content['Crop_range']
    # print(np.shape(Crop_Range))

    num_patient=16
    #num_patient = 30
    # print(num_patient)
    # iterate all test cases
    DSC1 = []
    DSC2 = []
    DSC3 = []
    DSC4 = []
    DSC5 = []
    DSC6 = []
    DSC7 = []

    for l in range(num_patient):

        print('*********************************************************************')
        idx = str(l + 1).zfill(2)
        print(l, idx)
        filename = idx + '.nii.gz'
        dir_img = data_path + 'Test/Img/' + idx + '.nii.gz'
        dir_gs = data_path + 'Test/GroundTruth/' + idx + '_seg.nii.gz'
        dir_segg3d = SEG3D_path + 'results_FD' + str(cur_fold) + '/Pred/Testing/ProbMap_class_1_' + idx + '.nii'

        data_img = sitk.ReadImage(dir_img)
        IMG_ori = sitk.GetArrayFromImage(data_img)

        data_seg3d = sitk.ReadImage(dir_segg3d)
        SEG3D_crop = sitk.GetArrayFromImage(data_seg3d)
        Spacing_img = data_img.GetSpacing()[::-1]

        IMG_array = ThreeD_resize(IMG_ori, Normlize_VS, Spacing_img)
        PROB_array = np.zeros(IMG_array.shape)

        #print(np.shape(SEG3D_crop))
        Min_z = Crop_Range[l,0]
        Min_x = Crop_Range[l,1]
        Min_y = Crop_Range[l,2]
        Max_z = Crop_Range[l,3]
        Max_x = Crop_Range[l,4]
        Max_y = Crop_Range[l,5]

        PROB_array[Min_z: Max_z, Min_x: Max_x, Min_y:Max_y]=SEG3D_crop
        Mask=Label_connection(PROB_array,0.1)
        #PROB_array = PROB_array * Mask

        Min_z, Max_z, Min_y, Max_y, Min_x, Max_x = GetBoundingBox(Mask, margin, ZMAX, YMAX, XMAX)
        IMG_crop_padding = np.zeros([ZMAX, YMAX, XMAX]) + low_range
        PROB_crop_padding = np.zeros([ZMAX, YMAX, XMAX])
        left_z = int(ZMAX / 2 - (Max_z - Min_z) / 2)
        left_y = int(YMAX / 2 - (Max_y - Min_y) / 2)
        left_x = int(XMAX / 2 - (Max_x - Min_x) / 2)

        IMG_array[IMG_array > high_range] = high_range
        IMG_array[IMG_array < low_range] = low_range
        #print(Min_z, Max_z, Min_x, Max_x, Min_y, Max_y)
        IMG_crop_padding[left_z:left_z + Max_z - Min_z, left_y:left_y + Max_y - Min_y,
        left_x:left_x + Max_x - Min_x] = IMG_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]

        PROB_crop_padding[left_z:left_z + Max_z - Min_z, left_y:left_y + Max_y - Min_y,
        left_x:left_x + Max_x - Min_x] = PROB_array[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]

        M = np.mean(IMG_array[Mask> 0.5])
        STD = np.mean(IMG_array[Mask > 0.5])
        IMG_crop_padding = (IMG_crop_padding - M) / STD

        ## Trans
        Crop_image = IMG_crop_padding[left_z:left_z + Max_z - Min_z, :, :]
        Crop_prob = PROB_crop_padding[left_z:left_z + Max_z - Min_z, :, :]
        input_Trans = np.concatenate([preprocess(Crop_image),preprocess(Crop_prob)],axis=3)
        #input_Trans = preprocess(Crop_image)
        out_Trans = model_Trans.predict(input_Trans,verbose=1, batch_size=32)
        SEG_Trans = np.zeros(IMG_array.shape)
        SEG_Trans[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x]=out_Trans[:, left_y:left_y + Max_y - Min_y,
        left_x:left_x + Max_x - Min_x,0]

        ## Coron
        Crop_image = IMG_crop_padding[:, left_y:left_y + Max_y - Min_y, :]
        Crop_prob = PROB_crop_padding[:, left_y:left_y + Max_y - Min_y, :]
        Crop_image = np.transpose(Crop_image, [1, 0, 2])
        Crop_prob = np.transpose(Crop_prob, [1, 0, 2])
        input_Coron = np.concatenate([preprocess(Crop_image), preprocess(Crop_prob)], axis=3)
        # input_Trans = preprocess(Crop_image)
        out_Coron = model_Coron.predict(input_Coron , verbose=1, batch_size=32)
        out_Coron = np.transpose(out_Coron,[1,0,2,3])
        SEG_Coron = np.zeros(IMG_array.shape)
        SEG_Coron[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x] = out_Coron[left_z:left_z + Max_z - Min_z, :,
                                                           left_x:left_x + Max_x - Min_x, 0]

        ## Sagit
        Crop_image = IMG_crop_padding[:, :, left_x:left_x + Max_x - Min_x]
        Crop_prob = PROB_crop_padding[:, :, left_x:left_x + Max_x - Min_x]
        Crop_image = np.transpose(Crop_image, [2, 1, 0])
        Crop_prob = np.transpose(Crop_prob, [2, 1, 0])
        input_Sagit = np.concatenate([preprocess(Crop_image), preprocess(Crop_prob)], axis=3)
        out_Sagit = model_Sagit.predict(input_Sagit , verbose=1, batch_size=32)
        out_Sagit = np.transpose(out_Sagit,[2,1,0,3])
        SEG_Sagit = np.zeros(IMG_array.shape)
        SEG_Sagit[Min_z:Max_z, Min_y:Max_y, Min_x:Max_x] = out_Sagit[left_z:left_z + Max_z - Min_z,  left_y:left_y + Max_y - Min_y,
                                                           :, 0]

       #SEG_Trans=Label_connection(SEG_Trans)
        ## Save Results
        SEG3D_ori = Fixed_shape_3D_resize(PROB_array, IMG_ori.shape)
        SEG3D_ori_LC = Fixed_shape_3D_resize(np.float32(Mask), IMG_ori.shape)
        SEG_Trans_ori = Fixed_shape_3D_resize(SEG_Trans, IMG_ori.shape)
        SEG_Coron_ori = Fixed_shape_3D_resize(SEG_Coron, IMG_ori.shape)
        SEG_Sagit_ori = Fixed_shape_3D_resize(SEG_Sagit, IMG_ori.shape)

        SEG_25D_ori =np.zeros(SEG3D_ori.shape)
        SEG_25D_ori[SEG_Trans_ori+SEG_Coron_ori+SEG_Sagit_ori>1.5]=1

        # print(PROB_array.shape, IMG_ori.shape, SEG3D_ori.shape)

        Save_AS_NII(data_img, SEG3D_ori, results_path_SEG3D, filename)
        Save_AS_NII(data_img, SEG3D_ori_LC, results_path_SEG3D_LC, filename)
        Save_AS_NII(data_img, SEG_Trans_ori, results_path_SEGTrans, filename)
        Save_AS_NII(data_img, SEG_Coron_ori, results_path_SEGCoron, filename)
        Save_AS_NII(data_img, SEG_Sagit_ori, results_path_SEGSagit, filename)
        Save_AS_NII(data_img, SEG_25D_ori, results_path_SEG25D, filename)

        ##Quan
        data_img = sitk.ReadImage(dir_gs)
        GS_ori = sitk.GetArrayFromImage(data_img)
        GS_ori[GS_ori > 0.5] = 1

        dsc_SEG3D, _, _, _ = DSC_computation(GS_ori, SEG3D_ori)
        dsc_SEG3D_LC, _, _, _ = DSC_computation(GS_ori, SEG3D_ori_LC)
        dsc_SEGTrans, _, _, _ = DSC_computation(GS_ori, SEG_Trans_ori)
        dsc_SEGCoron, _, _, _ = DSC_computation(GS_ori, SEG_Coron_ori)
        dsc_SEGSagit, _, _, _ = DSC_computation(GS_ori, SEG_Sagit_ori)
        dsc_SEG25D, _, _, _ = DSC_computation(GS_ori, SEG_25D_ori)
        dsc_SEG25D_LC, _, _, _ = DSC_computation(GS_ori, Label_connection(SEG_25D_ori,0.5))

        print dsc_SEG3D,dsc_SEG3D_LC,dsc_SEGTrans,dsc_SEG25D_LC
        DSC1.append(dsc_SEG3D)
        DSC2.append(dsc_SEG3D_LC)
        DSC3.append(dsc_SEGCoron)
        DSC4.append(dsc_SEGSagit)
        DSC5.append(dsc_SEGTrans)
        DSC6.append(dsc_SEG25D)
        DSC7.append(dsc_SEG25D_LC)

        row_num += 1
        sheet.write(row_num, 0, idx)
        sheet.write(row_num, 1, dsc_SEG3D)
        sheet.write(row_num, 2, dsc_SEG3D_LC)
        sheet.write(row_num, 3, dsc_SEGCoron)
        sheet.write(row_num, 4, dsc_SEGSagit)
        sheet.write(row_num, 5, dsc_SEGTrans)
        sheet.write(row_num, 6, dsc_SEG25D)
        sheet.write(row_num, 7, dsc_SEG25D_LC)

    DSC1 = np.array(DSC1)
    DSC2 = np.array(DSC2)
    DSC3 = np.array(DSC3)
    DSC4 = np.array(DSC4)
    DSC5 = np.array(DSC5)
    DSC6 = np.array(DSC6)
    DSC7 = np.array(DSC7)


    print "---------------------------------"

    print('Mean volumetric DSC:', DSC1.mean(), DSC2.mean(),DSC7.mean())

    row_num += 1
    sheet.write(row_num, 0, 'Average')
    sheet.write(row_num, 1, DSC1.mean())
    sheet.write(row_num, 2, DSC2.mean())
    sheet.write(row_num, 3, DSC3.mean())
    sheet.write(row_num, 4, DSC4.mean())
    sheet.write(row_num, 5, DSC5.mean())
    sheet.write(row_num, 6, DSC6.mean())
    sheet.write(row_num, 7, DSC7.mean())

    # record test dsc mean and standard deviation for each fold in the one file
    book.save(code_path+'test-records/'+sys.argv[10]+'_' + '25D' +'FD_' +cur_fold+'.xls')


if __name__ == "__main__":

    start_time = time.time()

    test()

    print "-----------test done, total time used: %s ------------"% (time.time() - start_time)
