"""
This code is to build and train 2D U-Net
"""
import numpy as np
import sys
import subprocess
import argparse
import os

from keras.models import Model
from keras.layers import Lambda, concatenate, Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation, Dropout, Add, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras import losses
from keras.layers.core import Layer
from keras.layers import DepthwiseConv2D, Dropout, UpSampling2D,Lambda,BatchNormalization, DepthwiseConv2D,AveragePooling2D, Concatenate, GlobalAveragePooling2D, Dense
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.layers.merge import add,multiply
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import csv

from utils import *

K.set_image_data_format('channels_last')  # Tensorflow dimension ordering

# ----- paths setting -----
code_path = sys.argv[1] +'Code_2D/'
model_path =code_path + "models/"
log_path = code_path  + "logs/"


# ----- params for training and testing -----
batch_size = 8
cur_fold = sys.argv[2]
plane = sys.argv[3]
epoch = int(sys.argv[4])
init_lr = float(sys.argv[5])
model_name = sys.argv[6] +'_' + plane +'FD_' +cur_fold

# ----- Dice Coefficient and cost function for training -----
smooth = 1.

def load_train_data(current_fold, plane):
    imgs_train = np.load(code_path+'NPY_data/%s_X_FD%s.npy'%(plane,current_fold))
    probs_train = np.load(code_path+'NPY_data/%s_P_FD%s.npy' % (plane,current_fold))
    labels_train = np.load(code_path+'NPY_data/%s_y_FD%s.npy'%(plane,current_fold))
    return imgs_train, probs_train,labels_train

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return  -dice_coef(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
	conv = Conv2D(nd, k, padding='same', kernel_initializer='he_normal')(inputs)  # , kernel_initializer='he_normal'
	bn = BatchNormalization()(conv)
	relu = Activation('relu')(bn)
	return relu

def get_unet((img_rows, img_cols), flt=16, pool_size=(2, 2, 2), init_lr=1.0e-5):
    """build and compile Neural Network"""

    print "start building NN"
    inputs = Input((img_rows, img_cols,2))

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

    up6 = concatenate([Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt*8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)

    up7 = concatenate([Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt*4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)

    up8 = concatenate([Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt*2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=init_lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def train(fold, plane, batch_size, nb_epoch,init_lr):
    """
    train an Unet model with data from load_train_data()

    Parameters
    ----------
    fold : string
        which fold is experimenting in 4-fold. It should be one of 0/1/2/3

    plane : char
        which plane is experimenting. It is from 'X'/'Y'/'Z'

    batch_size : int
        size of mini-batch

    nb_epoch : int
        number of epochs to train NN

    init_lr : float
        initial learning rate
    """

    print "number of epoch: ", nb_epoch
    print "learning rate: ", init_lr

    # --------------------- load and preprocess training data -----------------
    print '-'*80
    print '         Loading and preprocessing train data...'
    print '-'*80

    imgs_train, probs_train, labels_train = load_train_data(fold, plane)
    print(imgs_train.shape)
    print(probs_train.shape)
    print(labels_train.shape)

    #imgs_train = smooth_images(imgs_train)

    imgs_row = imgs_train.shape[1]
    imgs_col = imgs_train.shape[2]

    #imgs_train = preprocess(imgs_train)
    #probs_train = preprocess(probs_train)
    #labels_train = preprocess(labels_train)
    imgs_train = np.concatenate([imgs_train,probs_train],axis=3)

    #imgs_train = imgs_train.astype('float32')
    #imgs_mask_train = imgs_mask_train.astype('float32')

    # ---------------------- Create, compile, and train model ------------------------
    print '-'*80
    print '		Creating and compiling model...'
    print '-'*80
    model = get_unet((imgs_row, imgs_col), pool_size=(2, 2, 2), init_lr=init_lr)
    #print model.summary()

    print '-'*80
    print '		Fitting model...'
    print '-'*80

    val_factor = 1
    if val_factor == 1:
        print(np.shape(imgs_train))
        sample_num =int(imgs_train.shape[0])
        list_all = set(range(sample_num))
        train_list = random.sample(list_all, int(sample_num * 0.8))
        val_list = list(list_all - set(train_list))
        X_train = imgs_train[train_list, ...]
        y_train = labels_train[train_list, ...]
        X_val = imgs_train[val_list, ...]
        y_val = labels_train[val_list, ...]
        ver = 'unet_fd%s_%s_ep%s_lr%s.csv' % (cur_fold, plane, epoch, init_lr)
        csv_logger = CSVLogger(log_path + ver)
        model_checkpoint = ModelCheckpoint(model_path + model_name + ".h5", monitor='val_loss', save_best_only=True)
        history = model.fit(X_train, y_train,
                            #validation_data=(np.concatenate([X_train,X_val]), np.concatenate([y_train,y_val])),
                            validation_data=(X_val, y_val),
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=1, shuffle=True,
                            callbacks=[model_checkpoint, csv_logger])

    else:
        ver = 'unet_fd%s_%s_ep%s_lr%s.csv' % (cur_fold, plane, epoch, init_lr)
        csv_logger = CSVLogger(log_path + ver)
        model_checkpoint = ModelCheckpoint(model_path + model_name + ".h5",
                                           monitor='loss',
                                           save_best_only=False,
                                           period=10)

        history = model.fit(imgs_train, imgs_mask_train,
                            batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True,
                            callbacks=[model_checkpoint, csv_logger])


if __name__ == "__main__":

    train(cur_fold, plane, batch_size, epoch, init_lr)

    print "training done"
