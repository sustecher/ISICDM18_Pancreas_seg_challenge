from keras.models import Model
from keras.initializers import RandomNormal, VarianceScaling
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D, BatchNormalization
import os
import time
import scipy
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras import metrics
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation, Dropout, Add
from keras.layers import DepthwiseConv2D
#SE
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
#
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
from scipy.misc import imresize
import warnings
from keras.initializers import RandomNormal, VarianceScaling
import numpy as np
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
import numpy as np

smooth=1.
def dice_coef_for_training(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1. - dice_coef_for_training(y_true, y_pred)


def conv_bn_relu(nd, k=3, inputs=None):
	conv = Conv2D(nd, k, padding='same', kernel_initializer='he_normal')(inputs)  # , kernel_initializer='he_normal'
	#conv = spatial_se(conv)
	bn = BatchNormalization()(conv)
	relu = Activation('relu')(bn)
	#relu= channel_se(relu)
	return relu


def conv_relu(nd, k=3, inputs=None):
	conv = Conv2D(nd, k, padding='same', kernel_initializer='he_normal')(inputs)  # , kernel_initializer='he_normal'
	# bn = BatchNormalization()(conv)
	relu = Activation('relu')(conv)
	return relu

def Z_Block(nd1, nd2, input=None):
	conv2 = conv_bn_relu(nd2, 3, input)

	pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = conv_bn_relu(nd1, 3, pool1)

	conv4 = conv_bn_relu(nd2, 3, conv3)

	# ch, cw = get_crop_shape(conv2, conv4)
	# crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
	# concat1 = concatenate([crop_conv2, conv4], axis=-1)
	ch, cw = get_crop_shape(pool1, conv4)
	crop_pool1 = Cropping2D(cropping=(ch, cw))(pool1)
	concat1 = concatenate([crop_pool1, conv4], axis=-1)

	output = concat1
	return conv2, output


def Decoder_Z_Block(nd1, nd2, input_left=None, input_down=None):
	conv2 = conv_bn_relu(nd1, 3, input_down)
	up_conv2 = UpSampling2D(size=(2, 2))(conv2)
	conv3 = conv_bn_relu(nd2, 3, up_conv2)

	ch, cw = get_crop_shape(input_left, conv3)
	crop_input_left = Cropping2D(cropping=(ch, cw))(input_left)
	concat1 = concatenate([crop_input_left, conv3], axis=-1)

	concat2 = concatenate([up_conv2, concat1], axis=-1)

	conv4 = conv_bn_relu(nd2, 3, concat2)
	return conv4


def get_crop_shape(target, refer):
	# width, the 3rd dimension
	cw = (target.get_shape()[2] - refer.get_shape()[2]).value
	assert (cw >= 0)
	if cw % 2 != 0:
		cw1, cw2 = int(cw / 2), int(cw / 2) + 1
	else:
		cw1, cw2 = int(cw / 2), int(cw / 2)
	# height, the 2nd dimension
	ch = (target.get_shape()[1] - refer.get_shape()[1]).value
	assert (ch >= 0)
	if ch % 2 != 0:
		ch1, ch2 = int(ch / 2), int(ch / 2) + 1
	else:
		ch1, ch2 = int(ch / 2), int(ch / 2)

	return (ch1, ch2), (cw1, cw2)

'''
def ASPP(input):
	b1 = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(6, 6), padding='same', use_bias=False)(input)
	b2 = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(12, 12), padding='same', use_bias=False)(input)
	b3 = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(18, 18), padding='same', use_bias=False)(input)
	x = concatenate([input, b1, b2, b3])
	x = BatchNormalization(epsilon=1e-5)(x)
	x = Activation('relu')(x)
	return x
## 8690
def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	#conv4 = conv_bn_relu(256, 3, pool3)
	conv5 = ASPP(pool4)
	conv5 = conv_bn_relu(1024, 3, conv5)


	up1_C5 = UpSampling2D(size=(2, 2))(conv5)
	up1 = concatenate([up1_C5, conv4], axis=concat_axis)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(512, 3, conv6)


	up2_C6 = UpSampling2D(size=(2, 2))(conv6)
	up2= concatenate([up2_C6, conv3], axis=concat_axis)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(256, 3, conv7)


	up3_C7 = UpSampling2D(size=(2, 2))(conv7)
	up3 = concatenate([up3_C7, conv2], axis=concat_axis)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(128, 3, conv8)


	up3_C8 = UpSampling2D(size=(2, 2))(conv8)
	up4 = concatenate([up3_C8, conv1], axis=concat_axis)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9= conv_bn_relu(64, 3, conv9)


	conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'

	model = Model(inputs=inputs, outputs=conv10)
	model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model
'''

def channel_se(input, ratio=16):
	''' Create a squeeze-excite block
	Args:
		input: input tensor
		filters: number of output filters
		k: width factor
	Returns: a keras tensor
	'''
	init = input
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	filters = init._keras_shape[channel_axis]
	se_shape = (1, 1, filters)

	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

	if K.image_data_format() == 'channels_first':
		se = Permute((3, 1, 2))(se)

	x = multiply([init, se])
	return x

def spatial_se(input):
	init = input
	se = Conv2D(1, 1, activation='sigmoid', padding='same')(init)
	x = multiply([init, se])
	return x

''' 865
def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	A = ASPP(pool4)
	cSE = channel_se(pool4)
	sSE = spatial_se(pool4)
	scSE = Add()([cSE, sSE])

	ASP_OC=concatenate([A,scSE])
	conv5 = conv_bn_relu(512, 3, ASP_OC)


	up1_C5 = UpSampling2D(size=(2, 2))(conv5)
	up1 = concatenate([up1_C5, conv4], axis=concat_axis)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(512, 3, conv6)


	up2_C6 = UpSampling2D(size=(2, 2))(conv6)
	up2= concatenate([up2_C6, conv3], axis=concat_axis)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(256, 3, conv7)


	up3_C7 = UpSampling2D(size=(2, 2))(conv7)
	up3 = concatenate([up3_C7, conv2], axis=concat_axis)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(128, 3, conv8)


	up3_C8 = UpSampling2D(size=(2, 2))(conv8)
	up4 = concatenate([up3_C8, conv1], axis=concat_axis)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9= conv_bn_relu(64, 3, conv9)


	conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'

	model = Model(inputs=inputs, outputs=conv10)
	model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model
'''
''' 867
def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#	A = ASPP(pool4)
#	cSE = channel_se(pool4)
#	sSE = spatial_se(pool4)
#	scSE = Add()([cSE, sSE])

#	ASP_OC=concatenate([A,scSE])
#	conv5 = conv_bn_relu(512, 3, ASP_OC)

	conv5 = conv_bn_relu(1024, 3, pool4)
	conv5 = conv_bn_relu(1024, 3, conv5)

	up1_C5 = UpSampling2D(size=(2, 2))(conv5)
	up1 = concatenate([up1_C5, conv4], axis=concat_axis)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(512, 3, conv6)


	up2_C6 = UpSampling2D(size=(2, 2))(conv6)
	up2= concatenate([up2_C6, conv3], axis=concat_axis)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(256, 3, conv7)


	up3_C7 = UpSampling2D(size=(2, 2))(conv7)
	up3 = concatenate([up3_C7, conv2], axis=concat_axis)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(128, 3, conv8)


	up3_C8 = UpSampling2D(size=(2, 2))(conv8)
	up4 = concatenate([up3_C8, conv1], axis=concat_axis)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9= conv_bn_relu(64, 3, conv9)


	conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'

	model = Model(inputs=inputs, outputs=conv10)
	model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model
'''

def asp_block(features,x, atrous_rates=(6, 12, 18)):
	# OC
	#b1 = channel_se(x)
	#b1_SE = spatial_se(x)
	#b1 = Add()([b1_CE, b1_SE])
	#b1=b1_SE
	#b1 = GlobalAveragePooling2D()(x)
	# 1x1 conv
	'''
	b2 = Conv2D(features, (1, 1), padding='same', dilation_rate=1, use_bias=False, name='aspp_2')(x)
	b2 = BatchNormalization(name='aspp2_BN')(b2)
	b2 = Activation('relu', name='aspp2_activation')(b2)

	# 3x3 conv rate = 6
	b3 = Conv2D(features, (3, 3), padding='same', dilation_rate=atrous_rates[0], use_bias=False, name='aspp_3')(x)
	b3 = BatchNormalization(name='aspp3_BN')(b3)
	b3 = Activation('relu', name='aspp3_activation')(b3)

	# 3x3 conv rate = 12 (24)
	b4 = Conv2D(features, (3, 3), padding='same', dilation_rate=atrous_rates[1], use_bias=False, name='aspp_4')(x)
	b4 = BatchNormalization(name='aspp4_BN')(b4)
	b4 = Activation('relu', name='aspp4_activation')(b4)

	# 3x3 conv rate = 18 (36)
	b5 = Conv2D(features, (3, 3), padding='same', dilation_rate=atrous_rates[2], use_bias=False, name='aspp_5')(x)
	b5 = BatchNormalization(name='aspp5_BN')(b5)
	b5 = Activation('relu', name='aspp5_activation')(b5)
	'''
	b2 = Conv2D(features, (1, 1), padding='same', dilation_rate=1, use_bias=False)(x)
	b2 = BatchNormalization()(b2)
	b2 = Activation('relu')(b2)

	# 3x3 conv rate = 6
	b3 = Conv2D(features, (3, 3), padding='same', dilation_rate=atrous_rates[0], use_bias=False)(x)
	b3 = BatchNormalization()(b3)
	b3 = Activation('relu')(b3)

	# 3x3 conv rate = 12 (24)
	b4 = Conv2D(features, (3, 3), padding='same', dilation_rate=atrous_rates[1], use_bias=False)(x)
	b4 = BatchNormalization()(b4)
	b4 = Activation('relu')(b4)

	# 3x3 conv rate = 18 (36)
	b5 = Conv2D(features, (3, 3), padding='same', dilation_rate=atrous_rates[2], use_bias=False)(x)
	b5 = BatchNormalization()(b5)
	b5 = Activation('relu')(b5)
	x = concatenate([b2, b3, b4, b5])

	#x = conv_bn_droupout(features,x)
	return x


def conv_bn_droupout(features,x):
	# reduce channels from 5N to N
	x = Conv2D(features, (1, 1), padding='same', use_bias=False, name='concat_projection',kernel_initializer='he_normal')(x)
	x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
	x = Activation('relu')(x)
	x = Dropout(0.1)(x)
	return x


def ASPP(input):
	b1 = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(6, 6), padding='same', use_bias=False)(input)
	b2 = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(12, 12), padding='same', use_bias=False)(input)
	b3 = DepthwiseConv2D((3, 3), strides=(1, 1), dilation_rate=(18, 18), padding='same', use_bias=False)(input)
	x = concatenate([input, b1, b2, b3])
	x = BatchNormalization(epsilon=1e-5)(x)
	x = Activation('relu')(x)
	return x

def sc_se(x):
	b1_CE = channel_se(x)
	b1_SE = spatial_se(x)
	b1 = Add()([b1_CE, b1_SE])
	return b1

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)

	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)

	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)

	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	#pool4 = ASPP(pool4)
	pool4_ASP = asp_block(512, pool4, atrous_rates=(6, 12, 18))
	pool4_SE = channel_se(pool4)
	pool4 = concatenate([pool4_ASP,pool4_SE])

	#pool4=asp_block(512, pool4, atrous_rates=(6, 12, 18))

	conv5 = conv_bn_relu(1024, 3, pool4)
	conv5 = conv_bn_relu(512, 3, conv5)

	up1_C5 = UpSampling2D(size=(2, 2))(conv5)
	up1 = concatenate([up1_C5, conv4], axis=concat_axis)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(256, 3, conv6)

	up2_C6 = UpSampling2D(size=(2, 2))(conv6)
	up2= concatenate([up2_C6, conv3], axis=concat_axis)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(128, 3, conv7)

	up3_C7 = UpSampling2D(size=(2, 2))(conv7)
	up3 = concatenate([up3_C7, conv2], axis=concat_axis)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(64, 3, conv8)

	up3_C8 = UpSampling2D(size=(2, 2))(conv8)
	up4 = concatenate([up3_C8, conv1], axis=concat_axis)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9 = conv_bn_relu(32, 3, conv9)

###############################################0314
	#conv9_SE = channel_se(conv9)
	#conv9_ASP = asp_block(32, conv9, atrous_rates=(6, 12, 18))
	#conv9 = concatenate([conv9_SE,conv9_ASP])

#	conv9 = conv_bn_relu(32, 3, conv9)
	#conv9 = asp_block(1, conv9, atrous_rates=(6, 12, 18))
	#conv9=channel_se(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'

	model = Model(inputs=inputs, outputs=conv10)
	#model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model

def FFFUNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)

	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)

	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)

	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	#pool4 = ASPP(pool4)
	pool4_ASP = asp_block(512, pool4, atrous_rates=(6, 12, 18))
	pool4_SE = channel_se(pool4)
	pool4 = concatenate([pool4_ASP,pool4_SE])

	#pool4=asp_block(512, pool4, atrous_rates=(6, 12, 18))

	conv5 = conv_bn_relu(1024, 3, pool4)
	conv5 = conv_bn_relu(512, 3, conv5)

	up1_C5 = UpSampling2D(size=(2, 2))(conv5)
	up1 = concatenate([up1_C5, conv4], axis=concat_axis)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(256, 3, conv6)

	up2_C6 = UpSampling2D(size=(2, 2))(conv6)
	up2= concatenate([up2_C6, conv3], axis=concat_axis)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(128, 3, conv7)

	up3_C7 = UpSampling2D(size=(2, 2))(conv7)
	up3 = concatenate([up3_C7, conv2], axis=concat_axis)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(64, 3, conv8)

	up3_C8 = UpSampling2D(size=(2, 2))(conv8)
	up4 = concatenate([up3_C8, conv1], axis=concat_axis)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9 = conv_bn_relu(32, 3, conv9)

###############################################0314
	#conv9_SE = channel_se(conv9)
	#conv9_ASP = asp_block(32, conv9, atrous_rates=(6, 12, 18))
	#conv9 = concatenate([conv9_SE,conv9_ASP])

#	conv9 = conv_bn_relu(32, 3, conv9)
	#conv9 = asp_block(1, conv9, atrous_rates=(6, 12, 18))
	#conv9=channel_se(conv9)
	#conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'
	conv10 = Conv2D(2, 1, activation=None, padding='same')(conv9)
	conv10 = Activation('softmax')(conv10)

	model = Model(inputs=inputs, outputs=conv10)
	#model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model

'''
 # Baseline U-net
def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	#conv5 = asp_block(512, pool4, atrous_rates=(6, 12, 18))
	#conv5 = asp_block(512, pool4 , atrous_rates=(6, 12, 18))

	conv5 = conv_bn_relu(1024, 3, pool4)
	conv5 = conv_bn_relu(512, 3, conv5)

	up1_C5 = UpSampling2D(size=(2, 2))(conv5)
	up1 = concatenate([up1_C5, conv4], axis=concat_axis)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(256, 3, conv6)

	up2_C6 = UpSampling2D(size=(2, 2))(conv6)
	up2= concatenate([up2_C6, conv3], axis=concat_axis)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(128, 3, conv7)

	up3_C7 = UpSampling2D(size=(2, 2))(conv7)
	up3 = concatenate([up3_C7, conv2], axis=concat_axis)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(64, 3, conv8)


	up3_C8 = UpSampling2D(size=(2, 2))(conv8)
	up4 = concatenate([up3_C8, conv1], axis=concat_axis)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9 = conv_bn_relu(32, 3, conv9)

	#conv9 = spatial_se(conv9)

	#conv9 = asp_block(1, conv9, atrous_rates=(6, 12, 18))
	conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'
	#
	#conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv10)  # , kernel_initializer='he_normal'

	model = Model(inputs=inputs, outputs=conv10)
	model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model

'''

'''
def UnetGatingSignal(input, is_batchnorm=False):
	shape = K.int_shape(input)
	x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
	if is_batchnorm:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def expend_as(tensor, rep):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
	return my_repeat

def AttnGatingBlock(x, g, inter_shape):
	shape_x = K.int_shape(x)  # 32
	shape_g = K.int_shape(g)  # 16

	theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
	shape_theta_x = K.int_shape(theta_x)

	phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
	upsample_g = Conv2DTranspose(inter_shape, (3, 3),
								 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
								 padding='same')(phi_g)  # 16

	concat_xg = Add()([upsample_g, theta_x])
	act_xg = Activation('relu')(concat_xg)
	psi = Conv2D(1, (1, 1), padding='same')(act_xg)
	sigmoid_xg = Activation('sigmoid')(psi)
	shape_sigmoid = K.int_shape(sigmoid_xg)
	upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

	# my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
	# upsample_psi=my_repeat([upsample_psi])
	upsample_psi = expend_as(upsample_psi, shape_x[3])

	y = multiply([upsample_psi, x])

	# print(K.is_keras_tensor(upsample_psi))

	result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
	result_bn = BatchNormalization()(result)
	return result_bn

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)

	concat_axis = -1
	filters = 3

	conv1 = conv_bn_relu(64, filters, inputs)
	conv1 = conv_bn_relu(64, 3, conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = conv_bn_relu(128, 3, pool1)
	conv2 = conv_bn_relu(128, 3, conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = conv_bn_relu(256, 3, pool2)
	conv3 = conv_bn_relu(256, 3, conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = conv_bn_relu(512, 3, pool3)
	conv4 = conv_bn_relu(512, 3, conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = conv_bn_relu(1024, 3, pool4)
	conv5 = conv_bn_relu(512, 3, conv5)

	gating = UnetGatingSignal(conv5, is_batchnorm=True)
	attn_1 = AttnGatingBlock(conv4, gating, 512)
	up1 = concatenate([Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv5), attn_1],axis=3)
	conv6 = conv_bn_relu(512, 3, up1)
	conv6 = conv_bn_relu(256, 3, conv6)

	gating = UnetGatingSignal(conv6, is_batchnorm=True)
	attn_2 = AttnGatingBlock(conv3, gating, 256)
	up2 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv6), attn_2],axis=3)
	conv7 = conv_bn_relu(256, 3, up2)
	conv7 = conv_bn_relu(128, 3, conv7)

	gating = UnetGatingSignal(conv7, is_batchnorm=True)
	attn_3 = AttnGatingBlock(conv2, gating, 128)
	up3 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv7), attn_3],axis=3)
	conv8 = conv_bn_relu(128, 3, up3)
	conv8 = conv_bn_relu(64, 3, conv8)

	gating = UnetGatingSignal(conv8, is_batchnorm=True)
	attn_4 = AttnGatingBlock(conv1, gating, 64)
	up4 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation="relu")(conv8), attn_4], axis=3)
	conv9 = conv_bn_relu(64, 3, up4)
	conv9 = conv_bn_relu(32, 3, conv9)

	conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'

	model = Model(inputs=inputs, outputs=conv10)
	model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

	return model
'''

if  __name__=='__main__':

	model = UNet((256, 256,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, upconv=True, maxpool=True, residual=True)
	model.summary()
