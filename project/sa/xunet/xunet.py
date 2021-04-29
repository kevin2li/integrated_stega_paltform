'''
Author: 李大秋
Date: 2021-04-21 21:10:07
LastEditTime: 2021-04-28 20:00:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /myapps/src/xunet/XuNet_Test.py
'''
# %%
from pathlib import Path
import glob

import cv2
import numpy
import numpy as np
import tensorflow as tf
import tqdm
from keras import backend as K
from keras.utils import np_utils
from PIL import Image
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Concatenate, Conv2D, Dense,
                                     GlobalAveragePooling2D, Lambda, ReLU)
from tqdm import tqdm

root_dir = Path('/home/kevin2li/wave/myapps/')  # wsl
# root_dir = Path('/root/wave/myapp')  # aliyun
# root_dir = Path('/home/likai/integrated_stega_paltform/') # lab

srm_weights = np.load(str(root_dir / 'project/sa/xunet/SRM_Kernels.npy'))
biasSRM = numpy.ones(30)

T3 = 3
def Tanh3(x):
    tanh3 = K.tanh(x) * T3
    return tanh3

# %%
class XuNet():
    def __init__(self, img_size=256):
        tf.keras.backend.clear_session()
        # Preprocessing
        inputs = tf.keras.Input(shape=(img_size, img_size, 1), name="input_1")
        layers = tf.keras.layers.Conv2D(30, (5, 5), weights=[srm_weights, biasSRM], strides=(1, 1), trainable=False,
                                        activation=Tanh3, use_bias=True)(inputs)
        # Block 1
        # Layer 0
        layers = Conv2D(8, (5, 5), strides=(1, 1), padding="same", kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = Lambda(tf.keras.backend.abs)(layers)
        layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None,
                                    renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        layers = Concatenate()([layers, layers, layers])

        # Block 2
        # Layer 1
        layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
        layers = Conv2D(16, (5, 5), strides=1, padding="same", kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
        layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None,
                                    renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        layers = AveragePooling2D((5, 5), strides=2, padding='same')(layers)

        # Block 3
        # Layer 2
        layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
        layers = Conv2D(32, (1, 1), strides=1, padding="same", kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
        layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None,
                                    renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        layers = AveragePooling2D((5, 5), strides=2, padding="same")(layers)

        # Block 4
        # Layer 3
        layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
        layers = Conv2D(64, (1, 1), strides=1, padding="same", kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
        layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None,
                                    renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        layers = AveragePooling2D((5, 5), strides=2, padding="same")(layers)
        # Block 5
        # Layer 4
        layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
        layers = Conv2D(128, (1, 1), strides=1, padding="same", kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
        layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None,
                                    renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        layers = Concatenate()([layers, layers, layers])
        layers = GlobalAveragePooling2D(data_format="channels_last")(layers)

        # Block 6
        # Layer 5, FC, Softmax
        # FC
        layers = Dense(128, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = Dense(64, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)
        layers = Dense(32, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = ReLU(negative_slope=0.1, threshold=0)(layers)

        # Softmax
        predictions = Dense(2, activation="softmax", name="output_1", kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                            bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)
        # Compile
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    def load_weights(self, path: str) -> None:
        self.model.reset_states()
        self.model.load_weights(path)

    def evaluate(self, X_test, Y_test, verbose=None):
        out = self.model.evaluate(X_test, Y_test)
        return out

    def __call__(self, img: np.ndarray):
        assert len(img.shape) == 2, 'input should be gray image'
        w, h = img.shape
        img = img.reshape(1, w, h, 1)
        out = self.model.predict(img)
        return out
        
# %%
# model = XuNet()
#     model.reset_states()
# model.load_weights('/home/kevin2li/wave/myapps/project/sa/xunet/saved-model-117-0.85.hdf5')
# # path = '/mnt/f/code/steganography_platform_pl/data/0/19.png'
# # img = np.array(Image.open(path))
# out = model(img)
# out
# %%
# def load_images(path_pattern):
#     print("path_patternpath_patternpath_patternpath_pattern:", path_pattern)
#     # im_files=glob.glob(os.path.join(path_pattern,'*.png'))
#     im_files = glob.glob(path_pattern)
#     # files=glob.glob(path_pattern)
#     # print("im_filesim_filesim_filesim_filesim_filesim_files:", im_files)

#     X = []
#     Y = []
#     progressbar = tqdm(im_files, total=len(im_files), desc='reading images')
#     for f in progressbar:
#         # print("fffffffffffffffffffffffffff",f)
#         I = np.array(Image.open(f))
#         progressbar.set_postfix({'shape': I.shape})
#         # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", I.shape)
#         # I = np.resize(I, ( 3, 512, 512))
#         # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",I.shape)
#         # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",I.shape)
#         # patches = view_as_blocks(I, (n, n))
#         # print("patchespatchespatchespatchespatches",patches.shape)

#         # for i in range(patches.shape[0]):
#         #     for j in range(patches.shape[1]):
#         #         X.append([patches[i, j]])
#         X.append(I)
#     # print(X)
#     X = numpy.array(X)
#     print(X.shape)
#     # X = np.resize(X,(7000,3, 512,512))
#     print(X.shape)
#     X=  np.resize(X,(20,1,256,256))
#     print("load_imagesload_imagesload_imagesload_images", X.shape)

#     X1 = X[:,:,:,:]
#     Y1 = X[:,:,:,:]
#     T1 = X[:,:,:,:]
#     print("X.shape--load_imagesload_imagesload_imagesload_images",X1.shape)
#     print("Y.shape--load_imagesload_imagesload_imagesload_images",Y1.shape)
#     print("T1.shape--load_imagesload_imagesload_imagesload_images", T1.shape)
#     # X1 =X1.resize()
#     # Y1 =Y1.resize()
#     return X1,Y1,T1

# Xc,Yc,Tc = load_images('/mnt/f/code/steganography_platform_pl/data/0/*.png')
# # Xs,Ys,Ts = load_images('D:/1\DAQIU\DLGP\Steganalysis/alaska-master/alaska2-image-steganalysis/33/*.jpg')
# Xs,Ys,Ts = load_images('/mnt/f/code/steganography_platform_pl/data/1/*.png')

# X = (numpy.vstack((Xc, Xs)))
# Y = (numpy.vstack((Yc, Ys)))
# T = (numpy.vstack((Tc, Ts)))
# # print("xxxxxxxxxxxxxxxxxxxxxxxxx",X.shape)
# # print("yyyyyyyyyyyyyyyyyyyyyyyyy",Y.shape)

# Xt = (numpy.hstack(([0] * len(Xc), [1] * len(Xs))))
# Yt = (numpy.hstack(([0] * len(Yc), [1] * len(Ys))))
# Tt = (numpy.hstack(([0] * len(Tc), [1] * len(Ts))))

# Xt = np_utils.to_categorical(Xt, 2)
# Yt = np_utils.to_categorical(Yt, 2)
# Tt = np_utils.to_categorical(Tt, 2)
# T = np.rollaxis(T, 1, 4) 

# model = XuNet()
# model.load_weights('/home/kevin2li/wave/myapps/project/sa/xunet/saved-model-117-0.85.hdf5')
# # path = '/mnt/f/code/steganography_platform_pl/data/0/19.png'
# # img = np.array(Image.open(path))
# metrics = model.evaluate(T, Tt, verbose=0)
# print("test:metrics",metrics)
# # %%
