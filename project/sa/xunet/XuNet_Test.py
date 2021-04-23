'''
Author: 李大秋
Date: 2021-04-21 21:10:07
LastEditTime: 2021-04-23 15:22:09
LastEditors: vscode
Description: In User Settings Edit
FilePath: /myapps/src/xunet/XuNet_Test.py
'''

import numpy
import numpy as np
import random
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from keras.layers import Activation
import tensorflow as tf
import cv2c
from tensorflow.keras.layers import Lambda, Layer, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, SpatialDropout2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, \
    BatchNormalization
from keras.layers.core import Reshape
from keras import optimizers
from tensorflow.keras import regularizers
from keras import Input, Model
from time import time
import time as tm
from keras.initializers import Constant, RandomNormal, glorot_normal
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.utils import plot_model
from keras.layers import concatenate
import glob
from skimage.util.shape import view_as_blocks
from keras.utils import np_utils



################################################## 30 SRM FILTERS
srm_weights = np.load('SRM_Kernels.npy')
biasSRM = numpy.ones(30)
print(srm_weights.shape)
# srm_weights=np.resize(srm_weights,(5,5,3,30))
print(srm_weights.shape)
################################################## TLU ACTIVATION FUNCTION
T3 = 3;


def Tanh3(x):
    tanh3 = K.tanh(x) * T3
    return tanh3


##################################################

def Xu_Net(img_size=256, compile=True):
    # tf.reset_default_graph()
    print("compile:",compile)
    tf.keras.backend.clear_session()
    print("using", 2, "classes")

    # Preprocessing
    inputs = tf.keras.Input(shape=(img_size, img_size, 1), name="input_1")
    # print("testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest", inputs.shape)
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
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    # Compile
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)

    if compile:
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['acc'])
        print("Xunet")
    return model


def test(model,  X_test, y_test, batch_size, epochs, initial_epoch=0,
          model_name=""):
    start_time = tm.time()
    # tensorboard_dir = os.path.join('D:\\OneDrive - Lolihub\\DAQIU\\DLGP\\CrossDomain\\usual\\logs\\')
    log_dir = "D:\\OneDrive - Lolihub\\DAQIU\\DLGP\\CrossDomain\\usual\\logs\\" + model_name + "_" +'1617771589.92243' #"1617420166.623087"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir)
    # print("1111111111111111111111111111111111111111111111111111111111111111111111111111")
    filepath = log_dir + "\saved-model-373-0.60.hdf5"
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, mode='max')
    model.reset_states()
    print("filepath",filepath)
    model.load_weights(filepath)
    # history = model.fit(X_train, y_train, epochs=epochs,
    #                     callbacks=[tensorboard, checkpoint],
    #                     batch_size=batch_size, validation_data=(X_valid, y_valid), initial_epoch=initial_epoch)
    # history = model.fit(X_train, y_train, epochs=epochs,
    #                     callbacks=[tensorboard, checkpoint],
    #                     batch_size=batch_size,  initial_epoch=initial_epoch)
    # print("网络评估参数：---------------------------------------------------------------------------------------")
    # print(history.history.keys())
    model.summary()  #打印模型的概述信息
    plot_model(model, log_dir+'/'+'model_plot.png')# 保存模型的基本结构图

    metrics = model.evaluate(X_test, y_test, verbose=0)
    print("test:metrics",metrics)


    # results_dir = "D:\\OneDrive - Lolihub\\DAQIU\\DLGP\\CrossDomain\\usual\\Results\\" + model_name + "_" + "{}".format(time())+ "\\"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    # with plt.style.context('seaborn-white'):
    #     plt.figure(figsize=(10, 10))
    #     # plt.subplot(1,2,1)
    #     # Plot training & validation accuracy values
    #     plt.plot(history.history['acc'])
    #     plt.plot(history.history['val_acc'])
    #     plt.title('Accuracy Vs Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Validation'], loc='upper left')
    #     plt.grid('on')
    #     plt.savefig(results_dir + 'Accuracy_Xu_Net_' + model_name + '.eps', format='eps')
    #     plt.savefig(results_dir + 'Accuracy_Xu_Net_' + model_name + '.svg', format='svg')
    #     plt.savefig(results_dir + 'Accuracy_Xu_Net_' + model_name + '.pdf', format='pdf')
    #     plt.show()
    #
    #     plt.figure(figsize=(10, 10))
    #     # plt.subplot(1,2,2)
    #     # Plot training & validation loss values
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    #     plt.title('Loss Vs Epochs')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Validation'], loc='upper left')
    #     plt.grid('on')
    #     plt.savefig(results_dir + 'Loss_Xu_Net_' + model_name + '.eps', format='eps')
    #     plt.savefig(results_dir + 'Loss_Xu_Net_' + model_name + '.svg', format='svg')
    #     plt.savefig(results_dir + 'Loss_Xu_Net_' + model_name + '.pdf', format='pdf')
    #     plt.show()

    TIME = tm.time() - start_time
    print("Time " + model_name + " = %s [seconds]" % TIME)
    return {k: v for k, v in zip(model.metrics_names, metrics)}


n = 512


# def load_images2(path_pattern):
#     print("path_patternpath_patternpath_patternpath_pattern:", path_pattern)
#     # im_files=glob.glob(os.path.join(path_pattern,'*.png'))
#     im_files = glob.glob(path_pattern)
#     # files=glob.glob(path_pattern)
#     print("im_filesim_filesim_filesim_filesim_filesim_files:", im_files)
#
#     X = []
#     # for f in im_files:
#     #     I = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#     #     patches = view_as_blocks(I, (n, n))
#     #     for i in range(patches.shape[0]):
#     #         for j in range(patches.shape[1]):
#     #             X.append([patches[i, j]])
#     # X = numpy.array(X)
#     X = X[5000:, :, :, :]
#     print("load_imagesload_imagesload_imagesload_images",X.shape)
#     return X

def load_images0(path_pattern):
    print("path_patternpath_patternpath_patternpath_pattern:", path_pattern)
    # im_files=glob.glob(os.path.join(path_pattern,'*.png'))
    im_files = glob.glob(path_pattern)
    # files=glob.glob(path_pattern)
    print("im_filesim_filesim_filesim_filesim_filesim_files:", im_files)

    X = []
    Y = []
    for f in im_files:
        # print("fffffffffffffffffffffffffff",f)
        I = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",I.shape)
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append([patches[i, j]])
    X = numpy.array(X)
    X=  np.resize(X,(7000,1,256,256))
    print("load_imagesload_imagesload_imagesload_images", X.shape)

    X1 = X[:5000,:,:,:]
    Y1 = X[5000:6000,:,:,:]
    T1 = X[6000:,:,:,:]
    print("X.shape--load_imagesload_imagesload_imagesload_images",X1.shape)
    print("Y.shape--load_imagesload_imagesload_imagesload_images",Y1.shape)
    print("T1.shape--load_imagesload_imagesload_imagesload_images", T1.shape)
    # X1 =X1.resize()
    # Y1 =Y1.resize()
    return X1,Y1,T1

def load_images(path_pattern):
    print("path_patternpath_patternpath_patternpath_pattern:", path_pattern)
    # im_files=glob.glob(os.path.join(path_pattern,'*.png'))
    im_files = glob.glob(path_pattern)
    # files=glob.glob(path_pattern)
    print("im_filesim_filesim_filesim_filesim_filesim_files:", im_files)

    X = []
    Y = []
    for f in im_files:
        # print("fffffffffffffffffffffffffff",f)
        I = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", I.shape)
        # I = np.resize(I, ( 3, 512, 512))
        print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",I.shape)
        # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",I.shape)
        # patches = view_as_blocks(I, (n, n))
        # print("patchespatchespatchespatchespatches",patches.shape)

        # for i in range(patches.shape[0]):
        #     for j in range(patches.shape[1]):
        #         X.append([patches[i, j]])
        X.append(I)
    # print(X)
    X = numpy.array(X)
    print(X.shape)
    # X = np.resize(X,(7000,3, 512,512))
    print(X.shape)
    X=  np.resize(X,(2000,1,256,256))
    print("load_imagesload_imagesload_imagesload_images", X.shape)

    X1 = X[:,:,:,:]
    Y1 = X[:1000,:,:,:]
    T1 = X[1000:,:,:,:]
    print("X.shape--load_imagesload_imagesload_imagesload_images",X1.shape)
    print("Y.shape--load_imagesload_imagesload_imagesload_images",Y1.shape)
    print("T1.shape--load_imagesload_imagesload_imagesload_images", T1.shape)
    # X1 =X1.resize()
    # Y1 =Y1.resize()
    return X1,Y1,T1

# Train Images
Xc,Yc,Tc = load_images('D:/1\DAQIU\DLGP\Steganalysis/alaska-master/alaska2-image-steganalysis/00/*.jpg')
# Xc,Yc,Tc = load_images('D:\OneDrive - Lolihub\DAQIU\Desktop\DataSet/COVER1/*.png')
Xs,Ys,Ts = load_images('D:/1\DAQIU\DLGP\Steganalysis/alaska-master/alaska2-image-steganalysis/33/*.jpg')
# Xs,Ys,Ts = load_images('D:\OneDrive - Lolihub\DAQIU\Desktop\DataSet/WOWstego(0.4)/*.png')

# print("Xc.shapeXc.shapeXc.shapeXc.shapeXc.shapeXc.shapeXc.shapeXc.shape",Xc.shape)
# print("Xs.shapeXs.shapeXs.shapeXs.shapeXs.shapeXs.shapeXs.shapeXs.shape",Xs.shape)

# Validation Images
# _,Yc = load_images('D:\OneDrive - Lolihub\DAQIU\Desktop\DataSet/COVER1/*.png')
# _,Ys = load_images('D:\OneDrive - Lolihub\DAQIU\Desktop\DataSet/WOWstego(0.4)/*.png')

X = (numpy.vstack((Xc, Xs)))
Y = (numpy.vstack((Yc, Ys)))
T = (numpy.vstack((Tc, Ts)))
# print("xxxxxxxxxxxxxxxxxxxxxxxxx",X.shape)
# print("yyyyyyyyyyyyyyyyyyyyyyyyy",Y.shape)

Xt = (numpy.hstack(([0] * len(Xc), [1] * len(Xs))))
Yt = (numpy.hstack(([0] * len(Yc), [1] * len(Ys))))
Tt = (numpy.hstack(([0] * len(Tc), [1] * len(Ts))))

Xt = np_utils.to_categorical(Xt, 2)
Yt = np_utils.to_categorical(Yt, 2)
Tt = np_utils.to_categorical(Tt, 2)

####random train
# idx=np.arange(len(X))
# random.shuffle(idx)
# X=X[idx]
# Xt=Xt[idx]
# print(X.shape)
X = np.rollaxis(X, 1, 4)  # channel axis shifted to last axis
# print(X.shape)

# print(Y.shape)
Y = np.rollaxis(Y, 1, 4)  # channel axis shifted to last axis
# print(Y.shape)
T = np.rollaxis(T, 1, 4)  # channel axis shifted to last axis

X_train = X
y_train = Xt
X_valid = Y
y_valid = Yt
X_test  = T
y_test  = Tt

print("训练集样本C+S",X_train.shape)
print("训练集标签C+S",y_train.shape)
print("验证集样本C+S",X_valid.shape)
print("验证集标签C+S",y_valid.shape)
print("测试集样本C+S",X_test.shape)
print("测试集标签C+S",y_test.shape)

base_name = "04WOW1"
m_name = "XU_Net"

model = Xu_Net(compile=True)
name = "Model_" + m_name + "_" + base_name
historu_name, history = test(model, X_test, y_test,batch_size=64, epochs=600,
                   model_name=name)
# print("historu_name:",historu_name)
# print("history:",history)
