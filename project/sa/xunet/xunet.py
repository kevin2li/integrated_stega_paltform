'''
Author: 李大秋
Date: 2021-04-21 21:10:07
LastEditTime: 2021-04-28 18:58:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /myapps/src/xunet/XuNet_Test.py
'''
# %%
import numpy
import numpy as np
import tensorflow as tf
from keras import backend as K
from PIL import Image
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, GlobalAveragePooling2D, Lambda, ReLU)
from tqdm import tqdm

srm_weights = np.load('/home/kevin2li/wave/myapps/project/sa/xunet/SRM_Kernels.npy')
biasSRM = numpy.ones(30)

T3 = 3
def Tanh3(x):
    tanh3 = K.tanh(x) * T3
    return tanh3

# %%
# def XuNet(img_size=256, compile=True):
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
    
    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)

    def __call__(self, img: np.ndarray):
        assert len(img.shape) == 2, 'input should be gray image'
        w, h = img.shape
        img = img.reshape(1, w, h, 1)
        out = self.model.predict(img)
        return out
        
# %%
# model = XuNet()
# model.load_weights('/home/kevin2li/wave/myapps/project/sa/xunet/saved-model-117-0.85.hdf5')
# path = '/mnt/f/code/steganography_platform_pl/data/0/19.png'
# img = np.array(Image.open(path))
# out = model(img)
# out
# %%
