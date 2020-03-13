import os
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, ConvLSTM2D, Conv2D, Lambda
import tensorflow.keras.backend as K

from weatherbench.train_nn import PeriodicConv2D


def build_cnn_ltsm(filters, kernels, input_shape, activation='elu', dr=0):
    """Fully convolutional network"""
    x = input = Input(shape=input_shape)
    x = ConvLSTM2D(filters = 64, kernel_size = 5, dilation_rate = 2, padding= 'same',activation= 'tanh',
                   data_format = 'channels_last')(x)
    for f, k in zip(filters[:-1], kernels[:-1]):
        x = PeriodicConv2D(f, k, activation=activation)(x)
        if dr > 0: x = Dropout(dr)(x)
    output = PeriodicConv2D(filters[-1], kernels[-1])(x)
    return keras.models.Model(input, output)