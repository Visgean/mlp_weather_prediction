from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras import regularizers

from weatherbench.train_nn import PeriodicConv2D


def get_vgg16(filters, kernels, input_shape, activation='elu', dr=0):
    model = Sequential()

    # Encoder
    # Block 1
    model.add(BatchNormalization(axis=3, input_shape=(32, 64, 2)))
    # model.add(Dropout(.2))
    model.add(PeriodicConv2D(64, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block1_conv1', input_shape=(32, 64, 2)))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(64, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(128, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block2_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(128, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(256, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block3_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(256, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block3_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(512, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block4_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(512, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block4_conv2'))
    model.add(MaxPooling2D((2, 3), strides=(2, 3), name='block4_pool'))

    # Block 5
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(512, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block5_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(512, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block5_conv2'))

    # Decoder
    # Block 6
    model.add(UpSampling2D((2, 4), name='block6_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(512, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block6_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(512, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block6_conv2'))

    # Block 7
    model.add(UpSampling2D((2, 2), name='block7_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(256, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block7_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(256, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block7_conv2'))

    # Block 8
    model.add(UpSampling2D((2, 2), name='block8_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(128, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block8_conv1'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(128, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block8_conv2'))

    # Block 9
    model.add(UpSampling2D((2, 2), name='block9_upsampl'))
    model.add(BatchNormalization(axis=3))
    model.add(PeriodicConv2D(64, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block9_conv1'))
    model.add(BatchNormalization(axis=3))
    # model.add(Dropout(.2))
    model.add(PeriodicConv2D(64, 3, padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block9_conv2'))

    # Output
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(2, (1, 1), padding='valid', activation='relu', bias_regularizer=regularizers.l1(0.01),
                     name='block10_conv1'))

    return model


model = get_vgg16(1,2,2,3,4)