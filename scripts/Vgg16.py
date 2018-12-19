import tensorflow as tf
from keras.models import Sequential
import keras
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, ZeroPadding2D
from Setting import *
import cv2
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import VGG16

'''build model'''


def vgg16_model():
    model = VGG16(include_top=True, weights='imagenet')
    return model


def vgg16_model2(path=path_vgg16, use_weight=use_vgg_weight):
    data_format = 'channels_first'
    model = Sequential()
    model.add(ZeroPadding2D((1,1), data_format=data_format, input_shape=(3, 224, 224), name='b1z1'))
    model.add(Conv2D(64, (3, 3), data_format=data_format, padding='same', activation='relu', name='b1c1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='b1m1'))

    model.add(Conv2D(128, (3, 3), data_format=data_format, padding='same', activation='relu', name='b2c1'))
    model.add(Conv2D(128, (3, 3), data_format=data_format, padding='same', activation='relu', name='b2c2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='b2m1'))

    model.add(Conv2D(256, (3, 3), data_format=data_format, padding='same', activation='relu', name='b3c1'))
    model.add(Conv2D(256, (3, 3), data_format=data_format, padding='same', activation='relu', name='b3c2'))
    model.add(Conv2D(256, (3, 3), data_format=data_format, padding='same', activation='relu', name='b3c3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='b3m1'))

    model.add(Conv2D(512, (3, 3), data_format=data_format, padding='same', activation='relu', name='b4c1'))
    model.add(Conv2D(512, (3, 3), data_format=data_format, padding='same', activation='relu', name='b4c2'))
    model.add(Conv2D(512, (3, 3), data_format=data_format, padding='same', activation='relu', name='b4c3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='b4m1'))

    model.add(Conv2D(512, (3, 3), data_format=data_format, padding='same', activation='relu', name='b5c1'))
    model.add(Conv2D(512, (3, 3), data_format=data_format, padding='same', activation='relu', name='b5c2'))
    model.add(Conv2D(512, (3, 3), data_format=data_format, padding='same', activation='relu', name='b5c3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='b5m1'))

    model.add(Flatten(name='fal'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if use_weight:
        model.load_weights(path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def feature_output(image, model):
    im = cv2.resize(image, (224, 224)).astype(np.float32)
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        im = im[::-1, :, :]
        # Zero-center by mean pixel
        im[0, :, :] -= 103.939
        im[1, :, :] -= 116.779
        im[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        # Zero-center by mean pixel
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    # im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    inputs = [K.learning_phase()] + model.inputs

    # @The flatten layer is 18
    flatten_output = K.function(inputs, [model.layers[19].output])
    return flatten_output([0] + [im])


# model = vgg16_model()
# model.summary()
# print(model.layers[19])