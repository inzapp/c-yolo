"""
Authors : inzapp

Github url : https://github.com/inzapp/c-yolo

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, output_channel):
        self.__input_shape = input_shape
        self.__output_channel = output_channel

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def build(self):
        return self.__vgg_19()
        # return self.__darknet_53()

    def __darknet_53(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(1, 32, 3, input_layer, False)
        x = self.__conv_blocks(1, 64, 3, x, True)
        skip_connection = x

        # residual block 1
        x = self.__conv_blocks(1, 32, 1, x, False)
        x = self.__conv_blocks(1, 64, 3, x, False)
        x = tf.keras.layers.Add()([skip_connection, x])
        x = self.__conv_blocks(1, 128, 3, x, True)
        skip_connection = x

        # residual block 2
        for _ in range(2):
            x = self.__conv_blocks(1, 64, 1, x, False)
            x = self.__conv_blocks(1, 128, 3, x, False)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 256, 3, x, True)
        skip_connection = x

        # residual block 3
        for _ in range(8):
            x = self.__conv_blocks(1, 128, 1, x, False)
            x = self.__conv_blocks(1, 256, 3, x, False)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 512, 3, x, True)
        skip_connection = x

        # residual block 3
        for _ in range(4):
            x = self.__conv_blocks(1, 256, 1, x, False)
            x = self.__conv_blocks(1, 512, 3, x, False)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x
        x = self.__conv_blocks(1, 1024, 3, x, True)
        skip_connection = x

        # residual block 4
        for _ in range(4):
            x = self.__conv_blocks(1, 512, 1, x, False)
            x = self.__conv_blocks(1, 1024, 3, x, False)
            x = tf.keras.layers.Add()([skip_connection, x])
            skip_connection = x

        x = self.__point_wise_conv(self.__output_channel, x)
        return tf.keras.models.Model(input_layer, x)

    def __vgg_19(self):
        input_layer = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.__conv_blocks(2, 64, 3, input_layer, True)
        x = self.__conv_blocks(3, 128, 3, x, True)
        x = self.__conv_blocks(4, 256, 3, x, True)
        x = self.__conv_blocks(4, 512, 3, x, True)
        x = self.__conv_blocks(4, 512, 3, x, True)
        x = self.__point_wise_conv(self.__output_channel, x)
        return tf.keras.models.Model(input_layer, x)

    def __conv_blocks(self, n_convolutions, filters, kernel_size, x, max_pool=False):
        for _ in range(n_convolutions):
            x = self.__conv_block(filters, kernel_size, x, False)
        if max_pool:
            x = tf.keras.layers.MaxPool2D()(x)
        return x

    @staticmethod
    def __conv_block(filters, kernel_size, x, max_pool=False):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if max_pool:
            x = tf.keras.layers.MaxPool2D()(x)
        return x

    @staticmethod
    def __point_wise_conv(filters, x):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            activation='sigmoid')(x)
