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
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
import tensorflow as tf


class YoloDataGenerator:
    def __init__(self, train_image_path, input_shape, output_shape, batch_size, validation_split=0.0):
        """
        :param train_image_path:
            Path where training data is stored.
            The file name of the image and the label should be the same.

        :param input_shape:
            (height, width, channel) format of model input size
            If the channel is 1, train with a gray image, otherwise train with a color image.

        :param output_shape:
            Output shape extracted from built model.

        :param batch_size:
            Batch size of training.

        :param validation_split:
            The percentage of data that will be used as validation data.
        """
        image_paths = self.__init_image_paths(train_image_path)
        self.train_image_paths, self.validation_image_paths = self.__split_paths(image_paths, validation_split)
        self.__train_generator_flow = GeneratorFlow(self.train_image_paths, input_shape, output_shape, batch_size)
        self.__validation_generator_flow = GeneratorFlow(self.validation_image_paths, input_shape, output_shape, batch_size)

    @classmethod
    def empty(cls):
        """
        Empty class method for only initializing.
        """
        return cls.__new__(cls)

    def flow(self, subset='training'):
        """
        Flow function to load and return the batch.
        """
        if subset == 'training':
            return self.__train_generator_flow
        elif subset == 'validation':
            return self.__validation_generator_flow

    @staticmethod
    def __init_image_paths(train_image_path):
        """
        The path of the training data is extracted from the sub-path of train_image_path.
        The backslash of all paths is replaced by a slash because in case of running on the unix system.
        """
        image_paths = []
        image_paths += glob(f'{train_image_path}/*.jpg')
        image_paths += glob(f'{train_image_path}/*.png')
        image_paths += glob(f'{train_image_path}/*/*.jpg')
        image_paths += glob(f'{train_image_path}/*/*.png')
        image_paths = np.asarray(image_paths)
        for i in range(len(image_paths)):
            image_paths[i] = image_paths[i].replace('\\', '/')
        return sorted(image_paths)

    @staticmethod
    def __split_paths(image_paths, validation_split):
        """
        After mixing the paths of all training data, ths paths are divided according to the validation split ratio.
        """
        assert 0.0 <= validation_split <= 1.0
        image_paths = np.asarray(image_paths)
        if validation_split == 0.0:
            return image_paths, np.asarray([])
        r = np.arange(len(image_paths))
        np.random.shuffle(r)
        image_paths = image_paths[r]
        num_train_image_paths = int(len(image_paths) * (1.0 - validation_split))
        train_image_paths = image_paths[:num_train_image_paths]
        validation_image_paths = image_paths[num_train_image_paths:]
        return train_image_paths, validation_image_paths


class GeneratorFlow(tf.keras.utils.Sequence):
    """
    Custom data generator flow for YOLO model.
    Usage:
        generator_flow = GeneratorFlow(image_paths=image_paths)
    """

    def __init__(self, image_paths, input_shape, output_shape, batch_size):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size
        fs = []
        for i in range(start_index, start_index + self.batch_size):
            fs.append(self.pool.submit(self.__load_img, self.image_paths[self.random_indexes[i]]))
        for f in fs:
            cur_img_path, x = f.result()
            if x.shape[1] > self.input_shape[1] or x.shape[0] > self.input_shape[0]:
                x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
            else:
                x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR)
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            with open(f'{cur_img_path[:-4]}.txt', mode='rt') as file:
                label_lines = file.readlines()
            y = np.zeros((self.output_shape[2], self.output_shape[0], self.output_shape[1]), dtype=np.float32)
            grid_width_ratio = 1 / float(self.output_shape[1])
            grid_height_ratio = 1 / float(self.output_shape[0])
            for label_line in label_lines:
                class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                center_row = int(cy * self.output_shape[0])
                center_col = int(cx * self.output_shape[1])
                y[0][center_row][center_col] = 1.0
                y[1][center_row][center_col] = (cx - (center_col * grid_width_ratio)) / grid_width_ratio
                y[2][center_row][center_col] = (cy - (center_row * grid_height_ratio)) / grid_height_ratio
                y[3][center_row][center_col] = w
                y[4][center_row][center_col] = h
                y[int(class_index + 5)][center_row][center_col] = 1.0
            y = np.moveaxis(np.asarray(y), 0, -1).reshape(self.output_shape)
            batch_y.append(y)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        return batch_x, batch_y

    def __load_img(self, path):
        return path, cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)

    def __len__(self):
        """
        Number of total iteration.
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Mix the image paths at the end of each epoch.
        """
        np.random.shuffle(self.random_indexes)
