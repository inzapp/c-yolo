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
import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor_v2


class ConfidenceLoss(tf.keras.losses.Loss):
    """
    This loss function is used to reduce the loss of the confidence channel with some epochs before training begins.
    """

    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        """
        SSE at whole confidence channel.
        No used lambda_no_obj factor in here.
        In this architecture, the SSE loss for the whole confidence channel converges reliably because of pre confidence training.
        
        If pre_confidence_train_epochs is 0, the loss to the confidence channel may not be reduced reliably.
        In this case, you should use lambda_no_obj or change the loss function for the confidence channel.
        The larger the output grid size, the worse this will be.
        
        We recommend using pre confidence train because experimentally demonstrates that it is better to use it.
        """
        return tf.reduce_sum(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]))


class ConfidenceWithBoundingBoxLoss(tf.keras.losses.Loss):
    """
    This loss function is used to reduce the loss of the confidence and bounding box channel with some epochs before training begins.
    """

    def __init__(self, coord=5.0):
        """
        :param coord: coord value of bounding box loss. 5.0 is recommendation value in yolo paper.
        """
        self.coord = coord
        super(ConfidenceWithBoundingBoxLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_loss = ConfidenceLoss()(y_true, y_pred)
        """
        SSE at x, y regression
        """
        x_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 1] - (y_pred[:, :, :, 1] * y_true[:, :, :, 0])))
        y_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 2] - (y_pred[:, :, :, 2] * y_true[:, :, :, 0])))

        """
        SSE (sqrt(obj(x))) at width and height regression loss
        Sqrt was used to weight the width, height loss for small boxes.
        
        To avoid dividing by zero when going through the derivative formula of sqrt,
        Adds the eps value to the sqrt parameter.
        
        Derivative of sqrt : 1 / 2 * sqrt(x)
        """
        w_true = tf.sqrt(y_true[:, :, :, 3] + 1e-4)
        w_pred = tf.sqrt(y_pred[:, :, :, 3] + 1e-4)
        w_loss = tf.reduce_sum(tf.square(w_true - (w_pred * y_true[:, :, :, 0])))
        h_true = tf.sqrt(y_true[:, :, :, 4] + 1e-4)
        h_pred = tf.sqrt(y_pred[:, :, :, 4] + 1e-4)
        h_loss = tf.reduce_sum(tf.square(h_true - (h_pred * y_true[:, :, :, 0])))
        bbox_loss = x_loss + y_loss + w_loss + h_loss
        return confidence_loss + (bbox_loss * self.coord)


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_bbox_loss = ConfidenceWithBoundingBoxLoss()(y_true, y_pred)

        """
        SSE at all classification
        """
        classification_loss = tf.reduce_sum(tf.reduce_sum(tf.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]), axis=-1) * y_true[:, :, :, 0])
        return confidence_bbox_loss + classification_loss
