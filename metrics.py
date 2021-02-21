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


def precision(y_true, y_pred):
    """
    precision = True Positive / True Positive + False Positive
    precision = True Positive / All Detections
    """
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_pred[:, :, :, 0]) + 1e-5)


def recall(y_true, y_pred):
    """
    recall = True Positive / True Positive + False Negative
    recall = True Positive / All Ground Truths
    """
    tp = tf.reduce_sum(y_pred[:, :, :, 0] * y_true[:, :, :, 0])
    return tp / (tf.reduce_sum(y_true[:, :, :, 0]) + 1e-5)


def f1(y_true, y_pred):
    """
    Harmonic mean of precision and recall.
    f1 = 1 / (precision^-1 + precision^-1) * 0.5
    f1 = 2 * precision * recall / (precision + recall)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (p * r * 2.0) / (p + r + 1e-5)
