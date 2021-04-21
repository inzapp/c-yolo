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
from yolo import Yolo

if __name__ == '__main__':
    """
    Train model using fit method.
    
    train_image_path:
        The path to the directory where the training data is located.
        
        There must be images and labels in this directory.
        The image and label must have the same file name and not be in different directories.
        
    input_shape:
        (height, width, channel) format of model input size
        If the channel is 1, train with a gray image, otherwise train with a color image.
        
    batch_size:
        2 batch is recommended.
        
    lr:
        Learning rate value while training. 1e-3 ~ 1e-4 is recommended.
        
    epochs:
        Epochs value.
        
    curriculum_epochs:
        Epochs to pre-reduce the loss to the confidence and bounding box channel before starting the training.
        
    validation_split:
        The percentage of data that will be used as validation data.
        
    validation_image_path:
        Use this parameter if the validation data is in a different path from the training data.
        
    training_view:
        During training, the image is forwarded in real time, showing the results are shown.
        False if training is on a server system without IO equipment.
    """
    model = Yolo()
    model.fit(
        train_image_path=r'C:\inz\train_data\coco_2017',
        model_name='coco_2017_416_416_3',
        input_shape=(416, 416, 3),
        batch_size=2,
        lr=1e-3,
        epochs=1000,
        curriculum_epochs=10,
        validation_split=0.2,
        training_view=True,
        mixed_float16_training=True)
    model.evaluate()
