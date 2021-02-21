# C-YOLO(Curriculum YOLO)

Curriculum YOLO is a new version of YOLO customizing YOLOv3.

The way the model finds objects is not different from YOLO.

However, the method of training a model is different.

C-YOLO uses a training approach based on curriculum learning in the following courses:

1. Train only on the presence or absence of objects in each grid.

2. Then train to predict the size of the object.

3. Finally, train on information about the entire object, including class classification.

We knew that using this training approach, we can reach the global optimum more reliably and quickly for complex datasets.

If you want, you may not use this training method.

Then, it will be the same as normal YOLO.

## Why did you use curriculum learning?

The loss function of Darknet YOLO is shown below.

(YOLO loss photo)

Propagate the gradient as it is for the loss to the object.

Propagate fewer gradient through lambda_noobj for non-object grids.

Why are you doing this?

Suppose there is no lamda_noobj.

As the model's output grid size grows, the propagating gradient becomes larger.

This is because YOLO uses an SSE function that sums up the loss of all channels.

Therefore, training can become very unstable.

To avoid this, Darknet adjusted the gradient propagating through lambda_noobj not too large, which worked very well.

But think about it. Due to the nature of the Object detection task, most grids in the image do not contain objects.

There are even sometimes images where objects do not exist (to reduce false positives).

Therefore, for Confidence channels, most values must converge to zero.

The curriculum training method is designed based on the fact that most grids should be zero.

Curriculum YOLO conducts training by dividing the loss function into three types as follows.

### Confidence loss

<img src="/md/confidence_loss.jpg" width="350">

### Confidence with bounding box loss

<img src="/md/confidence_with_bbox_loss.jpg" width="550">

### YOLO loss

<img src="/md/c_yolo_loss.jpg" width="550">

After stabilizing first with Confidence loss, you can propagate the gradient to the entire confidence channel without using lambda_noobj.

So there is no lambda-noobj in C-YOLO's loss function.

This is why Curriculum YOLO is better than Darknet YOLO in terms of learning efficiency.

## Backbone

C-YOLO provides darknet-53, vgg19 backbone networks by default.

You can test various backbone models and determine the best model.

## Training view

Unlike other object detectors, C-YOLO can see the actual forwarding results of the model in real time during training.

When we set the training_view=true flag, we use the models trained to date to forward random data.

The bounding boxed image is displayed in real time.

## Convenient customization with Keras

Customizing Darknet YOLO is more cumbersome than you think.

You only need to install Python and Tensorflow to make your own customized YOLO.
