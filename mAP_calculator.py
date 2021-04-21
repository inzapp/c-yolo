import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

iou_thresholds = [0.5]
confidence_threshold = 0.25  # only for tp, fp, fn
nms_iou_threshold = 0.45  # darknet yolo nms threshold value


def iou(a, b):
    a_x_min, a_y_min, a_x_max, a_y_max = a
    b_x_min, b_y_min, b_x_max, b_y_max = b
    intersection_width = min(a_x_max, b_x_max) - max(a_x_min, b_x_min)
    intersection_height = min(a_y_max, b_y_max) - max(a_y_min, b_y_min)
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
    b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
    union_area = a_area + b_area - intersection_area
    return intersection_area / (float(union_area) + 1e-5)


def get_y_true(label_lines, target_class_index):
    raw_width = 1000
    raw_height = 1000
    y_true = []
    for label_line in label_lines:
        class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
        if int(class_index) == target_class_index:
            x1 = int((cx - w / 2.0) * raw_width)
            x2 = int((cx + w / 2.0) * raw_width)
            y1 = int((cy - h / 2.0) * raw_height)
            y2 = int((cy + h / 2.0) * raw_height)
            y_true.append({
                'class': int(class_index),
                'bbox': [x1, y1, x2, y2],
                'discard': False})
    return y_true


def get_y_pred(y, target_class_index):
    global nms_iou_threshold, confidence_threshold
    raw_width = 1000
    raw_height = 1000
    rows, cols, channels = y.shape[0], y.shape[1], y.shape[2]

    y_pred = []
    for i in range(rows):
        for j in range(cols):
            confidence = y[i][j][0]
            if confidence < 0.005:  # darknet yolo mAP confidence threshold value
                continue

            class_index = -1
            max_percentage = -1
            for cur_channel_index in range(5, channels):
                if max_percentage < y[i][j][cur_channel_index]:
                    class_index = cur_channel_index - 5
                    max_percentage = y[i][j][cur_channel_index]
            if class_index != target_class_index:
                continue

            cx_f = j / float(cols) + 1.0 / float(cols) * y[i][j][1]
            cy_f = i / float(rows) + 1.0 / float(rows) * y[i][j][2]
            w = y[i][j][3]
            h = y[i][j][4]

            x_min_f = cx_f - w / 2.0
            y_min_f = cy_f - h / 2.0
            x_max_f = cx_f + w / 2.0
            y_max_f = cy_f + h / 2.0
            x_min = int(x_min_f * raw_width)
            y_min = int(y_min_f * raw_height)
            x_max = int(x_max_f * raw_width)
            y_max = int(y_max_f * raw_height)

            y_pred.append({
                'confidence': confidence,
                'bbox': [x_min, y_min, x_max, y_max],
                'class': class_index,
                'result': '',
                'precision': 0.0,
                'recall': 0.0,
                'discard': False})

    for i in range(len(y_pred)):
        if y_pred[i]['discard']:
            continue
        for j in range(len(y_pred)):
            if i == j or y_pred[j]['discard']:
                continue
            if iou(y_pred[i]['bbox'], y_pred[j]['bbox']) > nms_iou_threshold:
                if y_pred[i]['confidence'] >= y_pred[j]['confidence']:
                    y_pred[j]['discard'] = True

    y_pred_copy = np.asarray(y_pred.copy())
    y_pred = []
    for i in range(len(y_pred_copy)):
        if not y_pred_copy[i]['discard']:
            y_pred.append(y_pred_copy[i])
    return y_pred


def calc_ap(precisions, recalls):
    precisions = [1.0] + precisions
    recalls = [0.0] + recalls
    sorted_pure_precisions = sorted(list(set(precisions)), reverse=True)
    indexed_pure_precisions = list()
    prev_max_index = -1
    for i in range(len(sorted_pure_precisions)):
        max_index = -1
        for j in range(len(precisions)):
            if sorted_pure_precisions[i] == precisions[j]:
                max_index = j
        if max_index > prev_max_index:
            indexed_pure_precisions.append({'max_index': max_index, 'val': sorted_pure_precisions[i]})
            prev_max_index = max_index
    if len(indexed_pure_precisions) > 1:
        for i in range(1, len(indexed_pure_precisions)):
            start_index = indexed_pure_precisions[i - 1]['max_index'] + 1
            end_index = indexed_pure_precisions[i]['max_index']
            for interpolation_index in range(start_index, end_index + 1):
                precisions[interpolation_index] = indexed_pure_precisions[i]['val']

    if recalls[-1] < 1.0:
        precisions[-1] = 0.0

    ap = 0.0
    for i in range(len(precisions) - 1):
        ap += precisions[i] * (recalls[i + 1] - recalls[i])
    return ap


def calc_tp_fp_fn(y_true, y_pred, iou_threshold):
    global confidence_threshold
    for i in range(len(y_true)):
        y_true[i]['discard'] = False
    for i in range(len(y_pred)):
        y_pred[i]['discard'] = False

    tp = 0
    for i in range(len(y_true)):
        for j in range(len(y_pred)):
            if y_pred[j]['discard'] or y_true[i]['class'] != y_pred[j]['class']:
                continue
            if y_pred[j]['confidence'] < confidence_threshold:
                continue
            if iou(y_true[i]['bbox'], y_pred[j]['bbox']) > iou_threshold:
                y_true[i]['discard'] = True
                y_pred[j]['discard'] = True
                tp += 1
                break

    fp = 0
    for i in range(len(y_pred)):
        if not y_pred[i]['discard']:
            y_pred[i]['result'] = 'FP'
            fp += 1

    fn = 0
    for i in range(len(y_true)):
        if not y_true[i]['discard']:
            fn += 1
    return tp, fp, fn


def calc_ap_tp_fp_fn(y, label_lines, iou_threshold, target_class_index):
    y_true = get_y_true(label_lines, target_class_index)
    num_class_obj = len(y_true)
    if num_class_obj == 0:
        return None, None, None, None, None

    y_pred = get_y_pred(y, target_class_index)
    for i in range(len(y_true)):
        for j in range(len(y_pred)):
            if y_pred[j]['discard'] or y_true[i]['class'] != y_pred[j]['class']:
                continue
            if iou(y_true[i]['bbox'], y_pred[j]['bbox']) > iou_threshold:
                y_true[i]['discard'] = True
                y_pred[j]['discard'] = True
                y_pred[j]['result'] = 'TP'
                break

    for i in range(len(y_pred)):
        if not y_pred[i]['discard']:
            y_pred[i]['result'] = 'FP'

    tp_sum = 0
    fp_sum = 0
    y_pred = sorted(y_pred, key=lambda x: x['confidence'], reverse=True)
    for i in range(len(y_pred)):
        if y_pred[i]['result'] == 'TP':
            tp_sum += 1
        elif y_pred[i]['result'] == 'FP':
            fp_sum += 1

        total_sum = tp_sum + fp_sum
        y_pred[i]['precision'] = 0 if total_sum == 0 else tp_sum / float(total_sum)
        y_pred[i]['recall'] = tp_sum / float(len(y_pred))

    y_pred = sorted(y_pred, key=lambda x: x['recall'])
    precisions = []
    recalls = []
    ap = 0.0
    for i in range(len(y_pred)):
        precisions.append(y_pred[i]['precision'])
        recalls.append(y_pred[i]['recall'])

    if len(y_pred) > 0:
        ap = calc_ap(precisions, recalls)
    tp, fp, fn = calc_tp_fp_fn(y_true, y_pred, iou_threshold)
    return ap, tp, fp, fn, num_class_obj


@tf.function
def predict_on_graph(model, x):
    return model(x, training=False)


def load_x_label_lines(image_path, color_mode, input_size, input_shape):
    label_path = f'{image_path[:-4]}.txt'
    if not (os.path.exists(label_path) and os.path.isfile(label_path)):
        return None, None
    with open(label_path, mode='rt') as f:
        label_lines = f.readlines()
    if len(label_lines) == 0:
        return None, None
    x = cv2.imread(image_path, color_mode)
    x = cv2.resize(x, input_size)
    x = np.asarray(x).astype('float32').reshape((1,) + input_shape) / 255.0
    return x, label_lines


def calc_mean_average_precision(model_path, image_paths):
    global iou_thresholds
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    input_size = (input_shape[1], input_shape[0])
    color_mode = cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR
    num_classes = model.output_shape[-1] - 5

    obj_count = np.zeros((len(iou_thresholds), num_classes), dtype=np.int32)
    valid_count = np.zeros((len(iou_thresholds), num_classes), dtype=np.int32)
    aps = np.zeros((len(iou_thresholds), num_classes), dtype=np.int32)
    tps = np.zeros((len(iou_thresholds), num_classes), dtype=np.int32)
    fps = np.zeros((len(iou_thresholds), num_classes), dtype=np.int32)
    fns = np.zeros((len(iou_thresholds), num_classes), dtype=np.int32)

    pool = ThreadPoolExecutor(8)

    fs = []
    for image_path in image_paths:
        fs.append(pool.submit(load_x_label_lines, image_path, color_mode, input_size, input_shape))

    for f in tqdm(fs):
        x, label_lines = f.result()
        if x is None:
            continue
        y = np.asarray(predict_on_graph(model, x))[0]

        for iou_index, iou_threshold in enumerate(iou_thresholds):
            for class_index in range(num_classes):
                ap, tp, fp, fn, num_class_obj = calc_ap_tp_fp_fn(y, label_lines, iou_threshold, class_index)
                if ap is not None:
                    valid_count[iou_index][class_index] += 1
                    obj_count[iou_index][class_index] += num_class_obj
                    aps[iou_index][class_index] += ap
                    tps[iou_index][class_index] += tp
                    fps[iou_index][class_index] += fp
                    fns[iou_index][class_index] += fn

    f1_sum = 0.0
    mean_ap_sum = 0.0
    for iou_index, iou_threshold in enumerate(iou_thresholds):
        class_ap_sum = 0.0
        class_f1_sum = 0.0
        print(f'confidence threshold for tp, fp, fn calculate : {confidence_threshold:.2f}')
        for class_index in range(num_classes):
            cur_class_ap = aps[iou_index][class_index] / float(valid_count[iou_index][class_index])
            class_ap_sum += cur_class_ap
            cur_class_obj_count = obj_count[iou_index][class_index]
            cur_class_tp = tps[iou_index][class_index]
            cur_class_fp = fps[iou_index][class_index]
            cur_class_fn = fns[iou_index][class_index]
            cur_class_precision = cur_class_tp / (float(cur_class_tp + cur_class_fp) + 1e-5)
            cur_class_recall = cur_class_tp / (float(cur_class_tp + cur_class_fn) + 1e-5)
            cur_class_f1 = 2.0 * (cur_class_precision * cur_class_recall) / (cur_class_precision + cur_class_recall + 1e-5)
            class_f1_sum += cur_class_f1
            print(f'class {str(class_index):3s} ap : {cur_class_ap:.4f}, obj_count : {str(cur_class_obj_count):6s}, tp : {str(cur_class_tp):6s}, fp : {str(cur_class_fp):6s}, fn : {str(cur_class_fn):6s}, precision : {cur_class_precision:.4f}, recall : {cur_class_recall:.4f}, f1 score : {cur_class_f1:.4f}')
        mean_ap = class_ap_sum / float(num_classes)
        mean_ap_sum += mean_ap
        avg_f1_score = class_f1_sum / float(num_classes)
        f1_sum += avg_f1_score
        print(f'F1@{int(iou_threshold * 100)} : {avg_f1_score:.4f}')
        print(f'mAP@{int(iou_threshold * 100)} : {mean_ap:.4f}\n')
    return mean_ap_sum / len(iou_thresholds)


if __name__ == '__main__':
    avg_mAP = calc_mean_average_precision(
        r'C:\inz\model\coco_2017_416_416_3.h5',
        glob(r'C:\inz\train_data\coco_2017\*.jpg'))
    print(f'avg mAP : {avg_mAP:.4f}')
