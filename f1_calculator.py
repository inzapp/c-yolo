import os

from cv2 import cv2
from tqdm import tqdm

from yolo import Yolo


class F1Calculator:
    def __init__(self, model_paths, image_paths, confidence_threshold=0.25, iou_threshold=0.6):
        self.__model_paths = model_paths
        self.__image_paths = image_paths
        self.__iou_threshold = iou_threshold
        self.__confidence_threshold = confidence_threshold

    def calculate(self):
        for cur_model_path in self.__model_paths:
            cur_model_path = cur_model_path.replace('\\', '/')
            model = Yolo(pretrained_model_path=cur_model_path)
            input_shape = model.get_input_shape()
            f1_score_sum = 0.0
            for cur_image_path in tqdm(self.__image_paths):
                label_path = f'{cur_image_path[:-4]}.txt'
                if not (os.path.exists(label_path) and os.path.isfile(label_path)):
                    continue
                with open(label_path, 'rt') as f:
                    label_lines = f.readlines()
                img = cv2.imread(cur_image_path, cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
                raw_height, raw_width = img.shape[0], img.shape[1]
                y_true = []
                for label_line in label_lines:
                    class_index, cx, cy, w, h = list(map(float, label_line.split(' ')))
                    x1 = int((cx - w / 2.0) * raw_width)
                    x2 = int((cx + w / 2.0) * raw_width)
                    y1 = int((cy - h / 2.0) * raw_height)
                    y2 = int((cy + h / 2.0) * raw_height)
                    y_true.append({
                        'class': int(class_index),
                        'bbox': [x1, y1, x2, y2],
                        'discard': False
                    })
                y_pred = model.predict(img=img, confidence_threshold=self.__confidence_threshold)
                f1_score = self.__calculate_iou_f1_score(y_true, y_pred)
                f1_score_sum += f1_score
            avg_f1_score = f1_score_sum / len(self.__image_paths)
            print(f'\nf1 : {avg_f1_score:.4f} model path : {cur_model_path}')

    def __calculate_iou_f1_score(self, y_true, y_pred):
        tp = 0
        for i in range(len(y_true)):
            for j in range(len(y_pred)):
                if y_pred[j]['discard']:
                    continue
                if self.__iou(y_true[i]['bbox'], y_pred[j]['bbox']) > self.__iou_threshold:
                    y_true[i]['discard'] = True
                    y_pred[j]['discard'] = True
                    tp += 1
                    break

        fp = 0
        for i in range(len(y_pred)):
            if not y_pred[i]['discard']:
                fp += 1

        fn = 0
        for i in range(len(y_true)):
            if not y_true[i]['discard']:
                fn += 1

        """
        precision = True Positive / True Positive + False Positive
        precision = True Positive / All Detections
        """
        p = tp / float(tp + fp + 1e-5)

        """
        recall = True Positive / True Positive + False Negative
        recall = True Positive / All Ground Truths
        """
        r = tp / float(tp + fn + 1e-5)

        """
        Harmonic mean of precision and recall.
        f1 = 1 / (precision^-1 + precision^-1) * 0.5
        f1 = 2 * precision * recall / (precision + recall)
        """
        return (p * r * 2.0) / (p + r + 1e-5)

    @staticmethod
    def __iou(a, b):
        """
        Intersection of union function.
        :param a: [x_min, y_min, x_max, y_max] format box a
        :param b: [x_min, y_min, x_max, y_max] format box b
        """
        a_x_min, a_y_min, a_x_max, a_y_max = a
        b_x_min, b_y_min, b_x_max, b_y_max = b
        intersection_width = min(a_x_max, b_x_max) - max(a_x_min, b_x_min)
        intersection_height = min(a_y_max, b_y_max) - max(a_y_min, b_y_min)
        if intersection_width < 0.0 or intersection_height < 0.0:
            return 0.0
        intersection_area = intersection_width * intersection_height
        a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
        b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
        union_area = a_area + b_area - intersection_area
        return intersection_area / float(union_area)
