import os
from concurrent.futures.thread import ThreadPoolExecutor

from cv2 import cv2
from tqdm import tqdm

from yolo import Yolo


class F1Calculator:
    def __init__(self, model_paths, image_paths, confidence_threshold=0.25, iou_threshold=0.6):
        self.__model_paths = model_paths
        self.__image_paths = image_paths
        self.__iou_threshold = iou_threshold
        self.__confidence_threshold = confidence_threshold
        self.__pool = ThreadPoolExecutor(16)

    def calculate(self):
        results = []
        for cur_model_path in self.__model_paths:
            cur_model_path = cur_model_path.replace('\\', '/')
            model = Yolo(pretrained_model_path=cur_model_path)
            input_shape = model.get_input_shape()
            f1_score_sum = 0.0

            label_fs = []
            for path in self.__image_paths:
                label_fs.append(self.__pool.submit(self.__load_label, path))

            img_fs = []
            for f in label_fs:
                path, label_lines = f.result()
                if path is not None:
                    img_fs.append(self.__pool.submit(self.__load_img, path, input_shape[-1], label_lines))

            for f in tqdm(img_fs):
                img, label_lines = f.result()
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
            results.append({
                'score': avg_f1_score,
                'model_path': cur_model_path})
            print(f'\nf1 : {avg_f1_score:.4f} model path : {cur_model_path}')

        print('\n\ntotal results')
        results = sorted(results, key=lambda x: x['score'])
        total_result_str = ''
        for cur_result in results:
            cur_result_str = f'score : {cur_result["score"]:.4f}, model_path : {cur_result["model_path"]}'
            total_result_str += cur_result_str + '\n'
            print(cur_result_str)

        with open('f1_result.txt', mode='wt') as f:
            f.write(total_result_str)

    @staticmethod
    def __load_label(path):
        label_path = f'{path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            return None, None
        with open(label_path, 'rt') as f:
            return path, f.readlines()

    @staticmethod
    def __load_img(path, channels, label_lines):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR), label_lines

    def __calculate_iou_f1_score(self, y_true, y_pred):
        tp = 0
        for i in range(len(y_true)):
            for j in range(len(y_pred)):
                if y_pred[j]['discard'] or y_true[i]['class'] != y_pred[j]['class']:
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
