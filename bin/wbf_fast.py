import math
import os
from pathlib import Path

import pandas as pd

'''
Code taken and modified for 1D sequences from:
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py
'''
import warnings
import numpy as np


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]),
                                                                                                       len(scores[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = labels[t][j]
            box_part = boxes[t][j]

            x = float(box_part[0])
            y = float(box_part[1])

            # Box data checks
            if y < x:
                warnings.warn('Y < X value in box. Swap them.')
                x, y = y, x

            # [label, score, weight, model index, x, y]
            b = [label, float(score) * weights[t], weights[t], t, x, y]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x, y)
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1  # model index field is retained for consistensy but is not used.
    box[4:] /= conf
    return box


def find_matching_box_quickly(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.

        boxes_list: shape: (N, label, score, weight, model index, x, y)
        new_box: shape: (label, score, weight, model index, x, y)
    """

    def bb_iou_array(boxes, new_box):
        '''
        boxes: shape: (N, x, y)
        new_box: shape: (x, y)
        '''
        # bb interesection over union
        x_min = np.minimum(boxes[:, 0], new_box[0])
        x_max = np.maximum(boxes[:, 0], new_box[0])
        y_min = np.minimum(boxes[:, 1], new_box[1]) + 1
        y_max = np.maximum(boxes[:, 1], new_box[1]) + 1

        iou = np.maximum(0, (y_min - x_max) / (y_max - x_min))

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0,
                          conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 2 numbers.
     It has 3 dimensions (models_number, model_preds, 2)
     Order of boxes: x, y.
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x, y).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
                                                                                                     len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(
            conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 6))  ## [label, score, weight, model index, x, y]
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_quickly(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (
                    weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()

        # REQUIRE BBOX TO BE PREDICTED BY AT LEAST 2 MODELS
        # for i in range(len(new_boxes)):
        #    clustered_boxes = np.array(new_boxes[i])
        #    if len(np.unique(clustered_boxes[:, 3])) > 1:
        #        overall_boxes.append(weighted_boxes[i])

        overall_boxes.append(weighted_boxes)  # NOT NEEDED FOR "REQUIRE TWO MODELS" ABOVE
    overall_boxes = np.concatenate(overall_boxes, axis=0)  # NOT NEEDED FOR "REQUIRE TWO MODELS" ABOVE
    # overall_boxes = np.array(overall_boxes) # NEEDED FOR "REQUIRE TWO MODELS" ABOVE
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


csv_dir = Path("tmp/raw_preds")
output_dir = Path("tmp")

# Check if the directory exists
if not output_dir.exists():
    # If it doesn't exist, create it
    output_dir.mkdir()

longformer_csv = pd.read_csv(csv_dir / "longformerwithlstm.csv").dropna()
# deberta_v3_csv = pd.read_csv(csv_dir / "debertawithlstm.csv").dropna()
deberta_v2_csv = pd.read_csv(csv_dir / "deberta_v2.csv").dropna()
# debertaxl_csv = pd.read_csv(csv_dir / "debertaxl.csv").dropna()
# debertal_chris_csv = pd.read_csv(csv_dir / "debertal_chris.csv").dropna()
# debertal_csv = pd.read_csv(csv_dir / "debertal.csv").dropna()
yoso_csv = pd.read_csv(csv_dir / "yoso.csv").dropna()
funnel_csv = pd.read_csv(csv_dir / "funnel.csv").dropna()
bird_base_chris_csv = pd.read_csv(csv_dir / "bigbird_base_chris.csv").dropna()
lsg_csv = pd.read_csv(csv_dir / "lsg.csv").dropna()

TEST_DIR = 'data/competition_data/test'
test_files = os.listdir(TEST_DIR)
v_ids = [f.replace('.txt', '') for f in test_files]

class_to_label = {
    'Claim': 0,
    'Evidence': 1,
    'Lead': 2,
    'Position': 3,
    'Concluding Statement': 4,
    'Counterclaim': 5,
    'Rebuttal': 6
}

# Threshold found from CV
label_to_threshold = {
    0: 0.275,  # Claim
    1: 0.375,  # Evidence
    2: 0.325,  # Lead
    3: 0.325,  # Position
    4: 0.4,  # Concluding Statement
    5: 0.275,  # Counterclaim
    6: 0.275  # Rebuttal
}

label_to_class = {v: k for k, v in class_to_label.items()}


def preprocess_for_wbf(df_list):
    boxes_list = []
    scores_list = []
    labels_list = []

    for df in df_list:
        scores_list.append(df['scores'].values.tolist())
        labels_list.append(df['class'].map(class_to_label).values.tolist())
        predictionstring = df.predictionstring.str.split().values
        df_box_list = []
        for bb in predictionstring:
            df_box_list.append([int(bb[0]), int(bb[-1])])
        boxes_list.append(df_box_list)
    return boxes_list, scores_list, labels_list


def postprocess_for_wbf(idx, boxes_list, scores_list, labels_list):
    preds = []
    for box, score, label in zip(boxes_list, scores_list, labels_list):
        if score > label_to_threshold[label]:
            start = math.ceil(box[0])
            end = int(box[1])
            preds.append((idx, label_to_class[label], ' '.join([str(x) for x in range(start, end + 1)])))
    return preds


def generate_wbf_for_id(i):
    # df1 = debertal_csv[debertal_csv['id'] == i]
    # df2 = debertal_chris_csv[debertal_chris_csv['id'] == i]
    df3 = funnel_csv[funnel_csv['id'] == i]
    # df4 = debertaxl_csv[debertaxl_csv['id'] == i]
    df5 = longformer_csv[longformer_csv['id'] == i]
    # df6 = deberta_v3_csv[deberta_v3_csv['id'] == i]
    df7 = yoso_csv[yoso_csv['id'] == i]
    df8 = bird_base_chris_csv[bird_base_chris_csv['id'] == i]
    df9 = lsg_csv[lsg_csv['id'] == i]
    df10 = deberta_v2_csv[deberta_v2_csv['id'] == i]

    boxes_list, scores_list, labels_list = preprocess_for_wbf([
        # df1, df2,
        df3,  # df4,
        df5,
        # df6,
        df7, df8,
        df9, df10
    ])

    nboxes_list, nscores_list, nlabels_list = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.33,
                                                                    conf_type='avg')

    return postprocess_for_wbf(i, nboxes_list, nscores_list, nlabels_list)


import multiprocessing as mp

with mp.Pool(2) as p:
    list_of_list = p.map(generate_wbf_for_id, v_ids)

preds = [x for sub_list in list_of_list for x in sub_list]

sub = pd.DataFrame(preds)
sub.columns = ["id", "class", "predictionstring"]
sub.to_csv(output_dir / 'interim_submission.csv', index=False)
