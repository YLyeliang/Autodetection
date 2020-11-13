# -*- coding: utf-8 -*- 
# @Time : 2020/11/11 2:17 下午 
# @Author : yl

from det.apis import init_detector, inference_detector
import argparse
import mtcv
from mtcv import color_val
import os.path as osp
import os
import numpy as np
import shutil
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--out_dir', default='./out', help='The output directory of image having objects.')
    parser.add_argument('--score_thr', default=0.5, help='Threshold for bboxes scores')

    parser.add_argument()
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpu-ids', type=int,
                            help='ids of gpus to use (only applicable to non-distributed training)')

    args = parser.parse_args()
    return args


args = parse_args()

font_scale = 0.5
thickness = 1
root = ""
dirs = ""

mtcv.mkdir_or_exist(args.out_dir)
# build the model from a config file and a checkpoint file
print('Load model...')
model = init_detector(args.config_file, args.checkpoint_file, device='cuda:0')

class_names = model.CLASSES
# get file lists.
img_generator = mtcv.scandir(root)

for img_file in img_generator:
    # result : a list, where length =num_classes, each one is bboxes of corresponding class
    img = mtcv.imread(osp.join(root, img_file))
    result = inference_detector(model, img)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > args.score_thr  # score_thr = 0.5
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    bbox_color = color_val('green')
    text_color = color_val('green')
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])

        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        img_out = osp.join(args.out_dir, img_file)
        mtcv.imwrite(img, img_out)
