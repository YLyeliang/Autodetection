from det.apis import init_detector, inference_detector
import argparse
import mtcv
import os.path as osp
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='ids of gpus to use (only applicable to non-distributed training)')

    args = parser.parse_args()
    return args


def get_file_lists(root, dirs):
    """
    Get file list by given root directory and corresponding sub directory.
    Args:
        root: (str) root dir.
        dirs: (str| list |tuple) if list or tuple, means multiple directories.

    Returns:
        ids (list(str)): each str contains a image abs path.
    """
    if isinstance(dirs, (list, tuple)):
        path = [osp.join(root, dir) for dir in dirs]
    elif isinstance(dirs, str):
        path = osp.join(root, dirs)
    else:
        raise ValueError('dirs should be list or tuple or str, '
                         f'but got{type(dirs)}')
    ids = []
    if isinstance(path, str):
        imgs = os.listdir(path)
        for i in imgs:
            ids.append(osp.join(path, i))
    else:
        for folder in path:
            imgs = os.listdir(folder)
            for i in imgs:
                ids.append(osp.join(folder, i))
    return ids


def get_file_template(path, cls):
    filename = f'det__{cls}'
    if not osp.exists(path):
        os.makedirs(path)
    path = osp.join(path, filename)
    return path


def write_results_txt(path, all_bboxes):
    for class_name, file_bbox in all_bboxes.items():
        file = get_file_template(path, class_name)
        with open(file, 'wt') as f:
            for filename, bbox in file_bbox.items():
                f.write(f'{filename} {bbox[-1]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')


root = ''
img_path = ''

config_file = ''
checkpoint_file = ''

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

class_names = model.CLASSES
# get file lists.
img_list = get_file_lists(root, img_path)

all_bboxes = {}

# initialize the dict
for cls in class_names:
    all_bboxes[cls] = dict()

for img in img_list:
    # result : a list, where length =num_classes, each one is bboxes of corresponding class
    result = inference_detector(model, img)
    # TODO: waiting for debugging.
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > 0.01  # score_thr = 0.01
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        all_bboxes[label_text][img] = bbox
