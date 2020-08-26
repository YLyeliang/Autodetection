import argparse
import copy
import os
import os.path as osp
import time

import mtcv
import torch


from det.apis import
from det.dataset
from det.models import build_detector
from det.utils import get_root_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config',help='train config file path')
    parser.add_argument('--work-dir',help='the dir to save logs and models')
    parser.add_argument('--resume-from',help='the checkpoint file to resume from')
    parser.add_argument('--no-validate',action='store_true',help='whether not to evaluate the checkpoint during training')