import os.path as osp
import torch.utils.data as data
import numpy as np
import mtcv

import xml.etree.ElementTree as ET

from .builder import DATASETS
from PIL import Image
from .pipelines import Compose


@DATASETS.register_module()
class TBDataset(data.Dataset):
    CLASSES = {}

    def __init__(self,
                 data_root,
                 pipeline,
                 data_dirs=[],
                 min_size=None,
                 img_prefix='',
                 test_mode=False,
                 filter_empty_gt=True):
        self.data_root = data_root
        self.data_dirs = data_dirs
        self.img_prefix = img_prefix
        self.min_size = min_size
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        self.data_infos = self.load_annotations()
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}

        # filter images too small
        if not self.test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def load_annotations(self):
        """
        Loading image information by txt file from multi directories by self.data_dirs.

        self.data_dirs: (list(str)) a list contains data dirs, each one is a dir str.

        Returns:
            list[dict]: image info from txt file.
        """
        data_infos = []
        for folder in self.data_dirs:
            rootpath = osp.join(self.data_root, folder)
            txt_file = osp.join(rootpath, 'ImageSets', 'Main', 'trainval.txt')
            img_ids = mtcv.list_from_file(txt_file)
            for img_id in img_ids:
                filename = f'{folder}/JPEGImages/{img_id}.jpg'
                xml_path = osp.join(self.data_root, folder, 'Annotations', f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                width = 0
                height = 0
                if size is not None:
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                else:
                    img_path = osp.join(self.data_root, filename)
                    img = Image.open(img_path)
                    width, height = img.size
                data_infos.append(dict(id=img_id, folder=folder, filename=filename, width=width, height=height))
        return data_infos

    def get_ann_info(self, idx):
        """
        Get annotation from XML file by index.
        Args:
            idx: (int) Index of data.

        Returns:
            dict: Annotation info o specified index.
        """
        img_id = self.data_infos[idx]['id']
        folder = self.data_infos[idx]['folder']
        xml_path = osp.join(self.data_root, folder, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.find('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """
        Get category ids in XML file by index.
        Args:
            idx: (int) Index of data

        Returns:
            list[int]: All categories in the image of specified index.
        """
        cat_ids = []
        img_id = self.data_infos[idx]['id']
        folder = self.data_infos[idx]['folder']
        xml_path = osp.join(self.data_root, folder, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)
        return cat_ids

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
