import os
import cv2
import os.path as osp
import shutil
import argparse
import codecs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Given a dir contains "merged_imgs" and "labels", '
                    'write xml annotations and trainval.txt of VOC format by given txt files.')
    parser.add_argument('--input_dir', help='input dir waiting for tranformed.')
    parser.add_argument('--img_dir', default=None,
                        help='Default is None, which means "merged_imgs" are existed in the dir. If given, '
                             'the specified dir will be used.')
    parser.add_argument('--txt_dir', default=None,
                        help='Default is None, which means "labels" are existed in the dir. If given, the specified '
                             'dir will be used')

    args = parser.parse_args()
    return args


class DataTransform(object):
    """
    Given a dir contains "merged_imgs" and "labels",
    write xml annotations and trainval.txt of VOC format by given txt files.

    Args:
        input_dir (str): input dir waiting for tranformed.
        img_dir (str): Default is None, which means "merged_imgs" are existed in the dir.
        If given, the specified dir will be used.
        txt_dir (str): Default is None, which means "labels" are existed in the dir.
        If given, the specified dir will be used.
    """

    def __init__(self,
                 input_dir,
                 img_dir=None,
                 txt_dir=None):
        self.input_dir = input_dir
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.init_dirs()

    def init_dirs(self):
        if self.img_dir is None:
            merged_images = osp.join(self.input_dir, 'merged_imgs')
        else:
            merged_images = osp.abspath(self.img_dir)
        JPEG_images = osp.join(self.input_dir, 'JPEGImages')
        os.rename(merged_images, JPEG_images)
        self.img_dir = JPEG_images
        annot_path = osp.join(self.input_dir, 'Annotations')
        imagesets = osp.join(self.input_dir, 'ImageSets', 'Main')
        if not osp.exists(annot_path):
            os.makedirs(annot_path)
        if not osp.exists(imagesets):
            os.makedirs(imagesets)
        self.annot_dir = annot_path
        self.trainval = osp.join(imagesets, 'trainval.txt')
        if self.txt_dir is None:
            self.txt_dir = osp.join(self.input_dir, 'labels')
        if not self.txt_dir: raise FileExistsError(f'the "labels" are not found in the {self.input_dir}, '
                                                   f'you should try given --txt_dir or rename txt dir.')

    def tranformFormat(self):
        txt_list = os.listdir(self.txt_dir)
        trainval = open(self.trainval, 'w')
        for txt_name in txt_list:
            if '.txt' not in txt_name:
                continue
            txt_file = osp.join(self.txt_dir, txt_name)
            f = open(txt_file)

            line = f.readline()
            line = line.strip().split(' ')
            print(f'file {txt_name} is processed.')
            img = cv2.imread(self.img_dir, line[0])
            try:
                sp = img.shape
            except:
                print(f'image {line[0]} has no shape.')
                continue
            height = sp[0]
            width = sp[1]
            depth = sp[2]
            file_name = line[2].split('.')[0]
            logos = line[1::7]
            xmins = line[4::7]
            ymins = line[5::7]
            xmaxs = line[6::7]
            ymaxs = line[7::7]
            trainval.writelines(file_name + '\n')
            with codecs.open(osp.join(self.annot_dir, file_name + '.xml'), 'w', 'utf-8') as xml:
                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'fh_SSDdata' + '</folder>\n')
                xml.write('\t<filename>' + line[0] + '</filename>\n')
                xml.write('\t<size>\n')
                xml.write(f'\t\t<width>{width}</width>\n')
                xml.write(f'\t\t<height>{height}</height>\n')
                xml.write(f'\t\t<depth>{depth}</depth>\n')
                xml.write('\t</size>\n')
                for i in range(len(xmins)):
                    xml.write('\t<object>\n')
                    xml.write(f'\t\t<name>{logos[i]}</name>\n')
                    xml.write(f'\t\t<bndbox>\n')
                    xml.write(f'\t\t\t<xmin>{xmins[i]}</xmin>\n')
                    xml.write(f'\t\t\t<ymin>{ymins[i]}</ymin>\n')
                    xml.write(f'\t\t\t<xmax>{xmaxs[i]}</xmax>\n')
                    xml.write(f'\t\t\t<ymax>{ymaxs[i]}</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
                xml.write('\t</annotation>')
        trainval.close()
