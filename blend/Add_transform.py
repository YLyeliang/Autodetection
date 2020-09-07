import datetime
import os, sys
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from blend.transform import logoEffect
from blend.img_blend import channel_blend
import cv2
import matplotlib.pyplot as plt
import argparse

import random
import numpy as np


def args_arguments():
    parser = argparse.ArgumentParser(description="attching multiple logos or flags into the source image")
    parser.add_argument('--inputDir', type=str, default="/users/fiberhome/Downloads/source_data/source")
    parser.add_argument('--pngPath', type=str, default="/users/fiberhome/Downloads/source_data/png",
                        help="logo directory")
    parser.add_argument('--outputDir', type=str, default="/users/fiberhome/Downloads/source_data/blend",
                        help="output dir")
    parser.add_argument('--outTxtDir', type=str, default="/users/fiberhome/Downloads/source_data/txt",
                        help="directory of txt saving img name with locations")
    parser.add_argument('--class_name', type=str, default=None,
                        help='If specified, the class_name will be used to write output image file name.')
    parser.add_argument('--isResize', type=bool, default=True)
    parser.add_argument('--isaddNoise', type=bool, default=False)
    parser.add_argument('--isaddPerspective', type=bool, default=True)
    parser.add_argument('--isaddAffine', type=bool, default=True)
    parser.add_argument('--isVirtualEdge', type=bool, default=True)
    parser.add_argument('--isaddWave', type=bool, default=True)
    parser.add_argument('--blend_mode', type=str, default='weight')
    parser.add_argument('--locations', default=[None],
                        help=" the param used to specify the logo locations blended in the source image."
                             "If list(int),four paris of corner coordinates are specified. precise loc specified if list(list)"
                             "If None, then random.")
    return parser.parse_args()


def get_file_list(path):
    file_list = []
    for (root, dirs, files) in os.walk(path):
        if files:
            for i in files:
                file_path = os.path.join(root, i)
                file_list.append(file_path)
    return file_list


def is_inter(coord_1, coord_2):
    w = min(coord_1[2], coord_2[2]) - max(coord_1[0], coord_2[0])
    h = min(coord_1[3], coord_2[3]) - max(coord_1[1], coord_2[1])
    return 0 if w <= 0 or h <= 0 else 1  # 1 means overlap area existed and vice versa.


class addTransformation():
    """
    Perform all kinds of transformations on logo or flag images, then blending transformed image with source image.
    The whole pipeline is as followe:
    1、Given source and logo path, check the valid of path, and read images as a list from given path.
    2、perform transformations on logo images, which includes perspective, affine, normal augmentation, wave et al.
    3、blending source image and transformed image together using blending algorithm, like weight, poisson et al.
    4、write the image and corresponding logo location into dir and txt, respectively.
    Args:
        inputDir:
        pngPath:
        outputDir:
        outTxtDir:
        isResize:
        isaddNoise:
        isaddAffine:
        isaddPerspective:
        isVirtualEdge:
        isaddWave:
        blend_mode:
        locations:
        class_name: (str) If specify, the logoNames will be replaced by the class_name.
    """

    def __init__(self,
                 inputDir,
                 pngPath,
                 outputDir,
                 outTxtDir,
                 isaddWave,
                 locations,
                 isResize=True,
                 isaddNoise=False,
                 isaddAffine=True,
                 isaddPerspective=True,
                 isVirtualEdge=True,
                 blend_mode='weight',
                 class_name=None):

        self.inputDir = inputDir
        self.pngPath = pngPath
        self.outputDir = outputDir
        self.outTxtDir = outTxtDir
        self.isResize = isResize
        self.isaddNoise = isaddNoise
        self.isaddAffine = isaddAffine
        self.isaddPerspective = isaddPerspective
        self.isVirtualEdge = isVirtualEdge
        self.isaddWave = isaddWave
        self.blend_mode = blend_mode
        self.locations = locations
        self.logoSampleNumber = len(locations)
        self.class_name = class_name
        self.pathCheck()

    def pathCheck(self):
        """
        Check the valid of given dir.
        Returns:
        """
        if not os.path.exists(self.inputDir):
            raise FileNotFoundError("Error: inputDir is not correct.")
        if not os.path.exists(self.pngPath):
            raise FileNotFoundError("Error: pngPath is not correct.")
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        if not os.path.exists(self.outTxtDir):
            os.makedirs(self.outTxtDir)

    def specifyLogoLocation(self, location, srcH, srcW, height, width, heightTrans, widthTrans):
        """
        Given location, specify the top-left coordinates of logo.
        Args:
            location(int or tuple or None):if int(1,2,3,4), specify top-left, top-right, bottom-left, and bottom-right, respectively.
            if tuple, precise location specified. if None, random coords given.
            srcH: height of source image.
            srcW: ~
            height: height of logo image
            width: ~
            heightTrans: height of actual logo image ( because of affine or perspective).
            widthTrans: ~

        Returns: top-left coords of logo in source image.

        """
        if isinstance(location, int):
            if location == 1:
                x, y = 0, 0
            elif location == 2:
                x, y = int(srcW - width), 0
            elif location == 3:
                x, y = 0, int(srcH - height)
            elif location == 4:
                x, y = int(srcW - width), int(srcH - height)
            else:
                raise ValueError("location should be in [1,2,3,4]")
        elif isinstance(location, tuple) or isinstance(location, list):
            x, y = location

        else:
            x, y = random.randint(0, max(srcW - widthTrans, 0)), random.randint(0, max(0, srcH - heightTrans))
        return x, y

    def setPath(self, informations, write_multi=False, debug=False):
        """
                Given informations, parse the output image name, txt path, output image path.
        Args:
            informations (list(list)):each info in list contains: [iterations, idaddNoise,logoName]
            write_multi (bool): if True, write multi logo names on image name. else only first logo write.
            debug (bool): if True, create a new dir named by time to test.

        Returns:
            outImgName (str): the output image name
            outTxtpath (str): the output txt path
            outImgpath (str): the output image path
            logo_names (list(str)): the logo names.

        """
        if self.class_name is not None:
            logo_names = [self.class_name for info in informations]
        else:
            logo_names = [info[2].split('.')[0] for info in informations]
        if write_multi:
            outImgName = str(informations[0][0]) + "_" + "_".join(logo_names)
        else:
            outImgName = str(informations[0][0]) + "_" + logo_names[0]

        outTxtpath = os.path.join(self.outTxtDir, outImgName + '.txt')
        # debug: test
        if debug:
            output = os.path.join(self.outputDir,
                                  "{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M") + f'{self.blend_mode}'))
        else:
            output = self.outputDir
        if not os.path.exists(output): os.mkdir(output)
        outImgpath = os.path.join(output, outImgName + '.jpg')

        return outImgName, outTxtpath, outImgpath, logo_names

    def ImgOPS(self,
               srcImg,
               pngImg,
               isResize,
               isaddNoise,
               isaddPerspective,
               isaddAffine,
               isVirtualEdge,
               isaddWave,
               iters):
        """
        perform transformation on given logo image. Including monophonic processing, perspective and affine.
        the logo image converted to rgba, where alpha usually represents the opacity(不透明度）.
        Args:
            srcImg: (PIL format)
            pngImg: (PIL format)logo image, in mode RGBA.
            isResize: whether resize logo image
            isaddNoise: whether add noise on logo
            isaddPerspective: whether perform perspective warp on logo
            isaddAffine: whether perform affine warp on logo
            iters: number of current iterations.

        Returns:

        """
        info = [0, 0]
        srcImg = srcImg.convert("RGBA")
        (srcW, srcH) = srcImg.size
        (width, height) = pngImg.size

        if isResize:
            info[0] = iters

            # scale specifying minimum length of source divided by minimum length of logo,
            # using min here to guarantee the longest side of logo will be resized to smaller than the source.
            scale_w = srcW / width
            scale_H = srcH / height
            scale = min(scale_w, scale_H)

            # scale >1 means the w or h of source greater than logo, and vice versa.
            if scale > 1:
                scale1 = 1
                if 1 < scale < 2:
                    scale1 = random.uniform(0.2, 0.8)
                elif 2 <= scale < 4:
                    scale1 = random.uniform(0.3, 0.9)
                elif 4 <= scale < 6:
                    scale1 = random.uniform(0.4, 1)
                elif 6 <= scale < 10:
                    scale1 = random.uniform(0.5, 1.1)
                elif 10 <= scale:
                    scale1 = random.uniform(0.6, 3)
                pngImg = pngImg.resize((int(scale1 * width), int(scale1 * height)))
            else:
                scale1 = random.uniform(0.1, 0.4)
                leng_img = srcW if scale_w < scale_H else srcH
                pngImg = pngImg.resize((int(scale1 * leng_img), int(scale1 * leng_img)))

        if isaddNoise:
            info[1] = "M1"
            if random.randint(0, 1):
                random_angle = np.random.randint(-10, 10)
                mode = Image.BICUBIC
                pngImg = pngImg.rotate(random_angle, mode)
            if random.randint(0, 1):
                random_factor = np.random.randint(5, 35) / 10.
                pngImg = ImageEnhance.Color(pngImg).enhance(random_factor)  # color balance.
            if random.randint(0, 1):
                random_factor = np.random.randint(5, 15) / 10.
                pngImg = ImageEnhance.Brightness(pngImg).enhance(random_factor)  # brightness
            if random.randint(0, 1):
                random_factor = np.random.randint(5, 15) / 10.
                pngImg = ImageEnhance.Contrast(pngImg).enhance(random_factor)  # contrast
            if random.randint(0, 1):
                random_factor = np.random.randint(5, 15) / 10.
                pngImg = ImageEnhance.Sharpness(pngImg).enhance(random_factor)  # sharpness

        point_list = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        if isaddWave:
            rand_num = random.randint(0, 1)
            if rand_num == 1:
                pngImg, point_list = logoEffect.piecewiseAffineTrans(pngImg, point_list)
        if isaddPerspective:
            rand_num = random.randint(0, 1)
            if rand_num == 1:
                pngImg, point_list = logoEffect.Perspective(pngImg, point_list)
        if isaddAffine:
            rand_num = random.randint(0, 1)
            if rand_num == 1:
                pngImg, point_list = logoEffect.Affine(pngImg, point_list)
        if isVirtualEdge:
            pngImg, point_list = logoEffect.edge_virtual(pngImg, point_list)

        return pngImg, info, point_list

    def ImgBlend(self,
                 srcImg,
                 pngImgs,
                 outImgName,
                 outTxtpath,
                 outImgpath,
                 point_lists,
                 locations,
                 logo_names,
                 debug=False):
        """
        Given source image and a series of logo images, then blending two image together, and result will be saved in
        the out image path.
        Args:
            srcImg: source image, pillow format.
            pngImgs: logo image, a list of pillow format.
            outImgName(str): output image name
            outTxtpath(str): ~
            outImgpath(str):
            point_lists(list(list))): list of points, each points contain four or three actual corner coordinates of logo.
            locations(list or tuple): specifying the logo location.
            logo_names: a list of logo names.

        Returns:

        """
        assert len(pngImgs) == len(point_lists) == len(
            locations), "the length of logo iamges & point_lists & locations should equal."
        srcImg = srcImg.convert("RGBA")
        pixSrc = np.array(srcImg)
        srcW, srcH = srcImg.size

        # maximum or minimum coordinates of actual logo
        xmaxs = np.array([np.amax(point_list, axis=0)[0] for point_list in point_lists if point_list])
        ymaxs = np.array([np.amax(point_list, axis=0)[1] for point_list in point_lists if point_list])
        xmins = np.array([np.amin(point_list, axis=0)[0] for point_list in point_lists if point_list])
        ymins = np.array([np.amin(point_list, axis=0)[1] for point_list in point_lists if point_list])

        # actual shape of logo.
        widthTrans = xmaxs - xmins
        heightTrans = ymaxs - ymins

        logoCoord = []
        txt_content = ''

        for i in range(len(pngImgs)):

            width, height = pngImgs[i].size
            # get top-left of logo.
            x, y = self.specifyLogoLocation(locations[i], srcH, srcW, height, width, heightTrans[i], widthTrans[i])

            # check whether overlap exists in current logo among previous ones (actual location).
            coordinate = [x + xmins[i], y + ymins[i], x + xmins[i] + widthTrans[i], y + ymins[i] + heightTrans[i]]
            if len(logoCoord) >= 1:
                # if overlpa, try 50 times re-choose location.
                iters = 0
                while True:
                    ifoverlap = [is_inter(coordinate, checkArr) for checkArr in logoCoord]
                    if 1 in ifoverlap:
                        x, y = self.specifyLogoLocation(locations[i], srcH, srcW, height, width, heightTrans[i],
                                                        widthTrans[i])
                        coordinate = [x + xmins[i], y + ymins[i], x + xmins[i] + widthTrans[i],
                                      y + ymins[i] + heightTrans[i]]
                    else:
                        break
                    iters += 1
                    if iters == 49:
                        break
                # pass this logo
                if iters == 49: continue

            logoCoord.append(coordinate)

            # check if the logo exceed the border.
            if coordinate[2] - xmins[i] > srcW or coordinate[3] - ymins[i] > srcH:
                continue

            # blend image
            pixPng = np.array(pngImgs[i])
            pixSrc = channel_blend(pixSrc, pixPng, srcH, srcW, x, y, mode='weight')

            # write txt content.
            txt_content += f' {logo_names[i]} 0 0 {coordinate[0]} {coordinate[1]} {coordinate[2]} {coordinate[3]}'

        # debug: show logo box.
        if debug:
            for coord in logoCoord:
                pixSrc = cv2.rectangle(pixSrc, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)

        # write image and txt file.
        # ensure there are logo in images.
        if not txt_content == '':
            srcImg = Image.fromarray(pixSrc)
            srcImg = srcImg.convert("RGB")
            srcImg.save(outImgpath)

            f = open(outTxtpath, 'w+')
            f.write('{}.jpg'.format(outImgName) + txt_content)
            f.close()

    def get_logo_files(self, pngPath):
        """
        Each pngPath may contain 2 or 3 or 0 subdir, named with 'logo' 'with_wave' 'no_wave',
        read all files from these subdir to a single list, and preserve index range[i,j]
        corresponding to the subdir to decide the effect number in ImgOPS phase.
        Args:
            pngPath:

        Returns:

        """
        choices = ['logo', 'no_wave', 'with_wave']
        root, dirs, files = os.walk(pngPath).__next__()
        for dir in dirs:
            if not dir in choices:
                raise ValueError(f'Only {choices} subdirs are supported, but got {dir}')
        no_wave = []
        with_wave = []
        if len(dirs) == 2:
            no_wave = os.listdir(os.path.join(pngPath, 'no_wave'))
            with_wave = os.listdir(os.path.join(pngPath, 'with_wave'))
            logo = files
        elif len(dirs) == 3:
            no_wave = os.listdir(os.path.join(pngPath, 'no_wave'))
            with_wave = os.listdir(os.path.join(pngPath, 'with_wave'))
            logo = files + os.listdir(os.path.join(pngPath, 'logo'))
        else:
            logo = files

        logo_range, no_wave_range, with_wave_range = len(logo), len(no_wave) + len(logo), len(with_wave) + len(
            no_wave) + len(logo)

        return logo + no_wave + with_wave, [logo_range, no_wave_range, with_wave_range]

    def random_png(self, logo_files, indexes, num=1):
        logo_range, no_wave_range, with_wave_range = indexes
        ids = random.sample([i for i in range(len(logo_files))], num)
        logos = []
        kwargs = []
        for id in ids:
            if id < logo_range:
                kwarg = dict(isaddWave=False)
            elif id < no_wave_range:
                kwarg = dict(isaddWave=False)
            else:
                kwarg = dict(isaddWave=True)
            logos.append(logo_files[id])
            kwargs.append(kwarg)
        return logos, kwargs

    def generate_blended_images(self):
        """
        the operator to iteratively generate blended images.
        Returns:

        """

        files = get_file_list(self.inputDir)
        # pngFiles = os.listdir(self.pngPath)
        pngFiles, indexs = self.get_logo_files(self.pngPath)
        random.shuffle(files)

        for n, file in enumerate(files):
            if n == 100: break
            try:
                srcImg = Image.open(file)
                srcImg = srcImg.convert('RGBA')
            except:
                print("file {} can not open".format(file))
                continue
            if 150 <= srcImg.size[0] and 150 <= srcImg.size[1]:
                print(f'file {file} is processing.')
                # logofiles = random.sample(pngFiles, self.logoSampleNumber)
                logofiles, kwargs = self.random_png(pngFiles, indexs, self.logoSampleNumber)
                logoImgs = [Image.open(os.path.join(self.pngPath, logoFile)) for logoFile in logofiles]
                for logo_id in range(len(logoImgs)):
                    if logoImgs[logo_id].mode != "RGBA":
                        logoImgs[logo_id] = logoImgs[logo_id].convert("RGBA")

                logoImgAugs = []
                point_lists = []
                informations = []
                for (logoImg, logoName, kwarg) in zip(logoImgs, logofiles, kwargs):
                    logoImgAug, info, point_list = self.ImgOPS(srcImg, logoImg, self.isResize, self.isaddNoise,
                                                               self.isaddPerspective, self.isaddAffine,
                                                               self.isVirtualEdge, iters=n, **kwarg)
                    logoImgAugs.append(logoImgAug)
                    point_lists.append(point_list)
                    info.append(logoName)
                    informations.append(info)
                outImgName, outTxtpath, outImgpath, logo_names = self.setPath(informations)
                self.ImgBlend(srcImg, logoImgAugs, outImgName, outTxtpath, outImgpath, point_lists, self.locations,
                              logo_names, debug=False)


if __name__ == '__main__':
    args = args_arguments()
    arguments = dict(inputDir=args.inputDir,
                     pngPath=args.pngPath,
                     outputDir=args.outputDir,
                     outTxtDir=args.outTxtDir,
                     isResize=args.isResize,
                     isaddNoise=args.isaddNoise,
                     isaddAffine=args.isaddAffine,
                     isaddPerspective=args.isaddPerspective,
                     isVirtualEdge=args.isVirtualEdge,
                     isaddWave=args.isaddWave,
                     blend_mode=args.blend_mode,
                     locations=args.locations,
                     class_name=args.class_name)
    Transformer = addTransformation(**arguments)
    Transformer.generate_blended_images()
