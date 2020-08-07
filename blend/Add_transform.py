import datetime
import os, sys
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from blend.AffineTrans import Affine
from blend.PerspectiveTrans import Perspective
from blend.img_blend import paste,edge_blur,poisson_blend,channel_blend,naive_paste
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
    parser.add_argument('--isResize', type=bool, default=True)
    parser.add_argument('--isaddNoise', type=bool, default=False)
    parser.add_argument('--isaddPerspective', type=bool, default=True)
    parser.add_argument('--isaddAffine', type=bool, default=True)
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


class addTransformation:
    def __init__(self, args):
        self.inputDir = args.inputDir
        self.pngPath = args.pngPath
        self.outputDir = args.outputDir
        self.outTxtDir = args.outTxtDir
        self.isResize = args.isResize
        self.isaddNoise = args.isaddNoise
        self.isaddAffine = args.isaddAffine
        self.isaddPerspective = args.isaddPerspective
        self.locations = args.locations
        self.logoSampleNumber = len(args.locations)
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
            x, y = random.randint(0, max(srcW - widthTrans,0)), random.randint(0, max(0,srcH - heightTrans))
        return x, y

    def setPath(self, informations):
        """
        Given informations, parse the output image name, txt path, output image path.
        Args:
            informations(list(list)):each info in list contains: [iterations, idaddNoise,logoName]

        Returns:

        """
        logo_names = [info[2][:-4] for info in informations]
        outImgName = str(informations[0][0]) + "_" + "_".join(logo_names) + "_{}".format(
            datetime.datetime.now().strftime("%Y%m%d"))
        outTxtpath = os.path.join(self.outTxtDir, outImgName + '.txt')
        # debug: test
        output = os.path.join(self.outputDir,"{}".format(datetime.datetime.now().strftime("%Y%m%d%M")))
        if not os.path.exists(output): os.mkdir(output)
        outImgpath = os.path.join(output, outImgName + '.jpg')
        # outImgpath = os.path.join(self.outputDir, outImgName + '.jpg')

        return outImgName, outTxtpath, outImgpath, logo_names

    def ImgOPS(self,
               srcImg,
               pngImg,
               isResize,
               isaddNoise,
               isaddPerspective,
               isaddAffine,
               iters):
        """
        perform transformation on given logo image. Including monophonic processing, perspective and affine.
        the logo image converted to rgba, where alpha usually represents the opacity(不透明度）.
        Args:
            srcImg:
            pngImg: logo image
            isResize:
            isaddNoise:
            isaddPerspective:
            isaddAffine:
            iters:

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
                    scale1 = random.uniform(0.4, 1)
                elif 2 <= scale < 4:
                    scale1 = random.uniform(0.4, 1.2)
                elif 4 <= scale < 6:
                    scale1 = random.uniform(0.5, 1.2)
                elif 6 <= scale < 10:
                    scale1 = random.uniform(0.8, 1.68)
                elif 10 <= scale:
                    scale1 = random.uniform(0.8, 3)
                pngImg = pngImg.resize((int(scale1 * width), int(scale1 * height)))
            else:
                scale1 = random.uniform(0.2, 0.5)
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
        if isaddPerspective:
            pngImg, point_list = Perspective(pngImg, point_list)
        if isaddAffine:
            pngImg, point_list = Affine(pngImg, point_list)

        # debug:
        pngImg, point_list =edge_blur(pngImg,point_list)

        return pngImg, info, point_list

    def ImgBlend(self,
                 srcImg,
                 pngImgs,
                 outImgName,
                 outTxtpath,
                 outImgpath,
                 point_lists,
                 locations,
                 logo_names):
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
            # pixSrc= paste(pixSrc,pixPng,srcH,srcW,x,y)
            naive_paste(pixSrc,pixPng,srcH,srcW,x,y)
            # pixSrc= poisson_blend(pixSrc,pixPng,srcH,srcW,x,y)
            # pixSrc= channel_blend(pixSrc,pixPng,srcH,srcW,x,y)



        # debug: show logo box.
        for coord in logoCoord:
              pixSrc =cv2.rectangle(pixSrc,(coord[0],coord[1]),(coord[2],coord[3]),(0,255,0),2)

        # write image and txt file.
        srcImg = Image.fromarray(pixSrc)
        srcImg = srcImg.convert("RGB")
        srcImg.save(outImgpath)

        f = open(outTxtpath, 'w+')
        f.write('{}.jpg'.format(outImgName) + txt_content)
        f.close()

    def generate_blended_images(self):
        """
        the operator to iteratively generate blended images.
        Returns:

        """
        files = get_file_list(self.inputDir)
        pngFiles = os.listdir(self.pngPath)
        random.shuffle(files)

        for n, file in enumerate(files):
            if n == 30: break
            try:
                srcImg = Image.open(file)
            except:
                print("file {} can not open".format(file))
                continue
            if 150 <= srcImg.convert("RGBA").size[0] and 150 <= srcImg.convert("RGBA").size[1]:
                logofiles = random.sample(pngFiles, self.logoSampleNumber)
                logoImgs = [Image.open(os.path.join(self.pngPath, logoFile)) for logoFile in logofiles]
                for logo_id in range(len(logoImgs)):
                    if logoImgs[logo_id].mode != "RGBA":
                        logoImgs[logo_id] = logoImgs[logo_id].convert("RGBA")

                logoImgAugs = []
                point_lists = []
                informations = []
                for (logoImg, logoName) in zip(logoImgs, logofiles):
                    logoImgAug, info, point_list = self.ImgOPS(srcImg, logoImg, self.isResize, self.isaddNoise,
                                                               self.isaddPerspective, self.isaddAffine, n)
                    logoImgAugs.append(logoImgAug)
                    point_lists.append(point_list)
                    info.append(logoName)
                    informations.append(info)
                outImgName, outTxtpath, outImgpath, logo_names = self.setPath(informations)
                self.ImgBlend(srcImg, logoImgAugs, outImgName, outTxtpath, outImgpath, point_lists, self.locations,
                              logo_names)


if __name__ == '__main__':
    args = args_arguments()
    Transformer = addTransformation(args)
    Transformer.generate_blended_images()
