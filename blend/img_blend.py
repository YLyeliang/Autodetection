import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from mtcv.misc import histEqualize
from skimage.transform import PiecewiseAffineTransform, warp

blend_mode = {0: "naive", 1: "weight", 2: "poisson", 3: "multiply"}


def edge_blur(src, point_list=None):
    """
    Blurring the edge of logo image.
    1. blur the whole image.
    2. create a mask covering the actual logo, and erode the mask as non-blur part of image.
    3. replace the blurred image at mask part with source logo image.
    Args:
        src:
        point_list: four paris of coordinates of actual logos.
    Returns:

    """

    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
    src_cp = src[:, :, :3]
    src_blur = cv2.GaussianBlur(src_cp, (13, 13), 0)
    # src_blur = cv2.GaussianBlur(src,(13,13),0)

    idx = np.where(src[:, :, 3] > 15)

    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    mask[idx] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dil = cv2.erode(mask, kernel)

    mask_id = np.where(dil > 0)
    src_blur[mask_id] = src_cp[mask_id]
    src[:, :, :3] = src_blur

    img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    debug = 1
    return img, point_list


def edge_virtual(src, point_list=None):
    """
    virutal the edge of image. Simply blur the alpha channels and keep actual logo region.
    1. create a alpha channel and copy blur the alpha copy.
    2. create a mask covering the actual logo and erode the mask.
    3. replacing the parts in blurred alpha with alpha according the mask region.
    Args:
        src:
        point_list:

    Returns:

    """
    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
    alpha = src[:, :, 3]
    alpha_blur = cv2.GaussianBlur(alpha, (7, 7), 0)

    idx = np.where(alpha > 15)
    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    mask[idx] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dil = cv2.erode(mask, kernel)
    mask_id = np.where(dil > 0)

    alpha_blur[mask_id] = alpha[mask_id]
    src[:, :, 3] = alpha_blur
    img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    return img, point_list


def edge_virtualv2(src, point_list=None, degree=3):
    """
    virutal the edge of image. Iteratively reduce the alpha value of image from outside to inside.
        src:
        point_list:

    Returns:

    """
    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
    xmax = np.amax(point_list, axis=0)[0]
    ymax = np.amax(point_list, axis=0)[1]
    xmin = np.amin(point_list, axis=0)[0]
    ymin = np.amin(point_list, axis=0)[1]
    widthTrans = xmax - xmin
    heightTrans = ymax - ymin
    side = min(widthTrans, heightTrans)

    # create initial mask covering the actual image.
    alpha = src[:, :, 3]
    alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1] = 0, 0, 0, 0
    idx = np.where(src[:, :, 3] > 15)
    mask = np.zeros(src.shape[:2], dtype=np.uint8)
    mask[idx] = 255
    mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1] = 0, 0, 0, 0
    # iteratively erode the mask and assigned the removed region to division factor in descend.
    if side < 100:
        factors = np.linspace(1.5, 1, 5)
    else:
        factors = np.linspace(2, 1, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for factor in factors:
        dil = cv2.erode(mask, kernel)
        diff1 = mask - dil
        mask = dil
        removed_loc = np.where(diff1 > 0)
        alpha[removed_loc] = alpha[removed_loc] // factor

    src[:, :, 3] = alpha
    img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    return img, point_list


def edge_blur_backup(src, point_list=None):
    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
    mask = np.zeros(src.shape[:2], dtype=np.float32)

    idx = np.where(src[:, :, 3] > 15)
    mask[idx] = 255

    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    mask = mask / 255.
    mask = mask[:, :, np.newaxis]

    src[:, :, :3] = src[:, :, :3] * mask
    img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    plt.figure()
    plt.imshow(img)
    plt.show()

    return img, point_list


def weight_paste(pixSrc, pixPng, src_id, logo_id):
    """
    a weight blend method, that corresponding parts in logo and source will be blend
    according to their alpha value.
    Args:
        pixSrc: (numpy array) source image in RGBA format.
        pixPng: (numpy array) logo image in RGBA format.
        srcH: height of source image.
        srcW: width of source image.
        x: the top-left coordinates of logo in the source image.
        y: the top-left coordinates of logo in the source image.

    Returns:

    """
    weight = pixPng[:, :, 3] / 255
    weight = weight[:, :, np.newaxis]
    alpha = weight[logo_id]
    beta = 1 - alpha
    pixSrc[src_id] = pixSrc[src_id] * beta + pixPng[logo_id] * alpha
    return pixSrc


def naive_paste(pixSrc, pixPng, src_id, logo_id):
    """
    a naive blend method, that simply paste the logo onto the source image.
    The logo image after transformed
    Args:
        pixSrc: (numpy array) source image in RGBA format.
        pixPng: (numpy array) logo image in RGBA format.
        srcH: height of source image.
        srcW: width of source image.
        x: the top-left coordinates of logo in the source image.
        y: the top-left coordinates of logo in the source image.
    Returns:

    """
    pixSrc[src_id] = pixPng[logo_id]
    return pixSrc


def poisson_blend(pixSrc, pixPng, src_id, logo_id, x, y):
    height, width = pixPng.shape[:2]
    mask = np.zeros(pixPng.shape[:2], dtype=np.uint8)

    p_x = x + width // 2 - 1
    p_y = y + height // 2 - 1

    mask[logo_id] = 255
    pixSrc = cv2.cvtColor(pixSrc, cv2.COLOR_RGBA2BGR)
    pixPng = cv2.cvtColor(pixPng, cv2.COLOR_RGBA2BGR)
    pixPng = histEqualize(pixPng, space='rgb', clipLimit=40)
    mixed = cv2.seamlessClone(pixPng, pixSrc, mask, (p_x, p_y), cv2.NORMAL_CLONE)

    return cv2.cvtColor(mixed, cv2.COLOR_BGR2RGBA)


def multiply(src, logo, src_id, logo_id):
    r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    l_r, l_g, l_b, l_a = logo[:, :, 0], logo[:, :, 1], logo[:, :, 2], logo[:, :, 3]
    rgb = [r, g, b]
    l_rgb = [l_r, l_g, l_b]
    for i in range(3):
        l_rgb[i][logo_id] = l_rgb[i][logo_id] / 255 * rgb[i][src_id]
    lo = np.stack(l_rgb + [l_a], axis=2)

    weight = l_a / 255
    weight = weight[:, :, np.newaxis]
    alpha = weight[logo_id]
    beta = 1 - alpha
    src[src_id] = src[src_id] * beta + lo[logo_id] * alpha
    return src


def channel_blend(pixSrc, pixPng, srcH, srcW, x, y, mode='weight', color_match=False):
    """
    Blend the source image with logo image at corresponding locations.
    Args:
        pixSrc: (ndarray)
        pixPng: (ndarray)
        srcH:
        srcW:
        x:
        y:

    Returns:

    """
    modes = [item for i, item in blend_mode.items()]
    # 1.find all indices satisfying conditions, and replace the value of indices in source image with logo image.
    # note: from pillow to numpy, (w,h) has converted to (h,w).
    index = np.where(pixPng[:, :, 3] > 15)
    y_id = index[0] + y - 1
    x_id = index[1] + x - 1

    # ensure the exceeding part remained in boundary.
    y_id = np.where(y_id >= srcH, srcH - 1, y_id)
    x_id = np.where(x_id >= srcW, srcW - 1, x_id)
    id = (y_id, x_id)

    # matching logo color with source image.
    if color_match:
        pixSrc_ = pixSrc.copy()[..., :3]
        pixPng_ = pixPng.copy()[..., :3]
        mean_source, stddev_source = cv2.meanStdDev(pixSrc_)
        mean_png, stddev_png = cv2.meanStdDev(pixPng_)
        mdiff = mean_png - mean_source
        mdiff = np.array(mdiff).reshape((1, 1, 3))
        pixPng_ = pixPng_.astype(np.float64)
        pixPng_ -= mdiff
        pixPng_ = np.clip(pixPng_, 0, 255)
        pixPng_ = pixPng_.astype(np.uint8)
        pixPng[..., :3] = pixPng_

    if mode not in modes: raise NotImplementedError(
        "only {0:'naive',1:'weight',2:'poisson',3:'multiply'} are supported.")
    if mode == 'weight':
        pixSrc = weight_paste(pixSrc, pixPng, id, index)
    elif mode == 'naive':
        pixSrc = naive_paste(pixSrc, pixPng, id, index)
    elif mode == 'poisson':
        pixSrc = poisson_blend(pixSrc, pixPng, id, index, x, y)
    elif mode == 'multiply':
        pixSrc = multiply(pixSrc, pixPng, id, index)

    return cv2.cvtColor(pixSrc, cv2.COLOR_RGBA2RGB)
