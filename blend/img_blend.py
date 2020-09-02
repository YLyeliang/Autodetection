import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from mtcv.misc import histEqualize
from skimage.transform import PiecewiseAffineTransform, warp

blend_mode = {0: "naive", 1: "weight", 2: "poisson", 3: "multiply"}

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
