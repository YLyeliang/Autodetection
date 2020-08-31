import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from PIL import Image
import cv2

debug = True


def piecewiseAffineTrans(image, point_list):
    """
    Perform piecewise affine transformation on flags to fit the wave effect.
    Args:
        image: PIL image in mode RGBA.
        point_list: list(list[int])). four pairs of corner coordinates.

    Returns:
        warped logo and corresponding point_list.
    """
    w, h = image.size

    cols_point = 20
    rows_point = 10
    wave_num = np.random.randint(3, 6)

    # choose the number of points in rows and cols. generate the meshgrid,
    src_cols = np.linspace(0, w, cols_point)
    src_rows = np.linspace(0, h, rows_point)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]  # (x,y)

    # add sinusoidal oscillation to row coordinates
    factor = np.random.randint(h // 15, h // 10)

    # rows +[0,3*pi], which decides the wave.
    dst_rows = src[:, 1] - np.sin(np.linspace(0, wave_num * np.pi, src.shape[0])) * factor
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= factor * 1.5
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = int(h - 2 * factor)
    out_cols = w
    np_image = np.array(image)
    out = warp(np_image, tform, output_shape=(out_rows, out_cols), mode='constant', cval=0)
    out = out * 255
    out = out.astype(np.uint8)

    # get the actual logo coordinates, and clip it.
    y_id, x_id = np.where(out[..., 3] > 15)
    xmin, ymin = min(x_id), min(y_id)
    xmax, ymax = max(x_id), max(y_id)
    out = out[ymin:ymax + 1, xmin:xmax + 1, :]
    image = Image.fromarray(out)

    point_list[1], point_list[2], point_list[3] = [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]

    if debug:
        fig, ax = plt.subplots()
        ax.imshow(out)
        ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
        ax.axis((0, out_cols, out_rows, 0))
        plt.show()

    return image, point_list


def piecewiseAffineTransv2(image, point_list):
    """
    Perform piecewise affine transformation on flags to fit the wave effect.
    Args:
        image: PIL image in mode RGBA.
        point_list: list(list[int])). four pairs of corner coordinates.

    Returns:
        warped logo and corresponding point_list.
    """
    w, h = image.size

    cols_point = 20
    rows_point = 10
    wave_num = np.random.randint(3, 6)

    # choose the number of points in rows and cols. generate the meshgrid,
    src_cols = np.linspace(0, w, cols_point)
    src_rows = np.linspace(0, h, rows_point)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]  # (x,y)

    # add sinusoidal oscillation to row coordinates
    factor = np.random.randint(h // 15, h // 10)

    # rows +[0,3*pi], which decides the wave.
    dst_rows = src.copy()
    for i in range(10, 20):
        dst_rows[10 * i:10 * i + 5, 1] = dst_rows[10 * i + 9:-1:10 * 0 - 1, 1]

    dst_rows = src[:, 1] - np.random.uniform(0, 2, 50) * factor
    dst_cols = src[:, 0] - np.random.uniform(0, 2, 50) * factor
    dst_rows *= 1.5
    dst_rows -= factor * 1.5
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = int(h - 2 * factor)
    out_cols = w
    np_image = np.array(image)
    out = warp(np_image, tform, output_shape=(out_rows, out_cols), mode='constant', cval=0)
    out = out * 255
    out = out.astype(np.uint8)

    # get the actual logo coordinates, and clip it.
    y_id, x_id = np.where(out[..., 3] > 15)
    xmin, ymin = min(x_id), min(y_id)
    xmax, ymax = max(x_id), max(y_id)
    out = out[ymin:ymax + 1, xmin:xmax + 1, :]
    image = Image.fromarray(out)

    point_list[1], point_list[2], point_list[3] = [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]

    if debug:
        fig, ax = plt.subplots()
        ax.imshow(out)
        ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
        ax.axis((0, out_cols, out_rows, 0))
        plt.show()

    return image, point_list
