import random

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from PIL import Image
import cv2
from .img_blend import hardLight


class logoEffect(object):
    """
    The input of image should be format PIL.Image with shape (W,H,4) where 4 represents RGBA.
    Each function will process the image, the inter process may change the image style to BGRA or else,
    but eventually function will return a PIL.Image with shape (W,H,4) and a list (point_list), which
    represents the four coordinates of real logo images ( which means the min covered region of alpha !=0)
    """

    @staticmethod
    def piecewiseAffineTrans(image, point_list, debug=False):
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

        # add sinusoidal oscillation to row coordinates. Ensure low < high
        factor = np.random.randint(h // 15, max(h // 10, (h // 15) + 1))

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

    @staticmethod
    def Perspective(src, point_list=None):
        """
        Perspective transformation of image
        Args:
            src: source image, should be RGBA or BGR
            point_list:

        Returns:

        """
        src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
        w_scale = (random.uniform(0, 0.3), random.uniform(0, 0.3))
        h_scale = (random.uniform(0, 0.3), random.uniform(0, 0.3))
        height, width = src.shape[:2]
        shift_range_h = int(height * h_scale[0] + width * h_scale[1])
        shift_range_w = int(height * w_scale[0] + width * w_scale[1])
        src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        dst_points = src_points.copy()

        if shift_range_w > 0 or shift_range_h > 0:
            if shift_range_w > 0:
                for i in range(4):
                    dst_points[i][0] += np.random.randint(0, shift_range_w, 1)

            if shift_range_h > 0:
                for i in range(4):
                    dst_points[i][1] += np.random.randint(0, shift_range_h, 1)

            # calculate the perspective transformation matrix.
            # [t_i*x_i,t_i*y_i,t_i] = map_matrix*[x_i,y_i,1] where i=0,1,2,3 related to four pairs of corresponding coordinates.
            perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)

            i_min = 0
            i_max = width - 1
            j_min = 0
            j_max = height - 1
            for point in dst_points:
                i_min = min(i_min, point[0])
                i_max = max(i_max, point[0])
                j_min = min(j_min, point[1])
                j_max = max(j_max, point[1])
            for ii in range(3):
                perspective_mat[0][ii] -= perspective_mat[2][ii] * i_min
                perspective_mat[1][ii] -= perspective_mat[2][ii] * j_min

            width_new = int(i_max - i_min)
            height_new = int(j_max - j_min)

            dst = cv2.warpPerspective(src, perspective_mat, (width_new, height_new), borderValue=(0, 0, 0, 0))

            img = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGRA2RGBA))

            # debug:
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(img)
            # plt.show()

            return img, np.int32(dst_points).tolist()
        else:
            img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
        return img, point_list

    @staticmethod
    def Affine(src, point_list=None):
        """
        Affine transformation.
        Args:
            src: source image, in BGR or RGBA
            point_list: list used to preserve actual corner coord of logo

        Returns:

        """
        src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
        # decide the scale factor.
        w_scale = (random.uniform(0, 0.05), random.uniform(0, 0.05))
        h_scale = (random.uniform(0, 0.05), random.uniform(0, 0.05))
        height, width = src.shape[:2]
        shift_range_height = int(height * h_scale[0] + width * h_scale[1])
        shift_range_width = int(height * w_scale[0] + width * w_scale[1])

        src_points = np.float32([[0, 0], [width, 0], [0, height]])
        dst_points = src_points.copy()

        if shift_range_width > 0 or shift_range_height > 0:
            if shift_range_width > 0:
                for i in range(3):
                    dst_points[i][0] += np.random.randint(0, shift_range_width, 1)
            if shift_range_height > 0:
                for i in range(3):
                    dst_points[i][1] += np.random.randint(0, shift_range_height, 1)
            #  will create 2x3 matrix。 [x2,y2,b]=[A|b]*[x1,y1,1]
            perspective_mat = cv2.getAffineTransform(src_points, dst_points)
            i_min = 0
            i_max = width - 1
            j_min = 0
            j_max = height - 1
            bottom_right_x = int(
                perspective_mat[0][0] * (width - 1) + perspective_mat[0][1] * (height - 1) + perspective_mat[0][2])
            bottom_right_y = int(
                perspective_mat[1][0] * (width - 1) + perspective_mat[1][1] * (height - 1) + perspective_mat[1][2])
            bottom_right = np.array([[bottom_right_x, bottom_right_y]])
            dst_points = np.r_[dst_points, bottom_right]
            for point in dst_points:
                i_min = min(i_min, point[0])
                i_max = max(i_max, point[0])
                j_min = min(j_min, point[1])
                j_max = max(j_max, point[1])
            perspective_mat[0][2] -= i_min
            perspective_mat[1][2] -= j_min
            width_new = int(i_max - i_min)
            height_new = int(j_max - j_min)
            dst = cv2.warpAffine(src, perspective_mat, (width_new, height_new), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
            img = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGRA2RGBA))

            # debug:
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(img)
            # plt.show()

            return img, np.int32(dst_points).tolist()
        else:
            img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
        return img, point_list

    @staticmethod
    def edge_virtual(src, point_list=None, degree=3):
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
            factors = np.linspace(1.5, 1, degree)
        else:
            factors = np.linspace(2, 1, 2 * degree)
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

    @staticmethod
    def RandomFlip(src, point_list=None, direction='horizontal'):
        """
        Perform random horizontal or vetical flip of image.
        Args:
            src:
            point_list:
            direction:

        Returns:

        """
        assert direction in ['horizontal', 'vertical']
        src = np.asarray(src)
        if direction == 'vertical':
            flip = np.flip(src, axis=1)
            src = Image.fromarray(flip)
            return src, point_list
        else:
            flip = np.flip(src, axis=0)
            src = Image.fromarray(flip)
            return src, point_list

    @staticmethod
    def Randomrotate(src, point_list=None, border_value=(0, 0, 0, 0), auto_bound=True):
        """
        Perform random rotate of image.
        Args:
            src:
            point_list:

        Returns:

        """
        # TODO: To add point_list calculation and debug.
        src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
        h, w = src.shape[:2]
        angle = np.random.randint(-90, 90)
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, scale=1.0)
        dst_points = point_list.copy()
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
            dst_points = [[0, 0], [w, 0], [0, h], [w, h]]
        rotated = cv2.warpAffine(src, matrix, (w, h), borderValue=border_value)
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGRA2RGBA)), dst_points

    @staticmethod
    def flagTexture(flag, texture):
        flag = np.asarray(flag)
        dst = hardLight(texture, flag)
        return Image.fromarray(dst)


def piecewiseAffineTransv2(image, point_list, debug=False):
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
    dst_cols = np.linspace(0, w * 2, cols_point)
    dst_rows = np.linspace(0, h // 2, rows_point)
    dst_rows, dst_cols = np.meshgrid(dst_rows, dst_cols)
    dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]
    dst_rows = dst[:, 1]
    dst_cols = dst[:, 0]
    dst_rows *= 1.5
    dst_rows += factor * 1.5
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
