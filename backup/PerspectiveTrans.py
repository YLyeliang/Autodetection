from PIL import Image
import cv2
import numpy as np
import random
import math

def rad(x):
    return x*np.pi / 180

def Perspective(src,point_list=None):
    """
    Perspective transformation of image
    Args:
        src: source image, should be RGBA or BGR
        point_list:

    Returns:

    """
    src = cv2.cvtColor(np.asarray(src),cv2.COLOR_RGBA2BGRA)
    w_scale=(random.uniform(0,0.08),random.uniform(0,0.08))
    h_scale=(random.uniform(0,0.08),random.uniform(0,0.08))
    height,width = src.shape[:2]
    shift_range_h=int(height*h_scale[0]+width*h_scale[1])
    shift_range_w=int(height*w_scale[0]+width*w_scale[1])
    src_points = np.float32([[0,0],[width,0],[0,height],[width,height]])
    dst_points = src_points.copy()

    if shift_range_w>0 or shift_range_h>0:
        if shift_range_w>0:
            for i in range(4):
                dst_points[i][0]+=np.random.randint(0,shift_range_w,1)

        if shift_range_h>0:
            for i in range(4):
                dst_points[i][1]+=np.random.randint(0,shift_range_h,1)

        # calculate the perspective transformation matrix.
        # [t_i*x_i,t_i*y_i,t_i] = map_matrix*[x_i,y_i,1] where i=0,1,2,3 related to four pairs of corresponding coordinates.
        perspective_mat = cv2.getPerspectiveTransform(src=src_points,dst=dst_points)

        i_min=0
        i_max=width-1
        j_min=0
        j_max=height-1
        for point in dst_points:
            i_min=min(i_min,point[0])
            i_max=max(i_max,point[0])
            j_min=min(j_min,point[1])
            j_max=max(j_max,point[1])
        for ii in range(3):
            perspective_mat[0][ii] -= perspective_mat[2][ii]*i_min
            perspective_mat[1][ii] -= perspective_mat[2][ii]*j_min

        width_new = int(i_max-i_min)
        height_new = int(j_max-j_min)

        point_list_trans=[]
        dst=cv2.warpPerspective(src,perspective_mat,(width_new,height_new),borderValue=(0,0,0,0))

        img = Image.fromarray(cv2.cvtColor(dst,cv2.COLOR_BGRA2RGBA))
        point_list=src_points.copy()

        if point_list is not None:
            for point in point_list:
                x = point[0]
                y = point[1]
                x_new = int((perspective_mat[0][0]*x + perspective_mat[0][1]*y+
                             perspective_mat[0][2])/(perspective_mat[2][0]*x+perspective_mat[2][1]*y+perspective_mat[2][2]))
                y_new = int((perspective_mat[1][0]*x + perspective_mat[1][1]*y +
                             perspective_mat[1][2])/(perspective_mat[2][0]*x+perspective_mat[2][1]*y+perspective_mat[2][2]))
                point_list_trans.append([x_new,y_new])

            # debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(img)
            plt.show()

            return img,point_list_trans
        else:
            return img,point_list

    else:
        img=Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGRA2RGBA))
    return img,point_list