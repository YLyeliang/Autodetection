import cv2
import numpy as np
from PIL import Image
import random

def Affine(src,point_list=None):
    """
    Affine transformation.
    Args:
        src: source image, in BGR or RGBA
        point_list: list used to preserve actual corner coord of logo

    Returns:

    """
    src = cv2.cvtColor(np.asarray(src),cv2.COLOR_RGBA2BGRA)
    # decide the scale factor.
    w_scale = (random.uniform(0,0.05),random.uniform(0,0.05))
    h_scale = (random.uniform(0,0.05),random.uniform(0,0.05))
    height,width =src.shape[:2]
    shift_range_height=int(height*h_scale[0]+width*h_scale[1])
    shift_range_width=int(height*w_scale[0]+width*w_scale[1])

    src_points = np.float32([[0,0],[width,0],[0,height]])
    dst_points = src_points.copy()

    if shift_range_width>0 or shift_range_height >0:
        if shift_range_width>0:
            for i in range(3):
                dst_points[i][0]+=np.random.randint(0,shift_range_width,1)
        if shift_range_height>0:
            for i in range(3):
                dst_points[i][1]+=np.random.randint(0,shift_range_height,1)
        #  will create 2x3 matrix。 [x2,y2,b]=[A|b]*[x1,y1,1]
        perspective_mat = cv2.getAffineTransform(src_points,dst_points)
        i_min=0
        i_max=width-1
        j_min=0
        j_max=height-1
        bottom_right_x=int(perspective_mat[0][0]*(width-1) + perspective_mat[0][1]*(height-1) + perspective_mat[0][2])
        bottom_right_y = int(perspective_mat[1][0]*(width-1)+perspective_mat[1][1]*(height-1)+perspective_mat[1][2])
        bottom_right = np.array([[bottom_right_x,bottom_right_y]])
        dst_points = np.r_[dst_points,bottom_right]
        for point in dst_points:
            i_min=min(i_min,point[0])
            i_max=max(i_max,point[0])
            j_min=min(j_min,point[1])
            j_max=max(j_max,point[1])
        perspective_mat[0][2] -=i_min
        perspective_mat[1][2] -=j_min
        width_new = int(i_max-i_min)
        height_new=int(j_max-j_min)
        dst = cv2.warpAffine(src,perspective_mat,(width_new,height_new),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255,0))
        img = Image.fromarray(cv2.cvtColor(dst,cv2.COLOR_BGRA2RGBA))
        point_list = src_points.copy()
        if point_list is not None:
            point_list_trans=[]
            for point in point_list:
                x= point[0]
                y=point[1]
                x_new = int(perspective_mat[0][0]*x + perspective_mat[0][1]*y + perspective_mat[0][2])
                y_new = int(perspective_mat[1][0]*x + perspective_mat[1][1]*y + perspective_mat[1][2])
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


