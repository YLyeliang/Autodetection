import cv2
from PIL import Image
import numpy as np
import os
from blend import piecewiseAffineTransv2
from mtcv import imrotate
from blend.transform import logoEffect

test_img = "/Users/fiberhome/Downloads/source_data/png/flag2.jpeg"

img_pil = Image.open(test_img)
img_pil = img_pil.convert('RGBA')
w, h = img_pil.size
point_list = [[0, 0], [w, 0], [0, h], [w, h]]
# img_rotate, dst_list = logoEffect.Randomrotate(img_pil, point_list)
img_rotate, dst_list = logoEffect.RandomFlip(img_pil, point_list)

cv2.namedWindow("source")
cv2.namedWindow("transform")
img_src = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGBA2BGRA)
img_rotate = cv2.cvtColor(np.asarray(img_rotate), cv2.COLOR_BGRA2RGBA)
cv2.imshow("source", img_src)
cv2.imshow("transform", img_rotate)
cv2.waitKey()
while True:
    random_angle = np.random.randint(-30, 30)
    # mode = Image.BICUBIC
    # img_rotate = img_pil.rotate(random_angle, mode)
    # img_show = cv2.cvtColor(np.asarray(img_src), cv2.COLOR_RGBA2BGRA)
    rotated = imrotate(img_src, random_angle, auto_bound=True, border_value=(0, 0, 0, 0))

    cv2.imshow("transform", rotated)
    cv2.waitKey()
