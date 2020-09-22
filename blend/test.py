import cv2
from PIL import Image
import numpy as np
import os
from blend import piecewiseAffineTransv2
from mtcv import imrotate

test_img = "/Users/fiberhome/Downloads/source_data/png/flag1.jpg"

img_pil = Image.open(test_img)
img_pil = img_pil.convert('RGBA')

cv2.namedWindow("source")
cv2.namedWindow("transform")
img_src = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGBA2BGRA)
cv2.imshow("source", img_src)
while True:
    random_angle = np.random.randint(-30, 30)
    # mode = Image.BICUBIC
    # img_rotate = img_pil.rotate(random_angle, mode)
    # img_show = cv2.cvtColor(np.asarray(img_src), cv2.COLOR_RGBA2BGRA)
    rotated = imrotate(img_src, random_angle, auto_bound=True, border_value=(0, 0, 0, 0))

    cv2.imshow("transform", rotated)
    cv2.waitKey()
