import cv2
from PIL import Image
import os
from blend import piecewiseAffineTransv2

test_img = "/Users/fiberhome/Downloads/source_data/png/flag1.jpg"

img_pil = Image.open(test_img)
img_pil = img_pil.convert('RGBA')
piecewiseAffineTransv2(img_pil, [0, 0, 0, 0], True)
img = cv2.imread(test_img)

cv2.namedWindow("source")
cv2.namedWindow("transform")

cv2.imshow("source", img)

h, w, _ = img.shape
img_resize = cv2.resize(img, (w // 3, h))
cv2.imshow("transform", img_resize)
cv2.waitKey()
