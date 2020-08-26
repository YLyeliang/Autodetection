import torchvision
from torchvision import models

print(dir(models))
resnext=models.resnext50_32x4d(pretrained=True)
# densenet=models.densenet121(pretrained=True)

# debug=1


# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.transform import PiecewiseAffineTransform, warp
# from skimage import dataset
# import os
# import cv2

# image = dataset.astronaut()
# image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
# png='/Users/fiberhome/Downloads/source_data/png'
# files=os.listdir(png)
# img_path = [os.path.join(png,file) for file in files if 'jp' in file]
# for img in img_path:
#     img = cv2.imread(img)
#     m1,std1=cv2.meanStdDev(img)
#     m2,std2=cv2.meanStdDev(image)
#     diff = m1-m2
#     diff=np.array(diff).reshape((1,1,3))
#     img1=img.copy()
#     img1=img1.astype(np.float64)
#     img1-=diff
#     img1=np.clip(img1,0,255)
#     img1=img1.astype(np.uint8)
#     cv2.imshow("img",img1)
#     cv2.waitKey()

# img1='/Users/fiberhome/Downloads/source_data/iO6S1m.jpeg'
# img2 ='/Users/fiberhome/Downloads/source_data/kfku3m.jpeg'
# img1 = cv2.imread(img1)
# img2 =cv2.imread(img2)
# m1,std1=cv2.meanStdDev(img1)
# m2,std2=cv2.meanStdDev(img2)
# diff = m1-m2
# diff=np.array(diff).reshape((1,1,3))
# img1=img1.astype(np.float64)
# img1-=diff
# img1=np.clip(img1,0,255)
# img1=img1.astype(np.uint8)
# cv2.imshow("img",img1)
# cv2.waitKey()

