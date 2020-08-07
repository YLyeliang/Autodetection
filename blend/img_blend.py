import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from mtcv.misc import histEqualize
from skimage.transform import PiecewiseAffineTransform,warp

# test_path="/users/fiberhome/Downloads/source_data/png"

def get_file_list(path):
    file_list = []
    for (root, dirs, files) in os.walk(path):
        if files:
            for i in files:
                file_path = os.path.join(root, i)
                file_list.append(file_path)
    return file_list

# png_list=get_file_list(test_path)

blend_mode={0:"naive",1:"edge_blur",2:"poisson",3:"multiply"}



def edge_blur(src,point_list=None):

    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
    src_blur = cv2.GaussianBlur(src,(13,13),11)

    mask=np.zeros(src.shape[:2],dtype=np.float32)

    idx=np.where(src[:,:,3]>15)
    mask[idx]=255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dil=cv2.erode(mask,kernel)
    mask_id=np.where(dil>0)
    src_blur[mask_id]=src[mask_id]

    # debug: show image
    # cv2.imshow("test_mask",mask)
    # cv2.imshow("test_dil", dil)
    # cv2.imshow("test_merge",src_blur)
    # cv2.waitKey()

    # mask=cv2.GaussianBlur(mask,(13,13),11)

    # mask=mask/255.
    # mask=mask[:,:,np.newaxis]
    #
    # src[:,:,:3]=src[:,:,:3]*mask
    img = Image.fromarray(cv2.cvtColor(src_blur, cv2.COLOR_BGRA2RGBA))
    return img,point_list



def edge_blur_backup(src,point_list=None):

    src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGBA2BGRA)
    mask=np.zeros(src.shape[:2],dtype=np.float32)

    idx=np.where(src[:,:,3]>15)
    mask[idx]=255

    mask=cv2.GaussianBlur(mask,(21,21),11)

    mask=mask/255.
    mask=mask[:,:,np.newaxis]

    src[:,:,:3]=src[:,:,:3]*mask
    img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA))
    plt.figure()
    plt.imshow(img)
    plt.show()

    # contours,hierarchy  =cv2.findContours(img_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(src,contours,-1,(255,0,0),2)
    return img,point_list

def paste(pixSrc,pixPng,srcH,srcW,x,y):
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
    # 1.find all indices satisfying conditions, and replace the value of indices in source image with logo image.
    # note: from pillow to numpy, (w,h) has converted to (h,w).
    # pixSrc= cv2.cvtColor(pixSrc, cv2.COLOR_RGBA2RGB)
    index = np.where(pixPng[:, :, 3] > 15)
    y_id = index[0] + y - 1
    x_id = index[1] + x - 1
    # ensure the exceeding part remained in boundary.
    y_id = np.where(y_id >= srcH, srcH - 1, y_id)
    x_id = np.where(x_id >= srcW, srcW - 1, x_id)
    id = (y_id, x_id)

    weight = pixPng[:, :, 3] / 255
    weight=weight[:,:,np.newaxis]
    alpha=weight[index]
    beta=1-alpha
    pixSrc[id]=pixSrc[id]*beta +pixPng[index]*alpha
    return cv2.cvtColor(pixSrc,cv2.COLOR_RGBA2RGB)

def naive_paste(pixSrc,pixPng,srcH,srcW,x,y):
    index = np.where(pixPng[:, :, 3] > 15)
    y_id = index[0] + y - 1
    x_id = index[1] + x - 1

    # ensure the exceeding part remained in boundary.
    y_id = np.where(y_id >= srcH, srcH - 1, y_id)
    x_id = np.where(x_id >= srcW, srcW - 1, x_id)
    id = (y_id, x_id)
    pixSrc[id] = pixPng[index]


def poisson_blend(pixSrc,pixPng,srcH,srcW,x,y):

    index = np.where(pixPng[:, :, 3] > 15)

    height,width=pixPng.shape[:2]
    mask=np.zeros(pixPng.shape[:2],dtype=np.uint8)

    p_x=x+ width//2-1
    p_y=y+height//2-1

    mask[index]=255
    pixSrc=cv2.cvtColor(pixSrc,cv2.COLOR_RGBA2BGR)
    pixPng=cv2.cvtColor(pixPng,cv2.COLOR_RGBA2BGR)
    pixPng=histEqualize(pixPng,space='rgb',clipLimit=40)
    mixed=cv2.seamlessClone(pixPng,pixSrc,mask,(p_x,p_y),cv2.NORMAL_CLONE)

    # cv2.imshow("mixed",mixed)
    # cv2.waitKey()
    return cv2.cvtColor(mixed,cv2.COLOR_BGR2RGB)

def multiply(src,logo,src_id,logo_id):
    # src_map=np.array(src,dtype=np.float32)/255
    # logo_map=np.array(logo,dtype=np.float32)/255
    # for i in range(3):
    #     src_map[src_id,i]=src_map[src_id,i]*logo_map[logo_id,i]
    # src_map[src_id]=src_map[src_id]*logo_map[logo_id]

    b,g,r = src[:,:,0],src[:,:,1],src[:,:,2]
    l_b,l_g,l_r = logo[:,:,0],logo[:,:,1],logo[:,:,2]
    bgr=[b,g,r]
    l_bgr=[l_b,l_g,l_r]
    for c,l_c in zip(bgr,l_bgr):
        c[src_id]=c[src_id]* l_c[logo_id]/255

    # s= np.hstack(bgr)
    s= np.stack(bgr,axis=2)
    # l=np.hstack(l_bgr)
    # src[src_id] = src[src_id] * logo[logo_id] / 255
    # return np.uint8(src_map)
    return s

def channel_blend(pixSrc,pixPng,srcH,srcW,x,y):

    index = np.where(pixPng[:, :, 3] > 15)
    y_id = index[0] + y - 1
    x_id = index[1] + x - 1
    # ensure the exceeding part remained in boundary.
    y_id = np.where(y_id >= srcH, srcH - 1, y_id)
    x_id = np.where(x_id >= srcW, srcW - 1, x_id)
    id = (y_id, x_id)
    # pixSrc=cv2.cvtColor(pixSrc,cv2.COLOR_RGBA2BGR)
    # pixPng=cv2.cvtColor(pixPng,cv2.COLOR_RGBA2BGR)
    pixSrc= multiply(pixSrc,pixPng,id,index)
    # pixSrc[id,:3]*=pixPng[index,:3]/255
    return cv2.cvtColor(pixSrc,cv2.COLOR_BGR2RGB)