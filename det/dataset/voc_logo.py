import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0]==2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_ROOT="/mnt/Data_Disk/lx/TBData/TB_confusion"

class VOCAnnotationTransform(object):
    """
    Transforms a VOC annotation into a Tensor fo bbox coords and label index initialized with a
    dictionary lookup of classnames to indexes
    """

    def __init__(self,class_to_id=None,keep_difficult=False):
        self.class_to_id = class_to_id or dict(
            zip(VOC_CLASSES,range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self,target,width,height):
        """
        Arguments:
            target (annotation): the target annotation to be made usable will be an ET.Element
        Returns:
            a list containing lists of bounding boxes [bbox coords, cls name]
        """
        res= []
        for obj in target.iter('object'):
            name=obj.find('name').text.strip()
            bbox=obj.find('bndbox')

            pts=['xmin','ymin','xmax','ymax']
            bndbox=[]
            for i,pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)-1
                # scale height or width:  xmin/width, xmax/width ymin/height
                # ymax/height
                cur_pt = cur_pt / width if i%2--0 else cur_pt /height
                bndbox.append(cur_pt)
            label_idx = self.class_to_id[name]
            bndbox.append(label_idx)
            res+=[bndbox]

        return res


class VOCDetection(data.Dataset):

    def __init__(self,
                 root,
                 image_sets=[],
                 transform=None,target_transform=VOCAnnotationTransform(),
                 dataset_name="logo"):
        self.root=root
        self.image_sets =image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name=dataset_name
        self._annopath=osp.join('$s','Annotations','%s.xml')
        self._imgpath=osp.join('%s','JPEGImages','%s.jpg')
        self.ids=[]

        for folder in image_sets:
            rootpath=osp.join(self.root,folder)
            for line in open(osp.join(rootpath,'ImageSets','Main','trainval.txt')):
                self.ids.append((rootpath,line.split())) # tuple (path, filename) in list


    def __getitem__(self,index):
        im,gt,h,w=self.pull_item(index)

        return im,gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self,index):
        img_id=self.ids[index]
        target=ET.parse(self._annopath%img_id).getroot()
        img =cv2.imread(self._imgpath%img_id)
        height,width,channels=img.shape
        if self.target_transform is not None:
            target = self.target_transform(target,width,height)

        if self.transform is not None:
            target = np.array(target)
            img,boxes,labels = self.transform(img,target[:,:4],target[:,4])
            # convert to rgb
            img =img[:,:,(2,1,0)]
            target = np.hstack((boxes,np.expand_dims(labels,axis=1)))
        return torch.from_numpy(img).permute(2,0,1),target,height,width # convert to CHW from HWC

    def pull_image(self,index):

        img_id = self.ids[index]
        return cv2.imread(self._imgpath%img_id, cv2.IMREAD_COLOR)

    def pull_anno(self,index):
        img_id = self.ids[index]
        anno=ET.parse(self._annopath%img_id).getroot()
        gt=self.target_transform(anno,1,1)
        return img_id[1],gt

    def pull_tensor(self,index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)





