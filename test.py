import torch
# from det.models.backbones.resnet import ResNet
from det.models.backbones.resnext import ResNeXt
from det.models.necks.pafpn import PAFPN
from det.models.builder import build_backbone
# import numpy as np
# lr=0.001
# a=np.linspace(lr/20,lr,20).tolist()
# for _ in range(20):
#     print(a.pop(0))

# from torchvision import models
# models.resnext101_32x8d()

net =build_backbone(dict(type='ResNet',depth=50))

debug=1
# net=ResNet(50)
# net=ResNeXt(depth=50,groups=32,base_width=4)
# neck=PAFPN([256,512,1024,2048],256,4,add_extra_convs=True,start_level=1)
# x=torch.rand([1,3,320,320])
# net.init_weights()
# out=net(x)
# neck_out=neck(out)
# print(out)
# print(neck_out)