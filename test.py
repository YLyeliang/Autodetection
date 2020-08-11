import torch
from det.models.backbones.resnet import ResNet
from det.models.necks.pafpn import PAFPN
net=ResNet(50)
neck=PAFPN([256,512,1024,2048],256,5)
x=torch.rand([1,3,320,320])
net.init_weights()
out=net(x)
neck_out=neck(out)
print(out)
print(neck_out)