import torch
from det.models.backbones.resnet import ResNet

net=ResNet(50)
x=torch.rand([1,3,320,320])
net.init_weights()
out=net(x)
print(out)