import torchvision
from torchvision import models

print(dir(models))
densenet=models.densenet121(pretrained=True)

debug=1