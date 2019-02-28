from torchvision import utils, models
from dataloaders import load_cifar10
from torch import nn
from utils import to_cuda

data_train, data_val, data_test = load_cifar10(1, 0)

tensor = models.resnet18(pretrained=True)
tensor.fc = nn.Linear(512*4, 10)
for param in tensor.parameters ():
    param.requires_grad = False
for param in tensor.fc.parameters ():
    param.requires_grad = True
for param in tensor.layer4.parameters ():
    param.requires_grad = True

henk = nn.Sequential(tensor.conv1, tensor.layer1, tensor.layer2, tensor.layer3, tensor.layer4)
print(henk)

for batch_it, (x,y) in enumerate(data_train):
    x = nn.functional.interpolate(x, scale_factor=8)
    print(x.size())
    derp = henk.forward(x)
    print(derp.size())
    utils.save_image(x, 'task_3_f/image.png', nrow=256)
    for i in range(0,10):
        utils.save_image(derp[0][i], 'task_3_f/filter'+str(i)+'.png', nrow=164)
    print(derp)
    break
