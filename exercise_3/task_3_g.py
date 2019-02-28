from torchvision import utils, models


tensor = models.resnet18(pretrained=True).conv1.weight.data
utils.save_image(tensor, 'data/weights.png', nrow=8)
