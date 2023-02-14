import torchvision
import torch


def get_arch(name, pretrained=True):
    if name == 'vgg16':
        return torchvision.models.vgg16_bn(pretrained=pretrained).features, 512

    if name == 'mobilenet_v2':
        return torchvision.models.mobilenet_v2(pretrained=pretrained).features, 1280

    if name == 'densenet121':
        return torchvision.models.densenet121(pretrained=pretrained).features, 1024

    if name == 'densenet161':
        return torchvision.models.densenet121(pretrained=pretrained).features, 2208

    if name == 'densenet201':
        return torchvision.models.densenet121(pretrained=pretrained).features, 1920

    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        model = torch.nn.Sequential(model.conv1,
                                    model.bn1,
                                    model.relu,
                                    model.maxpool,
                                    model.layer1,
                                    model.layer2,
                                    model.layer3,
                                    model.layer4)

        return model, 2048

      
      


