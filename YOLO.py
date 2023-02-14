import torch
from  ModelFactory import get_arch
from torch import nn
from torch.nn import init

class YOLO(torch.nn.Module):

    def __init__(self, arch_model= 'resnet50', s =14, classes = 4, b = 2, training = True):
        print(arch_model)
        super().__init__()
        self.training = training
        self.classes = classes
        self.s = s
        self.b = b
        _, channels = get_arch(arch_model)
        #self.upsample = nn.ConvTranspose2d(channels,1024,4,stride=2, padding=1);init.xavier_normal_(self.upsample.weight)
        #self.bn_up = nn.BatchNorm2d(1024)

        #self.conv1 = nn.Conv2d(1024,1024,3,padding=1);init.xavier_normal_(self.conv1.weight)
        #self.bn_1 = nn.BatchNorm2d(1024)

        #self.out = nn.Conv2d(1024, self.b*3+ self.classes,3,stride=1, padding=1, bias=False)
        self.out = nn.Conv2d(channels, self.b*3+ self.classes,3,stride=1, padding=1, bias=False)

        init.xavier_normal_(self.out.weight)
        self.bn_end = nn.BatchNorm2d( self.b*3 + self.classes)



        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.arch, _ = get_arch(arch_model,pretrained=True)

    def forward(self, x ):
        feat = self.arch(x)
        #up = self.bn_up(self.upsample(feat))
        #c1 = self.bn_1(self.conv1(up))
        #out = self.out(c1)
        out = self.out(feat)
        out_bn = torch.sigmoid(self.bn_end(out))
        return out_bn.permute(0,2,3,1)


if __name__ == '__main__':

    model = YOLO()
    x = torch.rand([4,3,448,448])
    out = model(x)
    print(out.shape)
