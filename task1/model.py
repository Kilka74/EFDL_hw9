from torch import nn as nn
from torchvision.models import ResNet101_Weights, resnet101


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Resnet101(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        if pretrained:
            print("prerained case")
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
            print(sum(p.numel() for p in self.model.parameters()))
            
        else:
            print("untrained case")
            self.model = resnet101()
            self.model.layer3 = self.model.layer3[0]
            print(sum(p.numel() for p in self.model.parameters()))

        self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, bias=False)
        self.model.maxpool = Identity()
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes, bias=True)

    def forward(self, x, return_temporary=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        layer1_x = self.model.layer1(x)
        layer2_x = self.model.layer2(layer1_x)
        layer3_x = self.model.layer3(layer2_x)
        layer4_x = self.model.layer4(layer3_x)
        out = self.model.avgpool(layer4_x)
        res = self.model.fc(out.squeeze())
        if return_temporary:
            return res, layer1_x, layer2_x, layer4_x
        return res
