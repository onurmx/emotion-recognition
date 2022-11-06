# Resnet general implementation in PyTorch

import torch

class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, expansion = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv0 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(out_channels)

        self.conv1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn0(self.conv0(x)))
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return  out

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, img_channels, num_classes = 1000):
        super(ResNet, self).__init__()
        self.expansion = 4

        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet50(img_channels=3, num_classes=7):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_channels, num_classes)

def resnet101(img_channels=3, num_classes=7):
    return ResNet(Bottleneck, [3, 4, 23, 3], img_channels, num_classes)