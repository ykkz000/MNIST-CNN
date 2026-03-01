import torch.nn as nn
from models.resnet.ResNetBlock import BasicBlock, BottleneckBlock


def make_layer(block, in_channels, out_channels, blocks, stride=1):
    layers = [block(in_channels, out_channels, stride)]
    in_channels = out_channels * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channels, out_channels))
    return nn.Sequential(*layers), in_channels


class ResNet18(nn.Module):
    def __init__(self, channels, labels):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = make_layer(BasicBlock, 64, 64, 2)
        self.layer2, in_channels = make_layer(BasicBlock, in_channels, 128, 2, 2)
        self.layer3, in_channels = make_layer(BasicBlock, in_channels, 256, 2, 2)
        self.layer4, in_channels = make_layer(BasicBlock, in_channels, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class ResNet34(nn.Module):
    def __init__(self, channels, labels):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = make_layer(BasicBlock, 64, 64, 3)
        self.layer2, in_channels = make_layer(BasicBlock, in_channels, 128, 4, 2)
        self.layer3, in_channels = make_layer(BasicBlock, in_channels, 256, 6, 2)
        self.layer4, in_channels = make_layer(BasicBlock, in_channels, 512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class ResNet50(nn.Module):
    def __init__(self, channels, labels):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = make_layer(BottleneckBlock, 64, 64, 3)
        self.layer2, in_channels = make_layer(BottleneckBlock, in_channels, 128, 4, 2)
        self.layer3, in_channels = make_layer(BottleneckBlock, in_channels, 256, 6, 2)
        self.layer4, in_channels = make_layer(BottleneckBlock, in_channels, 512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class ResNet101(nn.Module):
    def __init__(self, channels, labels):
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = make_layer(BottleneckBlock, 64, 64, 3)
        self.layer2, in_channels = make_layer(BottleneckBlock, in_channels, 128, 4, 2)
        self.layer3, in_channels = make_layer(BottleneckBlock, in_channels, 256, 23, 2)
        self.layer4, in_channels = make_layer(BottleneckBlock, in_channels, 512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class ResNet152(nn.Module):
    def __init__(self, channels, labels):
        super(ResNet152, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = make_layer(BottleneckBlock, 64, 64, 3)
        self.layer2, in_channels = make_layer(BottleneckBlock, in_channels, 128, 8, 2)
        self.layer3, in_channels = make_layer(BottleneckBlock, in_channels, 256, 36, 2)
        self.layer4, in_channels = make_layer(BottleneckBlock, in_channels, 512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y
