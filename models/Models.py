import torch.nn as nn
import torch.nn.functional as func

import models.resnet


class ELuBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ELuBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if (stride != 1 or in_channels != out_channels) else nn.Sequential()

    def forward(self, x):
        y = func.elu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = func.elu(y)
        return y

class GeLuBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(GeLuBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if (stride != 1 or in_channels != out_channels) else nn.Sequential()

    def forward(self, x):
        y = func.gelu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = func.gelu(y)
        return y


class AdaptiveBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(AdaptiveBasicBlock, self).__init__()
        self.same_shape = stride == 1 and in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y1 = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y1))
        y += (x if self.same_shape else y1) # Only short-cut conv layer 2 if the shape is not the same between in and out
        y = func.relu(y)
        return y


class RationalBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(RationalBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if (stride != 1 or in_channels != out_channels) else nn.Sequential()

    def forward(self, x):
        y = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        short_cut_x = self.shortcut(x)
        y = y * short_cut_x + short_cut_x
        y = func.relu(y)
        return y


class SmallerResNet(nn.Module):
    def __init__(self, channels, labels):
        super(SmallerResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = models.resnet.make_layer(models.resnet.BasicBlock, 64, 64, 1)
        self.layer2, in_channels = models.resnet.make_layer(models.resnet.BasicBlock, in_channels, 128, 1, 2)
        self.layer3, in_channels = models.resnet.make_layer(models.resnet.BasicBlock, in_channels, 256, 1, 2)
        self.layer4, in_channels = models.resnet.make_layer(models.resnet.BasicBlock, in_channels, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.elu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class EluResNet(nn.Module):
    def __init__(self, channels, labels):
        super(EluResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = models.resnet.make_layer(ELuBasicBlock, 64, 64, 2)
        self.layer2, in_channels = models.resnet.make_layer(ELuBasicBlock, in_channels, 128, 2, 2)
        self.layer3, in_channels = models.resnet.make_layer(ELuBasicBlock, in_channels, 256, 2, 2)
        self.layer4, in_channels = models.resnet.make_layer(ELuBasicBlock, in_channels, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.elu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

class GeluResNet(nn.Module):
    def __init__(self, channels, labels):
        super(GeluResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = models.resnet.make_layer(GeLuBasicBlock, 64, 64, 2)
        self.layer2, in_channels = models.resnet.make_layer(GeLuBasicBlock, in_channels, 128, 2, 2)
        self.layer3, in_channels = models.resnet.make_layer(GeLuBasicBlock, in_channels, 256, 2, 2)
        self.layer4, in_channels = models.resnet.make_layer(GeLuBasicBlock, in_channels, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.gelu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class AdaptiveResNet(nn.Module):
    def __init__(self, channels, labels):
        super(AdaptiveResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = models.resnet.make_layer(AdaptiveBasicBlock, 64, 64, 2)
        self.layer2, in_channels = models.resnet.make_layer(AdaptiveBasicBlock, in_channels, 128, 2, 2)
        self.layer3, in_channels = models.resnet.make_layer(AdaptiveBasicBlock, in_channels, 256, 2, 2)
        self.layer4, in_channels = models.resnet.make_layer(AdaptiveBasicBlock, in_channels, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.elu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class RationalResNet(nn.Module):
    def __init__(self, channels, labels):
        super(RationalResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = models.resnet.make_layer(RationalBasicBlock, 64, 64, 2)
        self.layer2, in_channels = models.resnet.make_layer(RationalBasicBlock, in_channels, 128, 2, 2)
        self.layer3, in_channels = models.resnet.make_layer(RationalBasicBlock, in_channels, 256, 2, 2)
        self.layer4, in_channels = models.resnet.make_layer(RationalBasicBlock, in_channels, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.elu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


class FinalModel(nn.Module):
    def __init__(self, channels, labels):
        super(FinalModel, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, in_channels = models.resnet.make_layer(GeLuBasicBlock, 64, 64, 2)
        self.layer2, in_channels = models.resnet.make_layer(GeLuBasicBlock, in_channels, 128, 2, 2)
        self.layer3, in_channels = models.resnet.make_layer(GeLuBasicBlock, in_channels, 256, 2, 2)
        self.layer4, in_channels = models.resnet.make_layer(GeLuBasicBlock, in_channels, 256, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, labels)

    def forward(self, x):
        y = nn.functional.elu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y
