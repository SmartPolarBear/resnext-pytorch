import torch.nn as nn
import torch.nn.functional as functional

from model.conv2d import conv2d_with_bn
from model.block import resnext_block


class resnext(nn.Module):
    def __init__(self, layers, cardinality, group_depth, num_classes):
        super(resnext, self).__init__()
        self.cardinality = cardinality
        self.channels = 64

        self.conv1 = conv2d_with_bn(3, self.channels, 7, stride=2, padding=3)

        d1 = group_depth
        self.conv2 = self.__make_layers(d1, layers[0], stride=1)

        d2 = 2 * d1
        self.conv3 = self.__make_layers(d2, layers[1], stride=2)

        d3 = d2 * 2
        self.conv4 = self.__make_layers(d3, layers[2], stride=2)

        d4 = d3 * 2
        self.conv5 = self.__make_layers(d4, layers[3], stride=2)

        self.fc = nn.Linear(self.channels, num_classes)

    def __make_layers(self, depth, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(resnext_block(self.channels, self.cardinality, depth, stride))
            self.channels = self.cardinality * depth * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = functional.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = functional.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = functional.softmax(self.fc(out))
        return out


def resnext50_32x4d(num_classes=1000):
    return resnext([3, 4, 6, 3], 32, 4, num_classes)


def resnext101_32x4d(num_classes=1000):
    return resnext([3, 4, 23, 3], 32, 4, num_classes)


def resnext101_64x4d(num_classes=1000):
    return resnext([3, 4, 23, 3], 64, 4, num_classes)
