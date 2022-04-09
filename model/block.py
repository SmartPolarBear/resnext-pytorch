import torch.nn as nn
import torch.nn.functional as functional

from model.conv2d import conv2d_with_bn


class resnext_block(nn.Module):
    """
    block of ResNeXt with group convolutions
    """

    def __init__(self, in_channels, cardinality, group_depth, stride):
        super(resnext_block, self).__init__()

        self.group_channels = cardinality * group_depth

        self.conv1 = conv2d_with_bn(in_channels, self.group_channels, 1, stride=1, padding=0)
        self.conv2 = conv2d_with_bn(self.group_channels, self.group_channels, 3, stride=stride, padding=1,
                                    groups=cardinality)

        self.conv3 = nn.Conv2d(self.group_channels, self.group_channels * 2, 1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(self.group_channels * 2)

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, self.group_channels * 2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_channels * 2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return functional.relu(out)
