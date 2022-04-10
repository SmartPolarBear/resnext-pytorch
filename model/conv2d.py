import torch.nn as nn
import torch.nn.functional as functional


class conv2d_with_bn(nn.Module):
    """
    Batch-normed convolution kernel with relu activation
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(conv2d_with_bn, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return functional.relu(self.seq(x))
