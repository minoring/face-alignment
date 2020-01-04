import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes,
                   out_planes,
                   kernel_size=3,
                   stride=stride,
                   padding=padding,
                   bias=bias)


class ConvBlock(nn.Module):

  def __init__(self, in_planes, out_planes):
    super(ConvBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = conv3x3(in_planes, int(out_planes / 2))
    self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
    self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
    self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
    self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

    if in_planes != out_planes:
      self.downsample = nn.Sequential(
          nn.BatchNorm2d(in_planes), nn.ReLU(True),
          nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
    else:
      self.downsample = None

  def forward(self, x):
    residual = x

    out1 = self.bn1(x)
    out1 = F.relu(out1, True)
    out1 = self.conv1(out1)

    out2 = self.bn2(out1)
    out2 = F.relu(out2, True)
    out2 = self.conv2(out2)

    out3 = self.bn3(out2)
    out3 = F.relu(out3, True)
    out3 = self.conv3(out3)

    out3 = torch.cat((out1, out2, out3), 1)

    if self.downsample is not None:
      residual = self.downsample(residual)

    out3 += residual  # Bottlenect 에서는 residual 더하고 relu 했는데 여기선 왜 안할까?

    return out3


class Bottleneck(nn.Module):

  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes,
                           planes,
                           kernel_size=3,
                           stride=stride,
                           padding=1,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)

    self.relu = nn.ReLU(True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class HourGlass(nn.Module):

  def __init__(self, num_modules, depth, num_features):
    super(HourGlass, self).__init__()
    self.num_modules = num_modules
    self.depth = depth
    self.features = num_features

    self._generate_network(self.depth)

  def _generate_network(self, level):
    self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))
    self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

    if level > 1:
      self._generate_network(level - 1)
    else:
      self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))
    
    self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))
  
  def _forward(self, level, inp):
    # Upper branch
    up1 = inp
    up1 = self._modules['b1_' + str(level)](up1)

    # Lower branch
    low1 = F.avg_pool2d(inp, 2, stride=2)
    low1 = self._modules['b2_' + str(level)](low1)

    if level > 1:
      low2 = self._forward(level - 1, low1)
    else:
      low2 = low1
      low2 = self._modules['b2_plus' + str(level)](low2)

    low3 = low2
    low3 = self._modules['b3_' + str(level)](low3)

    up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

    return up1 + up2
  
  def forward(self, x):
    return self._forward(self.depth, x)
