import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math


# Squeeze-Excitation block
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


# Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# Realization of Squeeze-Excitation Spatial Mixed module
class SqueezeExcitationSpatialMixed(nn.Module):
    """
    This function adds spatial attention mechanism to SE block
    It can be seen here:
    https://github.com/Haonanfan123/Mixed-attention-mechanism
    """
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4,
                 kernel_size: int = 7):
        super(SqueezeExcitationSpatialMixed, self).__init__()
        self.seattention = SqueezeExcitation(input_c, expand_c, squeeze_factor)
        self.spattention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.seattention(x) * self.spattention(x)
        return x


# Realization of Multilayer Mixed Attention Mechanism module
class MultilayerMixedAttentionMechanism(nn.Module):
    """
        This function refers to the mixed attention mechanism of ECAblock and PMAMblock
        It can be seen here:
        https://github.com/Haonanfan123/Mixed-attention-mechanism
    """
    def __init__(self, channel, b=1, gamma=2):
        super(MultilayerMixedAttentionMechanism, self).__init__()
        kernel_size_c = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size_c = kernel_size_c if kernel_size_c % 2 else kernel_size_c + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size_c, padding=(kernel_size_c - 1) // 2, bias=False)
        self.spatialattention = SpatialAttention(kernel_size=7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        yc1 = self.max_pool(x)
        yc1 = self.conv(yc1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        yc2 = self.avg_pool(x)
        yc2 = self.conv(yc2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        yc = yc1 + yc2
        yc = self.sigmoid(yc)
        ys = self.spatialattention(x)
        return x * yc.expand_as(x) * ys.expand_as(x)