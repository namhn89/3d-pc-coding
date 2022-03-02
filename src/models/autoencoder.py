from typing import Dict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


def convbn3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> List[nn.Module]:
    conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        stride=stride,
        bias=False,
    )
    bn = nn.BatchNorm3d(out_channels)
    relu = nn.ReLU()
    return [conv, bn, relu]


class VoxceptionResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0_0 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv0_1 = nn.Conv3d(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.conv1_0 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.conv1_1 = nn.Conv3d(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        self.conv1_2 = nn.Conv3d(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        out0 = F.relu(self.conv0_1(F.relu(self.conv0_0(x))))
        out1 = F.relu(self.conv1_2(F.relu(self.conv1_1(F.relu(self.conv1_0(x))))))
        residual = torch.cat((out0, out1), dim=1)
        out = F.relu(residual + x)

        return out


def multiple_layer(block, block_layers, channels):
    """
    Create multiple layers form base layer

    Args:
        block (_type_): _description_
        block_layers (_type_): _description_
        channels (_type_): _description_

    Returns:
        _type_: _description_
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)


class AnalysisTransform(nn.Module):
    def __init__(self, channels=[1, 16, 32, 64, 16]):
        super().__init__()
        self.conv_in = nn.Conv3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.vrn_1 = multiple_layer(
            block=VoxceptionResidualBlock,
            block_layers=3,
            channels=channels[1]
        )
        self.down_1 = nn.Conv3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )
        self.vrn_2 = multiple_layer(
            block=VoxceptionResidualBlock,
            block_layers=3,
            channels=channels[2]
        )
        self.down_2 = nn.Conv3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=2,
            bias=False,
        )
        self.vrn_3 = multiple_layer(
            block=VoxceptionResidualBlock,
            block_layers=3,
            channels=channels[3]
        )
        self.conv_out = nn.Conv3d(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.vrn_1(x)
        x = torch.nn.functional.pad(x, pad=(0, 1, 0, 1, 0, 1), mode='constant', value=0)
        x = F.relu(self.down_1(x))
        x = self.vrn_2(x)
        x = torch.nn.functional.pad(x, pad=(0, 1, 0, 1, 0, 1), mode='constant', value=0)
        x = F.relu(self.down_2(x))

        out = self.conv_out(self.vrn_3(x))

        return out


class SynthesisTransform(nn.Module):
    def __init__(self, channels=[16, 64, 32, 16, 1]):
        super().__init__()
        self.deconv_in = nn.Conv3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.vrn_1 = multiple_layer(
            block=VoxceptionResidualBlock,
            block_layers=3,
            channels=channels[1]
        )
        self.up_1 = nn.ConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
            bias=True,
        )
        self.vrn_2 = multiple_layer(
            block=VoxceptionResidualBlock,
            block_layers=3,
            channels=channels[2]
        )
        self.up_2 = nn.ConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
            bias=True,
        )
        self.vrn_3 = multiple_layer(
            block=VoxceptionResidualBlock,
            block_layers=3,
            channels=channels[3]
        )
        self.deconv_out = nn.Conv3d(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

    def forward(self, x):
        x = F.relu(self.deconv_in(x))
        x = F.relu(self.up_1(self.vrn_1(x)))
        x = F.relu(self.up_2(self.vrn_2(x)))
        out = self.deconv_out(self.vrn_3(x))

        return out


class HyperAnalysisTransform(nn.Module):
    def __init__(self, channels=[16, 16, 16, 8]):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )
        self.conv3 = nn.Conv3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.nn.functional.pad(x, pad=(0, 1, 0, 1, 0, 1), mode='constant', value=0)
        x = F.relu(self.conv2(x))
        out = self.conv3(x)
        return out


class HyperSynthesisTransform(nn.Module):
    def __init__(self, channels=[8, 16, 16, 32, 16]):
        super().__init__()
        self.deconv1 = nn.Conv3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
            bias=True,
        )

        self.deconv3 = nn.Conv3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.deconv4 = nn.Conv3d(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.deconv4_scale = nn.Conv3d(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        loc = self.deconv4(x)
        scale = self.deconv4_scale(x)

        return loc, torch.abs(scale)
