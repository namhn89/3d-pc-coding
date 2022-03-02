from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _VoxceptionResNet(nn.Module):
    """Voxception Residual Network Block.
    Arguments:
      num_filters: number of filters passed to a convolutional layer.
      in_chs：input channels
    """

    def __init__(self, num_filters, in_chs, activation=F.relu):
        super().__init__()
        self.activation = activation
        # path_1
        self.conv1_1 = nn.Conv3d(
            in_channels=in_chs,
            out_channels=int(num_filters / 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv1_2 = nn.Conv3d(
            in_channels=int(num_filters / 4),
            out_channels=int(num_filters / 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        # path_2
        self.conv2_1 = nn.Conv3d(
            in_channels=in_chs,
            out_channels=int(num_filters / 4),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.conv2_2 = nn.Conv3d(
            in_channels=int(num_filters / 4),
            out_channels=int(num_filters / 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2_3 = nn.Conv3d(
            in_channels=int(num_filters / 4),
            out_channels=int(num_filters / 2),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        # path1
        tensor1_1 = self.conv1_1(x)
        tensor1_1 = self.activation(tensor1_1)
        tensor1_2 = self.conv1_2(
            tensor1_1
        )  # tensor1_2.shape: (1, 32, 32, 32, 16)
        tensor1_2 = self.activation(tensor1_2)
        # path2
        tensor2_1 = self.conv2_1(x)
        tensor2_1 = self.activation(tensor2_1)
        tensor2_2 = self.conv2_2(tensor2_1)
        tensor2_2 = self.activation(tensor2_2)
        tensor2_3 = self.conv2_3(
            tensor2_2
        )  # tensor2_3.shape: (1, 32, 32, 32, 16)
        tensor2_3 = self.activation(tensor2_3)
        # concat paths
        residual = torch.cat(
            (tensor1_2, tensor2_3), dim=1
        )  # residual.shape: (1, 32, 32, 32, 32)
        # add & relu
        output = F.relu(x + residual)  # output.shape: (1, 32, 32, 32, 32)

        return output


class AnalysisTransform(nn.Module):
    """Analysis transformation.
    Arguments:
      None.
    """

    def __init__(self):
        super().__init__()

        def vrn_block(num_filters, vrnin_chs):
            return _VoxceptionResNet(num_filters, vrnin_chs)

        self.in_chs = 1
        self.conv_in = nn.Conv3d(
            in_channels=self.in_chs,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.vrn1_1 = vrn_block(num_filters=16, vrnin_chs=16)
        self.vrn1_2 = vrn_block(num_filters=16, vrnin_chs=16)
        self.vrn1_3 = vrn_block(num_filters=16, vrnin_chs=16)
        self.down_1 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )
        self.vrn2_1 = vrn_block(num_filters=32, vrnin_chs=32)
        self.vrn2_2 = vrn_block(num_filters=32, vrnin_chs=32)
        self.vrn2_3 = vrn_block(num_filters=32, vrnin_chs=32)

        self.down_2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )
        self.vrn3_1 = vrn_block(num_filters=64, vrnin_chs=64)
        self.vrn3_2 = vrn_block(num_filters=64, vrnin_chs=64)
        self.vrn3_3 = vrn_block(num_filters=64, vrnin_chs=64)

        self.conv_out = nn.Conv3d(
            in_channels=64,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        """针对tf的测试结果(batch, depth, height, width, channels)
    x.shape: (1, 64, 64, 64, 1)
    feature1.shape: (1, 64, 64, 64, 16)
    feature1_1.shape: (1, 64, 64, 64, 16)
    feature1_2.shape: (1, 64, 64, 64, 16)
    feature1_3.shape: (1, 64, 64, 64, 16)
    feature2.shape: (1, 32, 32, 32, 32)
    feature2_1.shape: (1, 32, 32, 32, 32)
    feature2_2.shape: (1, 32, 32, 32, 32)
    feature2_3.shape: (1, 32, 32, 32, 32)
    feature3.shape: (1, 16, 16, 16, 64)
    feature3_1.shape: (1, 16, 16, 16, 64)
    feature3_2.shape: (1, 16, 16, 16, 64)
    feature3_3.shape: (1, 16, 16, 16, 64)
    feature4.shape: (1, 16, 16, 16, 16)
    """

    def forward(self, x):
        feature1 = self.conv_in(x)  # [N,N,N,16]
        feature1 = F.relu(feature1)
        feature1_1 = self.vrn1_1(feature1)
        feature1_2 = self.vrn1_2(feature1_1)
        feature1_3 = self.vrn1_3(feature1_2)  # [N,N,N,16]

        feature1_3 = torch.nn.functional.pad(
            feature1_3, pad=(0, 1, 0, 1, 0, 1), mode="constant", value=0
        )  # padding:top 0 bottom 1 left 0 right 1
        feature2 = self.down_1(feature1_3)  # [N/2,N/2,N/2,32]
        feature2 = F.relu(feature2)
        feature2_1 = self.vrn2_1(feature2)
        feature2_2 = self.vrn2_2(feature2_1)
        feature2_3 = self.vrn2_3(feature2_2)  # [N/2,N/2,N/2,32]

        feature2_3 = torch.nn.functional.pad(
            feature2_3, pad=(0, 1, 0, 1, 0, 1), mode="constant", value=0
        )  # padding:top 0 bottom 1 left 0 right 1
        feature3 = self.down_2(feature2_3)  # [N/4,N/4,N/4,64]
        feature3 = F.relu(feature3)
        feature3_1 = self.vrn3_1(feature3)
        feature3_2 = self.vrn3_2(feature3_1)
        feature3_3 = self.vrn3_3(feature3_2)  # [N/4,N/4,N/4,64]

        feature4 = self.conv_out(feature3_3)  # [N/4,N/4,N/4,16]

        return feature4


class SynthesisTransform(nn.Module):
    def __init__(self):
        super().__init__()

        def vrn_block(num_filters, vrnin_chs):
            return _VoxceptionResNet(num_filters, vrnin_chs)

        self.in_chs = 16
        self.deconv_in = nn.Conv3d(
            in_channels=self.in_chs,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.vrn1_1 = vrn_block(num_filters=64, vrnin_chs=64)
        self.vrn1_2 = vrn_block(num_filters=64, vrnin_chs=64)
        self.vrn1_3 = vrn_block(num_filters=64, vrnin_chs=64)

        self.up_1 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.vrn2_1 = vrn_block(num_filters=32, vrnin_chs=32)
        self.vrn2_2 = vrn_block(num_filters=32, vrnin_chs=32)
        self.vrn2_3 = vrn_block(num_filters=32, vrnin_chs=32)

        self.up_2 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.vrn3_1 = vrn_block(num_filters=16, vrnin_chs=16)
        self.vrn3_2 = vrn_block(num_filters=16, vrnin_chs=16)
        self.vrn3_3 = vrn_block(num_filters=16, vrnin_chs=16)

        self.deconv_out = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
    """
    (batch, depth, height, width, channels)
    x.shape: (1, 16, 16, 16, 16)
    feature1.shape: (1, 16, 16, 16, 64)
    feature1_1.shape: (1, 16, 16, 16, 64)
    feature1_2.shape: (1, 16, 16, 16, 64)
    feature1_3.shape: (1, 16, 16, 16, 64)
    feature2.shape: (1, 32, 32, 32, 32)
    feature2_1.shape: (1, 32, 32, 32, 32)
    feature2_2.shape: (1, 32, 32, 32, 32)
    feature2_3.shape: (1, 32, 32, 32, 32)
    feature3.shape: (1, 64, 64, 64, 16)
    feature3_1.shape: (1, 64, 64, 64, 16)
    feature3_2.shape: (1, 64, 64, 64, 16)
    feature3_3.shape: (1, 64, 64, 64, 16)
    feature4.shape: (1, 64, 64, 64, 1)
    """

    def forward(self, x):
        feature1 = self.deconv_in(x)  # [N/4,N/4,N/4,64]
        feature1 = F.relu(feature1)
        feature1_1 = self.vrn1_1(feature1)
        feature1_2 = self.vrn1_2(feature1_1)
        feature1_3 = self.vrn1_3(feature1_2)  # [N/4,N/4,N/4,64]

        feature2 = self.up_1(feature1_3)  # [N/2,N/2,N/2,32]
        feature2 = F.relu(feature2)
        feature2_1 = self.vrn2_1(feature2)
        feature2_2 = self.vrn2_2(feature2_1)
        feature2_3 = self.vrn2_3(feature2_2)  # [N/2,N/2,N/2,32]

        feature3 = self.up_2(feature2_3)  # [N,N,N,16]
        feature3 = F.relu(feature3)
        feature3_1 = self.vrn3_1(feature3)
        feature3_2 = self.vrn3_2(feature3_1)
        feature3_3 = self.vrn3_3(feature3_2)  # [N,N,N,16]

        feature4 = self.deconv_out(feature3_3)  # [N,N,N,1]
        return feature4


class HyperEncoder(nn.Module):
    """Hyper encoder."""

    def __init__(self, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.in_chs = 16
        self.conv1 = nn.Conv3d(
            in_channels=self.in_chs,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )
        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
    """(batch, depth, height, width, channels)
    x.shape: (1, 16, 16, 16, 16)
    f1.shape: (1, 16, 16, 16, 16)
    f2.shape: (1, 8, 8, 8, 16)
    f3.shape: (1, 8, 8, 8, 8)
    """

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = self.activation(f1)
        f1 = torch.nn.functional.pad(
            f1, pad=(0, 1, 0, 1, 0, 1), mode="constant", value=0
        )
        f2 = self.conv2(f1)
        f2 = self.activation(f2)
        f3 = self.conv3(f2)
        return f3


class HyperDecoder(nn.Module):
    """Hyper decoder.
    Return: location, scale.
    """

    def __init__(self, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.in_chs = 8
        self.conv1 = nn.Conv3d(
            in_channels=self.in_chs,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv4_1 = nn.Conv3d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv4_2 = nn.Conv3d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        """(batch, depth, height, width, channels)
    x.shape: (1, 8, 8, 8, 8)
    f1.shape: (1, 8, 8, 8, 16)
    f2.shape: (1, 16, 16, 16, 16)
    f3.shape: (1, 16, 16, 16, 32)
    loc.shape: (1, 16, 16, 16, 16)
    scale.shape: (1, 16, 16, 16, 16)
    """

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = self.activation(f1)
        f2 = self.conv2(f1)
        f2 = self.activation(f2)
        f3 = self.conv3(f2)
        f3 = self.activation(f3)

        loc = self.conv4_1(f3)
        scale = self.conv4_2(f3)
        return loc, torch.abs(scale)


if __name__ == "__main__":
    # inputs
    inputs = torch.randn(1, 1, 64, 64, 64)
    # encoder & decoder
    encoder = AnalysisTransform()
    features = encoder(inputs)
    print("features.shape:", features.shape)  # torch.Size([1, 16, 16, 16, 16])
    decoder = SynthesisTransform()
    outputs = decoder(features)
    print("outputs.shape:", outputs.shape)  # torch.Size([1, 1, 64, 64, 64])

    # hyper_encoder & hyper_decoder
    hyper_encoder = HyperEncoder()
    hyper_decoder = HyperDecoder()
    #
    hyper_prior = hyper_encoder(features)
    print(
        "hyper_prior.shape:", hyper_prior.shape
    )  # torch.Size([1, 8, 8, 8, 8])
    loc, scale = hyper_decoder(hyper_prior)
    print(
        "loc.shape:", loc.shape, ", scale.shape:", scale.shape
    )  # loc.shape: torch.Size([1, 16, 16, 16, 16]) , scale.shape: torch.Size([1, 16, 16, 16, 16])

    print(encoder)
    print(decoder)

    print(hyper_encoder)
    print(hyper_decoder)
