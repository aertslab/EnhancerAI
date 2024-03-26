from __future__ import annotations

import torch
import torch.nn as nn

from enhancerai.tl.zoo.utils import ConvBlock, DenseBlock


class DeepTopicCNN(nn.Module):
    """DeepTopic CNN model architecture for topic modeling of chromatin accessibility data."""

    def __init__(
        self,
        num_classes: int,
        filter_size: int = 17,
        num_filters: int = 1024,
        pool_size: int = 4,
        dense_out: int = 1024,
        activation: str = "relu",
        seq_len: int = 500,
        seq_channels: int = 4,
        conv_do: float = 0.15,
        dense_do: float = 0.5,
        pre_dense_do: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.dense_out = dense_out
        self.activation = activation
        self.seq_len = seq_len
        self.seq_channels = seq_channels
        self.conv_do = conv_do
        self.dense_do = dense_do
        self.pre_dense_do = pre_dense_do

        # Conv layers
        self.conv1 = ConvBlock(
            num_filters,
            filter_size,
            seq_channels,
            pool_size,
            "gelu",
            conv_do,
            False,
        )
        self.conv2 = ConvBlock(
            num_filters // 2,
            11,
            num_filters,
            pool_size,
            activation,
            conv_do,
            False,
        )
        self.conv3 = ConvBlock(
            num_filters // 2,
            11,
            num_filters // 2,
            pool_size,
            activation,
            conv_do,
            False,
        )
        self.conv4 = ConvBlock(
            num_filters // 2,
            5,
            num_filters // 2,
            pool_size,
            activation,
            conv_do,
            True,
        )
        self.conv5 = ConvBlock(
            num_filters // 2,
            2,
            num_filters // 2,
            0,
            activation,
            0,
            True,
        )
        # Calculate dense input size
        dummy_input = torch.zeros(1, seq_channels, seq_len, requires_grad=False)
        dummy_output = self.conv5(
            self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
        )
        dense_in = dummy_output.view(1, -1).size(1)

        # Dense layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(pre_dense_do)
        self.dense = DenseBlock(dense_in, dense_out, activation, dense_do)
        self.fc = nn.Linear(dense_out, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.fc(x)

        return torch.sigmoid(x)

    def get_params(self):
        return {
            "architecture": self.__class__.__name__,
            "num_classes": self.num_classes,
            "filter_size": self.filter_size,
            "num_filters": self.num_filters,
            "pool_size": self.pool_size,
            "dense_out": self.dense_out,
            "activation": self.activation,
            "seq_len": self.seq_len,
            "seq_channels": self.seq_channels,
            "conv_do": self.conv_do,
            "dense_do": self.dense_do,
            "pre_dense_do": self.pre_dense_do,
        }
