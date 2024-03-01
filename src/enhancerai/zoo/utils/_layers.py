from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def activations(name):
    """Return the activation function corresponding to the given name."""
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "tanh":
        return torch.tanh
    elif name == "leaky_relu":
        return F.leaky_relu
    elif name == "elu":
        return F.elu
    elif name == "selu":
        return F.selu
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_same_padding(filter_size):
    """Calculate the padding needed for 'SAME' padding in 1D pooling.

    Only works for stride=1. Only really correct for odd filter sizes (torch won't accept tuples).
    """
    return (filter_size - 1) // 2


class ConvBlock(nn.Module):
    """1D Convolutional block with residuals, batch normalization, max pooling, and dropout."""

    def __init__(
        self,
        filters,
        kernel_size,
        in_channels,
        pool_size=2,
        activation="relu",
        dropout=0.25,
        res=False,
    ):
        super().__init__()
        self.res = res
        self.conv = nn.Conv1d(
            in_channels, filters, kernel_size, stride=1, padding="same", bias=False
        )
        self.bn = nn.BatchNorm1d(filters, momentum=0.9)
        self.activation = activations(activation)
        self.pool = (
            nn.MaxPool1d(pool_size, padding=get_same_padding(pool_size))
            if (pool_size > 1)
            else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.res_conv = nn.Conv1d(in_channels, filters, 1, stride=1) if res else None

    def forward(self, x):
        residual = x if self.res else None
        y = self.conv(x)
        y = self.bn(y)
        y = self.activation(y)
        if self.res:
            if x.shape[1] != y.shape[1]:
                residual = self.res_conv(residual)
            y = y + residual
        if self.pool is not None:
            y = self.pool(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class DenseBlock(nn.Module):
    """Dense block with batch normalization and dropout."""

    def __init__(self, in_features, out_features, activation="relu", dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        self.bn = nn.BatchNorm1d(num_features=out_features, momentum=0.9)
        self.activation = activations(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x) if x.shape[0] > 1 else x  # dummies with batch size 1
        x = self.activation(x)
        x = self.dropout(x)
        return x
