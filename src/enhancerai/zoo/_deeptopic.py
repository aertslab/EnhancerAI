import torch
import torch.nn as nn

from enhancerai.zoo.utils import ConvBlock, DenseBlock


class DeepTopic(nn.Module):
    """DeepTopic model architecture for topic modeling of chromatin accessibility data."""

    def __init__(
        self,
        num_classes,
        filter_size=17,
        num_filters=1024,
        pool_size=4,
        dense_out=1024,
        activation="relu",
        seq_shape=(4, 500),
        conv_do=0.15,
        dense_do=0.5,
        pre_dense_do=0.5,
    ):
        super().__init__()

        # Conv layers
        self.conv1 = ConvBlock(
            num_filters,
            filter_size,
            seq_shape[0],
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
        dummy_input = torch.zeros(1, *seq_shape, requires_grad=False)
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
