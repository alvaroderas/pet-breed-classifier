"""
Contains PyTorch code to instantiate a waste classifier.

Author: Alvaro Deras
Date: January 13, 2024
"""
import torch
from torch import nn

class WasteClassifier(nn.Module):
    """
    A class representing a convolutional neural network model for classifying waste.
    """
    def __init__(self, input_shape: int, hidden_units: int) -> None:
        """
        Instantiates an instance of the WasteClassifier model.

        Parameter input_shape: the number of input channels
        Precondition: input_shape is an int

        Parameter hidden_units: the number of hidden units between layers
        Precondition: hidden_units is an int
        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=1)
        )
    def forward(self, x: torch.Tensor):
        """
        Returns the output of the model

        Parameter x: the initial input data
        Precondition: x is an instance of torch.Tensor
        """
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))