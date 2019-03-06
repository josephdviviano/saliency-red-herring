import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


# Taken from the pytorch tutorial: https://github.com/pytorch/examples/tree/master/mnist
@register.setmodelname("SimpleCNN")
class CNN(nn.Module):
    def __init__(self, flat_layer=440, num_class=2):
        super(CNN, self).__init__()
        # TODO: Take this out of sequential and have a var to capture activations per layer
        #       or see whether per layer activations can be accessed within sequential
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
        )
        self.out = nn.Linear(flat_layer, num_class)

    def forward(self, x):
        # TODO: store all activations per layer with each pass (if they can't be accessed via sequential)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
