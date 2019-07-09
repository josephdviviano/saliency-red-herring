import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


@register.setmodelname("SimpleCNN3")
class CNN(nn.Module):
    def __init__(self, img_size=100, flat_layer=1, num_class=2):
        super(CNN, self).__init__()
        
        flat_layer = flat_layer * 100
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=8,
#                 kernel_size=3,
#                 stride=2,
#                 padding=0,
#             ),
#             nn.ReLU(),
        )
        
        self.pool1 = nn.AvgPool2d(2, stride=1)
        #self.pool1 = nn.MaxPool2d(40, stride=1)

        self.out = nn.Linear(400, num_class)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = x.view(x.size(0), -1)

        output = self.out(x)
        return output, x
