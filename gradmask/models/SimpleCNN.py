import torch
import torch.nn as nn
import torch.nn.functional as F
import gradmask.utils.register as register


@register.setmodelname("SimpleCNN")
class CNN(nn.Module):
    def __init__(self, flat_layer=440, num_class=2):
        super(CNN, self).__init__()
        
        self.all_activations = []
        
        self.ref = []
        
        self.conv1 = nn.Conv2d(
                        in_channels=1,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                     ) 
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                    )
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(
                        in_channels=32,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                    )
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(
                        in_channels=16,
                        out_channels=8,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                    )
        self.relu4 = nn.ReLU()

        self.out = nn.Linear(flat_layer, num_class)

    def forward(self, x):
        # reset so we only get the activations for this batch
        self.all_activations = []
        
        x = self.conv1(x)
        self.all_activations.append(x)

        x = self.relu1(x)
        self.all_activations.append(x)
        
        x = self.conv2(x)
        self.all_activations.append(x)
        
        x = self.relu2(x)
        self.all_activations.append(x)
        
        x = self.conv3(x)
        self.all_activations.append(x)
        
        x = self.relu3(x)
        self.all_activations.append(x)
        
        x = self.conv4(x)
        self.all_activations.append(x)
        
        x = self.relu4(x)
        self.all_activations.append(x)
        
        x = x.view(x.size(0), -1)
        # assuming we need to capture this last thing as well as a layer?
        
        output = self.out(x)
        return output, x
