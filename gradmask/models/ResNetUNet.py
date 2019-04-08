import torch
import torch.nn as nn
from torchvision import models
import math

import utils.register as register

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

@register.setmodelname("ResNetUNet")
class ResNetUNet(nn.Module):

    def __init__(self, flat_layer=1, n_class=2, img_size=300):
        super().__init__()
        
        if img_size % 32:
            self.padding = 1
        else:
            self.padding = 0
        
        flat_layer = flat_layer * 200

        # Use ResNet18 as the encoder with the pretrained weights
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, self.padding)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 128, 1, self.padding)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 256, 1, self.padding)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 512, 1, self.padding)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 1024, 1, self.padding)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up2 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up1 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        output_size = self.get_fc_layer_size(img_size)
        self.fc1 = nn.Linear(16 * output_size * output_size, flat_layer)
        #self.fc1 = nn.Linear(32768, flat_layer)
        self.relu_last = nn.ReLU()
        
        self.out = nn.Linear(flat_layer, n_class)
    
    def get_fc_layer_size(self, img_size):
        # self.conv_last = nn.Conv2d(64, n_class, 1)
        output_size = math.floor(img_size / 32) # after the 4th layer, H or W / 32
        # fourth 1x1 layer, kernel = 1, stride=1, padding=0 + upsample (size = size * 2)
        output_size = self.output_size(output_size, 1, 1, 0) * 2
        
        # concat size along the channels dimension so doesn't affect height/width, layer_3 == output_size_l4
        # ... I think... lol
        for i in range(5):
            # for five conv_up + resamples (0-4)
            output_size = self.output_size(output_size, 3, 1, 1) * 2
        
        output_size = self.output_size(output_size, 3, 1, 1) # for the last conv_original_2 layer
        return output_size
        
    def output_size(self, in_size, kernel_size, stride, padding):

        output = math.floor((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)
        
    def forward(self, input_img):
        print("PADDING: ", self.padding)

        # hack lol
        input_img = torch.cat([input_img, input_img, input_img], dim=1)
        x_original = self.conv_original_size0(input_img)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(input_img)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        print("layer4 shape: ", layer4.shape)
        # Upsample the last/bottom layer
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        print("layer 4 1x1 shape: ", layer4.shape)
        # Create the shortcut from the encoder
        layer3 = self.layer3_1x1(layer3)
        print("l3 x shape: ", x.shape, "layer3 shape: ", layer3.shape)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        print("l1 x shape: ", x.shape, "layer1 shape: ", layer2.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        
        # x = layer4 
        x = x.view(x.size(0), -1)
        x = self.fc1(x) # self.conv_last(x)
        # print("out shape: ", x.shape)

        x = self.relu_last(x)
        
        out = self.out(x)
        # print("last shape: ", out.shape)
        return out, x
