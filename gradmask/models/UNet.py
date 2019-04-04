import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


@register.setmodelname("UNet")
class UNet(nn.Module):
    def __init__(self, img_size=300):
        super(UNet, self).__init__()
        
        self.activations = []
        
        self.unet_enc_1, out_size = unet_encode_block(layer_num=1, img_size)
        
        self.unet_enc_2, out_size = unet_encode_block(layer_num=2, out_size)
        
        self.unet_enc_3, out_size = unet_encode_block(layer_num=3, out_size)
                

    def unet_encode_block(self, layer_num, in_size, base_filter_size=32):
        internal_layers = []
        if layer_num != 1:
            # if it's not the first layer, do max pool first and multiply conv filters by layer_num
            internal_layers.append(nn.MaxPool2d(2))
            # number of channels becomes whatever the number of output filters was on the last layer
            in_channels = in_size
            # get the output size for the max pool layer
            output_size = self.outputSize(in_size, kernel_size=2, stride=2, padding=0)
        else:
            # stuff for the first layer - 1 channel and output_size = whatever the image size is!
            output_size = in_size
            in_channels = 1
            
        for i in range(2):
            # do conv, ELU, BN twice through and store the layers
            internal_layers.append(nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=layer_num*base_filter_size,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                     ))
            output_size = self.outputSize(output_size, kernel_size=2, stride=2, padding=0)
            internal_layers.append(nn.ELU())
            internal_layers.append(nn.BatchNorm2d(layer_num*base_filter_size))  
        
        return internal_layers, output_size
    
    def unet_decode_block(self, in_size, is_last=False):
        internal_layers = []
        if not is_last:
            # do the thing
            if layer_num != 1:
                # not the first decode block, so do dropout, 2x(conv, ELU), upsample
            
            else:
                # first decode block so just do upsample
            
        return internal_layers, size
    
    def output_size(self, in_size, kernel_size, stride, padding):

        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)

    def forward(self, x):
        self.activations = []
        
        for layer in self.unet_enc_1:
            x = layer(x)
            self.activations.append(x)
        
        skip_1 = x # output of the first block for skip connections
        
        for layer in self.unet_enc_2:
            x = layer(x)
            self.activations.append(x)
        
        skip_2 = x
        
        for layer in self.unet_enc_3:
            x = layer(x)
            self.activations.append(x)
        
        skip_3 = x
        
        
        
        return x
