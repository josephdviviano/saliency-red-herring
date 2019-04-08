import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


@register.setmodelname("UNet")
class UNet(nn.Module):
    def __init__(self, img_size=300, num_class=2, flat_layer=1):
        super(UNet, self).__init__()
        
        flat_layer = flat_layer * 200

        self.activations = []
        
        # encode down the blocks
        self.unet_enc_1, out_size = self.unet_encode_block(1, img_size)
        
        self.unet_enc_2, out_size = self.unet_encode_block(2, out_size)
        
        self.unet_enc_3, out_size = self.unet_encode_block(3, out_size)
                
        self.unet_enc_4, out_size = self.unet_encode_block(4, out_size)
        
        # decode the blocks up
        self.unet_dec_4, out_size = self.unet_decode_block(4, out_size)

        self.unet_dec_3, out_size = self.unet_decode_block(3, out_size)

        self.unet_dec_2, out_size = self.unet_decode_block(2, out_size)

        self.unet_dec_1, out_size = self.unet_decode_block(1, out_size)

        self.fc1 = nn.Linear(17 * out_size * out_size, flat_layer)

        self.out = nn.Linear(flat_layer, num_class)

    def unet_encode_block(self, block_num, in_size, base_filter_size=32):
        internal_layers = []
        if block_num != 1:
            # if it's not the first layer, do max pool first and multiply conv filters by layer_num
            internal_layers.append(nn.MaxPool2d(2).cuda())
            # number of channels becomes whatever the number of output filters was on the last layer
            in_channels = (block_num - 1) * base_filter_size
            # get the output size for the max pool layer
            output_size = self.output_size(in_size, kernel_size=2, stride=2, padding=0)
        else:
            # stuff for the first layer - 1 channel and output_size = whatever the image size is!
            output_size = in_size
            in_channels = 1
            
        for i in range(2):
            # do conv, ELU, BN twice through and store the layers
            internal_layers.append(nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=block_num*base_filter_size,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                     ).cuda())
            output_size = self.output_size(output_size, kernel_size=3, stride=1, padding=0)
            internal_layers.append(nn.ELU().cuda())
            internal_layers.append(nn.BatchNorm2d(block_num*base_filter_size).cuda())  
            
            in_channels = base_filter_size*block_num

        return internal_layers, output_size
    
    def unet_decode_block(self, block_num, in_size, base_filter_size=32):
        internal_layers = []
        # do the thing
        output_size = in_size
        if block_num != 4:
            # not the first (i.e. coming from the fourth encoder block) decode block, so do dropout, 2x(conv, ELU), upsample
            internal_layers.append(nn.Dropout())
            in_channels = (block_num + 1) * base_filter_size
            for i in range(2):
                
                internal_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=block_num * base_filter_size,
                    kernel_size=3,
                    stride=1,
                    padding=0
                ))
                internal_layers.append(nn.ELU())
                output_size = self.output_size(output_size, kernel_size=3, stride=1, padding=0)
                in_channels = block_num * base_filter_size
            
            if block_num != 1:
                internal_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                output_size = output_size * 2
        else:
            # first decode block so just do upsample
            internal_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            
        return internal_layers, output_size
    
    def output_size(self, in_size, kernel_size, stride, padding):

        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)

    def forward(self, x):
        self.activations = []
        
        for layer in self.unet_enc_1:
            x = layer(x).cuda()
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
        
        for layer in self.unet_enc_4:
            x = layer(x)
            self.activations.append(x)
        
        for layer in self.unet_dec_4:
            x = layer(x)
            self.activations.append(x)
        
        x = torch.cat([x, skip_3], dim=1)

        for layer in self.unet_dec_3:
            x = layer(x)
            self.activations.append(x)

        x = torch.cat([x, skip_2], dim=1)

        for layer in self.unet_dec_2:
            x = layer(x)
            self.activations.append(x)

        x = torch.cat([x, skip_1], dim=1)

        for layer in self.unet_dec_1:
            x = layer(x)
            self.activations.append(x)

        x = self.fc1(x)
        self.activations.append(x)
        out = self.out(x)
        
        return out, x
