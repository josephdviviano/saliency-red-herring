import torch
import torch.nn as nn
import utils.register as register

class DeconvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, in_size):
        super().__init__()

        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.output_size = self.calc_size(in_size, kernel=3, stride=1, pad=1)

    def calc_size(self, in_size, kernel, stride, pad):
        return(int((in_size - kernel + 2*(pad)) / stride) + 1)

    def forward(self, x):
        """Double Convolution."""
        pre_activations = []

        x = self.conv1(x)
        pre_activations.append(x)
        x = self.activation(x)
        x = self.conv2(x)
        pre_activations.append(x)
        x = self.activation(x)

        return(x, pre_activations)


@register.setmodelname("UNet")
class UNet(nn.Module):

    def __init__(self, img_size=300, num_class=2, nc=64, mode='unet'):
        super().__init__()

        assert mode in ['unet', 'ae', 'cnn']

        self.all_activations = []
        self.mode = mode

        # Auxilary layers
        self.maxpool = nn.MaxPool2d(2)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Architecture.
        l1_size = self._calc_layer_size(nc*1, img_size//1)
        l2_size = self._calc_layer_size(nc*2, img_size//2)
        l3_size = self._calc_layer_size(nc*4, img_size//4)
        l4_size = self._calc_layer_size(nc*8, img_size//8)

        # Convolve down to bottleneck
        self.dconv_down1 = DeconvBlock(1,    nc*1, img_size)
        self.dconv_down2 = DeconvBlock(nc*1, nc*2, self.dconv_down1.output_size)
        self.dconv_down3 = DeconvBlock(nc*2, nc*4, self.dconv_down2.output_size)
        self.dconv_down4 = DeconvBlock(nc*4, nc*8, self.dconv_down3.output_size)

        # Make prediction off of bottleneck
        self.fc1 = nn.Linear(l4_size, num_class)

        # Upsample from bottleneck (for reconstruction: ae and unet)
        if mode not 'cnn':
            self.upsample3 = nn.Upsample(size=(img_size//4, img_size//4), 
                                         mode='bilinear', align_corners=True)
            if mode == 'unet':
                in_ch_size = nc*4+nc*8
            elif mode == 'ae':
                in_ch_size = nc*8
            self.dconv_up3 = DeconvBlock(in_ch_size, nc*4, 
                                         self.dconv_down4.output_size)

            self.upsample2 = nn.Upsample(size=(img_size//2, img_size//2),
                                         mode='bilinear', align_corners=True)
            if mode == 'unet':
                in_ch_size = nc*2+nc*4
            elif mode == 'ae':
                in_ch_size = nc*4
            self.dconv_up2 = DeconvBlock(in_ch_size, nc*2, 
                                         self.dconv_down4.output_size)

            self.upsample1 = nn.Upsample(size=(img_size, img_size),
                                         mode='bilinear', align_corners=True)
            if mode == 'unet':
                in_ch_size = nc*1+nc*2
            elif mode == 'ae':
                in_ch_size = nc*2
            self.dconv_up1 = DeconvBlock(in_ch_size, nc*1, 
                                         self.dconv_down4.output_size)

            self.conv_last = nn.Sequential(nn.Conv2d(nc, 1, 1), nn.Sigmoid())

    def _calc_layer_size(self, channels, img_size):
        return(channels * img_size**2)

    def forward(self, x):
        """Outputs predictions from bottleneck as well as reconstruction."""
        # Reset so we only get the activations for this batch.
        self.all_activations = []

        # We skip the whole reconstruction for cnn mode, so we just copy the 
        # input data to the output so as to not break the training loop.
        if mode == 'cnn':
            x_prime = torch.clone(x)

        # Convolve down to the bottleneck, saving activations.
        conv1, pre_activations = self.dconv_down1(x)
        self.all_activations.extend(pre_activations)
        x = self.maxpool(conv1)

        conv2, pre_activations = self.dconv_down2(x)
        self.all_activations.extend(pre_activations)
        x = self.maxpool(conv2)

        conv3, pre_activations = self.dconv_down3(x)
        self.all_activations.extend(pre_activations)
        x = self.maxpool(conv3)

        conv4, pre_activations = self.dconv_down4(x)
        self.all_activations.extend(pre_activations)

        # Generate predictions directly off of bottleneck.
        pred = conv4.view(conv4.size(0), -1)
        pred = self.fc1(pred)

        # Reconstruct the input in AE and Unet mode.
        if mode not 'cnn':
            x = self.upsample3(conv4)
 
            if self.mode == 'unet':
                x = torch.cat([x, conv3], dim=1)
            x, _ = self.dconv_up3(x)  # Don't collect activations for upsamples

            x = self.upsample2(x)
 
            if self.mode == 'unet':
                x = torch.cat([x, conv2], dim=1)
            x, _ = self.dconv_up2(x)

            x = self.upsample1(x)

            if self.mode == 'unet':
                x = torch.cat([x, conv1], dim=1)
            x, _ = self.dconv_up1(x)

            x_prime = self.conv_last(x)

        return(pred, x_prime)