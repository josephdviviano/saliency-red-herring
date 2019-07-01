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


@register.setmodelname("SimpleUNet")
class UNet(nn.Module):

    def __init__(self, img_size=300, num_class=2, flat_layer=1):
        super().__init__()

        flat_layer = flat_layer * 200
        self.all_activations = []

        # Auxilary layers
        self.maxpool = nn.MaxPool2d(2)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Convolve down to bottleneck
        self.dconv_down1 = DeconvBlock(1, 64, img_size)
        self.dconv_down2 = DeconvBlock(64, 128, self.dconv_down1.output_size)
        self.dconv_down3 = DeconvBlock(128, 256, self.dconv_down2.output_size)
        self.dconv_down4 = DeconvBlock(256, 512, self.dconv_down3.output_size)

        # Upsample from bottleneck (for reconstruction)
        self.upsample3 = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=True)
        self.dconv_up3 = DeconvBlock(256+512, 256, self.dconv_down4.output_size)
        self.upsample2 = nn.Upsample(size=(14, 14), mode='bilinear', align_corners=True)
        self.dconv_up2 = DeconvBlock(128+256, 128, self.dconv_down4.output_size)
        self.upsample1 = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)
        self.dconv_up1 = DeconvBlock(128+64, 64, self.dconv_down4.output_size)
        self.conv_last = nn.Conv2d(64, 1, 1)

        # Make prediction off of bottleneck
        self.fc1 = nn.Linear(512 * 3**2, num_class)
        #self.fc2 = nn.Linear(flat_layer, num_class)


    def forward(self, x):
        """Outputs predictions from bottleneck as well as reconstruction."""
        # Reset so we only get the activations for this batch.
        self.all_activations = []

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

        # Generate predictions, saving activations.
        pred = conv4.view(conv4.size(0), -1)
        pred = self.fc1(pred)
        #self.all_activations.append(pred)
        #pred = self.activation(pred)
        #pred = self.fc2(pred)  # Softmax operator used, so no scaling.

        # Reconstruct the input.
        x = self.upsample3(conv4)
        x = torch.cat([x, conv3], dim=1)
        x, _ = self.dconv_up3(x)  # Don't collect activations for upsamples

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x, _ = self.dconv_up2(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x, _ = self.dconv_up1(x)

        x_prime = self.conv_last(x)
        x_prime = self.sigmoid(x_prime)  # Scale b/t [0 1]

        return(pred, x_prime)
