import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


@register.setmodelname("SimpleCNN1")
class SimpleCNN1(nn.Module):
    def __init__(self, img_size, flat_layer=2, num_class=2):
        super(SimpleCNN1, self).__init__()

        self.img_size = img_size
        flat_layer = flat_layer * 100
        self.all_activations = []

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        output_size = self._output_size(img_size, kernel_size=3, stride=1, padding=0)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        output_size = self._output_size(output_size, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        output_size = self._output_size(output_size, kernel_size=3, stride=1, padding=0)

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        output_size = self._output_size(output_size, kernel_size=3, stride=2, padding=0)

        self.fc = nn.Linear(8 * output_size * output_size, num_class)

    def _output_size(self, in_size, kernel_size, stride, padding):
        return int((in_size - kernel_size + 2*(padding)) / stride) + 1

    def forward(self, x):
        # reset so we only get the activations for this batch
        self.all_activations = []

        x = self.conv1(x)
        self.all_activations.append(x)
        x = self.relu1(x)

        x = self.conv2(x)
        self.all_activations.append(x)
        x = self.relu2(x)

        x = self.conv3(x)
        self.all_activations.append(x)
        x = self.relu3(x)

        x = self.conv4(x)
        self.all_activations.append(x)
        x = self.relu4(x)

        x = x.view(x.size(0), -1) # collapses shape to [batch, channels*height*width]

        output = self.fc(x)
        return (output, x)


@register.setmodelname("SimpleCNN4")
class SimpleCNN4(nn.Module):
    def __init__(self, img_size, flat_layer=2, num_class=2):
        super(SimpleCNN4, self).__init__()

        self.img_size = img_size
        flat_layer = flat_layer * 100
        self.all_activations = []
        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)

        output_size = self._output_size(img_size, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=6, stride=1, padding=0)

        output_size = self._output_size(output_size, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)

        output_size = self._output_size(output_size, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=16, stride=1, padding=0)

        self.fc = nn.Linear(64, num_class)

    def _output_size(self, in_size, kernel_size, stride, padding):
        return int((in_size - kernel_size + 2*(padding)) / stride) + 1

    def forward(self, x):
        # Reset so we only get the activations for this batch.
        self.all_activations = []

        # print("input: ", x.shape, "img_size: ", self.img_size)
        x = self.conv1(x)
        self.all_activations.append(x)
        x = self.activation(x)

        x = self.conv2(x)
        self.all_activations.append(x)
        x = self.activation(x)

        x = self.conv3(x)
        self.all_activations.append(x)
        x = self.activation(x)

        x = self.conv4(x)
        self.all_activations.append(x)
        x = self.activation(x)

        # Collapses shape to [batch, channels*height*width].
        output = self.fc(x.view(x.size(0), -1))

        return (output, x)
