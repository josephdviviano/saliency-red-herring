'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Code from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """Pass through x, saving all pre-activation states."""
        all_activations = []

        out = self.bn1(self.conv1(x))
        all_activations.append(out)
        out = self.activation(out)

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        all_activations.append(out)
        out = self.activation(out)
        return (out, all_activations)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        all_activations = []

        out = self.bn1(self.conv1(x))
        all_activations.append(out)
        out = self.activation(out)

        out = self.bn2(self.conv2(out))
        all_activations.append(out)
        out = self.activation(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        all_activations.append(out)
        out = self.activation(out)
        return (out, all_activations)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, base_size=512,
                 avg_pool_size=4, avg_pool_stride=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.all_activations = []
        self.activation = nn.ReLU()
        self.avg_pool_size = avg_pool_size
        self.avg_pool_stride = avg_pool_stride

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(base_size*block.expansion, num_classes)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def _run_layers(self, layers, out):
        all_activations = []
        for layer in layers:
            out, activations = layer(out)
            all_activations.extend(activations)

        return (out, all_activations)

    def forward(self, x):
        self.all_activations = []  # Reset for all passes of network.

        out = self.bn1(self.conv1(x))
        self.all_activations.append(out)
        out = self.activation(out)

        out, activations = self._run_layers(self.layer1, out)
        self.all_activations.extend(activations)

        out, activations = self._run_layers(self.layer2, out)
        self.all_activations.extend(activations)

        out, activations = self._run_layers(self.layer3, out)
        self.all_activations.extend(activations)

        out, activations = self._run_layers(self.layer4, out)
        self.all_activations.extend(activations)

        out = F.avg_pool2d(out, self.avg_pool_size, self.avg_pool_stride)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(base_size=512, avg_pool_size=4):
    return ResNet(BasicBlock, [2,2,2,2], base_size=base_size)

def ResNet34(base_size=512, avg_pool_size=4):
    return ResNet(BasicBlock, [3,4,6,3], base_size=base_size)

def ResNet50(base_size=512, avg_pool_size=4, avg_pool_stride=1):
    return ResNet(Bottleneck, [3,4,6,3], base_size=base_size, avg_pool_size=avg_pool_size, avg_pool_stride=avg_pool_stride)

def ResNet101(base_size=512, avg_pool_size=4):
    return ResNet(Bottleneck, [3,4,23,3], base_size=base_size)

def ResNet152(base_size=512, avg_pool_size=4):
    return ResNet(Bottleneck, [3,8,36,3], base_size=base_size)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,24,24))
    print(y.size())


@register.setmodelname("ResNetSmall")
class ResNetSmall(nn.Module):

    def __init__(self, img_size=1, base_size=512):
        super(ResNetSmall, self).__init__()
        self.model = ResNet18(base_size=base_size, avg_pool_size=4)
        self.all_activations = []

    def forward(self, x):
        out = self.model(x)
        self.all_activations = self.model.all_activations
        return out, None


@register.setmodelname("ResNetBig")
class ResNetBig(nn.Module):

    def __init__(self, img_size=1, base_size=512):
        super(ResNetBig, self).__init__()
        self.model = ResNet50(base_size=base_size, avg_pool_size=12, avg_pool_stride=36)
        self.all_activations = []

    def forward(self, x):
        out = self.model(x)
        self.all_activations = self.model.all_activations
        return out, None
