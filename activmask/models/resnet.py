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
import activmask.utils.register as register
import torchvision
from activmask.models.loss import compare_activations, get_grad_contrast
from activmask.models.utils import shuffle_outside_mask

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

    def forward(self, x, save=False):
        """Pass through x, optionally saving all pre-activation states."""
        all_activations = []

        out = self.bn1(self.conv1(x))
        if save:
            all_activations.append(out)
        out = self.activation(out)

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if save:
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

    def forward(self, x, save=False):
        """Pass through x, optionally saving all pre-activation states."""
        all_activations = []

        out = self.bn1(self.conv1(x))
        if save:
            all_activations.append(out)
        out = self.activation(out)

        out = self.bn2(self.conv2(out))
        if save:
            all_activations.append(out)
        out = self.activation(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if save:
            all_activations.append(out)
        out = self.activation(out)
        return (out, all_activations)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, base_size=512,
                 avg_pool_size=4, avg_pool_stride=1, save_acts=[0, 1, 2, 3, 4]):
        super(ResNet, self).__init__()

        assert all(i <= 4 for i in save_acts)
        assert all(i >= 0 for i in save_acts)

        self.in_planes = 64
        self.all_activations = []
        self.activation = nn.ReLU()
        self.avg_pool_size = avg_pool_size
        self.avg_pool_stride = avg_pool_stride
        self.save_acts = save_acts

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(base_size*block.expansion, num_classes)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def _run_layers(self, layers, out, save=False):
        all_activations = []
        for layer in layers:
            out, activations = layer(out, save=save)
            all_activations.extend(activations)

        return (out, all_activations)

    def forward(self, x):
        self.all_activations = []  # Reset for all passes of network.

        out = self.bn1(self.conv1(x))
        if 0 in self.save_acts:
            self.all_activations.append(out)
        out = self.activation(out)

        out, activations = self._run_layers(self.layer1, out,
                                            save=1 in self.save_acts)
        self.all_activations.extend(activations)

        out, activations = self._run_layers(self.layer2, out,
                                            save=2 in self.save_acts)
        self.all_activations.extend(activations)

        out, activations = self._run_layers(self.layer3, out,
                                            save=3 in self.save_acts)
        self.all_activations.extend(activations)

        out, activations = self._run_layers(self.layer4, out,
                                            save=4 in self.save_acts)
        self.all_activations.extend(activations)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out


def ResNet18(base_size=512, avg_pool_size=4, save_acts=[0, 1, 2, 3, 4]):
    return ResNet(BasicBlock, [2,2,2,2], base_size=base_size, save_acts=save_acts)

def ResNet34(base_size=512, avg_pool_size=4, save_acts=[0, 1, 2, 3, 4]):
    return ResNet(BasicBlock, [3,4,6,3], base_size=base_size, save_acts=save_acts)

def ResNet50(base_size=512, avg_pool_size=4, avg_pool_stride=1,
             save_acts=[0, 1, 2, 3, 4]):
    return ResNet(Bottleneck, [3,4,6,3],
                  base_size=base_size,
                  avg_pool_size=avg_pool_size,
                  avg_pool_stride=avg_pool_stride,
                  save_acts=save_acts)

def ResNet101(base_size=512, avg_pool_size=4, save_acts=[0, 1, 2, 3, 4]):
    return ResNet(Bottleneck, [3,4,23,3], base_size=base_size, save_acts=save_acts)

def ResNet152(base_size=512, avg_pool_size=4, save_acts=[0, 1, 2, 3, 4]):
    return ResNet(Bottleneck, [3,8,36,3], base_size=base_size, save_acts=save_acts)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,24,24))
    print(y.size())


@register.setmodelname("ResNetModel")
class ResNetModel(nn.Module):
    def __init__(self, base_size=512, resnet_type="18",  actdiff_lamb=0,
                 gradmask_lamb=0, save_acts=[0, 1, 2, 3, 4]):


        assert resnet_type in ["18", "34"]
        assert actdiff_lamb >= 0
        assert gradmask_lamb >= 0

        super(ResNetModel, self).__init__()

        if resnet_type == "18":
            self.model = ResNet18(base_size=base_size, avg_pool_size=4,
                                  save_acts=save_acts)
        elif resnet_type == "34":
            self.model = ResNet34(base_size=base_size, avg_pool_size=4,
                                  save_acts=save_acts)
        self.all_activations = []
        self.actdiff_lamb = actdiff_lamb
        self.gradmask_lamb = gradmask_lamb
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, X, seg):

        # Actdiff: save the activations for the masked pass (training only).
        if self.training and self.actdiff_lamb > 0:
            X_masked = shuffle_outside_mask(X, seg).detach()
            _ = self.model(X_masked)
            masked_activations = self.model.all_activations
        else:
            masked_activations = []

        y_pred = self.model(X)
        activations = self.model.all_activations

        return {'y_pred': y_pred,
                'X': X,
                'activations': activations,
                'masked_activations': masked_activations,
                'seg': seg}

    def loss(self, y, outputs):
        device = y.device
        clf_loss = self.criterion(outputs['y_pred'], y)

        if self.training and self.actdiff_lamb > 0:
            actdiff_loss = compare_activations(
                outputs['masked_activations'], outputs['activations'])
        else:
            actdiff_loss = torch.zeros(1)[0].to(device)

        if self.training and self.gradmask_lamb > 0:
            gradients = get_grad_contrast(outputs['X'], outputs['y_pred'], y)
            grad_loss = gradients * outputs['seg'].float()
            grad_loss = grad_loss.abs().sum() * self.gradmask_lamb
        else:
            grad_loss = torch.zeros(1)[0].to(device)

        return {'clf_loss': clf_loss,
                'actdiff_loss': actdiff_loss,
                'gradmask_loss': grad_loss}
