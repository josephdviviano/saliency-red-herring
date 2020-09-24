"""Modified ResNet in PyTorch from Torchvison.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

Downloaded on Jul 26th 2020 from:
https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py
"""

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import activmask.utils.register as register
import torchvision
from activmask.models.loss import compare_activations, get_grad_contrast, get_grad_rrr
from activmask.models.utils import shuffle_outside_mask, Dummy
from activmask.models.linear import Discriminator


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, save=False):
        """Pass through x, optionally saving all pre-activation states."""
        all_activations = []

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if save:
            all_activations.append(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection.
        if save:
            all_activations.append(out)
        out = self.relu(out)

        return (out, all_activations)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, save=False):
        all_activations = []
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if save:
            all_activations.append(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if save:
            all_activations.append(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection.
        if save:
            all_activations.append(out)
        out = self.relu(out)

        return (out, all_activations)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, n_chan=1, save_acts=[5]):
        super(ResNet, self).__init__()

        assert all(0 <= i <= 5 for i in save_acts)
        assert n_chan >= 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.save_acts = save_acts

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # NOTE: Torchvision implementation assumes the following settings --
        #       kernel_size=7, stride=2, padding=3.
        #       We found those settings to be detremental to our localization,
        #       And replaced them with much smaller values, that led to larger
        #       feature maps throughout the network.
        self.conv1 = nn.Conv2d(n_chan, self.inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Sequential(
        #    nn.Linear(512 * 13 * 13 * block.expansion, 1024),
        #    nn.BatchNorm1d(num_features=1024),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(1024, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.ModuleList(layers)

    def _run_layers(self, layers, out, save=False):
        all_activations = []
        for layer in layers:
            out, activations = layer(out, save=save)
            all_activations.extend(activations)

        return (out, all_activations)

    def _forward_impl(self, x):

        self.all_activations = []  # Reset for all passes of network.

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        if 0 in self.save_acts:
            self.all_activations.extend([x])

        x = self.relu(x)
        # Note: Original torchvision implementation has a maxpool operation
        #       here, but this has a detremental effect on the gradients
        #       observed using our gradient computation approach, so it was
        #       removed for all experiments presented in this work.
        # x = self.maxpool(x)

        x, acts = self._run_layers(self.layer1, x, save=1 in self.save_acts)
        self.all_activations.extend(acts)
        x, acts = self._run_layers(self.layer2, x, save=2 in self.save_acts)
        self.all_activations.extend(acts)
        x, acts = self._run_layers(self.layer3, x, save=3 in self.save_acts)
        self.all_activations.extend(acts)
        x, acts = self._run_layers(self.layer4, x, save=4 in self.save_acts)
        self.all_activations.extend(acts)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if 5 in self.save_acts:
            self.all_activations.extend([x])

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        save_acts (list): Save the forward pass pre-activations at specified locations [0, 5].
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

@register.setmodelname("ResNetModel")
class ResNetModel(nn.Module):
    """
    actdiff: l2 distance b/t masked/unmasked data.
    gradmask: sum(abs(d y1-y0) / dx )) outside mask.
    rrr: sum( dlogprob / dx **2) outside mask.
    disc: adversarially-driven invariance to masked/unmasked data.

    """
    def __init__(self, base_size=512, resnet_type="18",
                 actdiff_lamb=0, gradmask_lamb=0, rrr_lamb=0, disc_lamb=0, 
                 disc_lr=0.0001, disc_iter=0, save_acts=[5]):

        assert resnet_type in ["18", "34"]
        assert all([x >= 0 for x in [actdiff_lamb, gradmask_lamb, 
                                     rrr_lamb, disc_lamb, disc_iter]])
        # Discriminator and actdiff penalty are incompatible.
        assert not all([x > 0 for x in [actdiff_lamb, disc_lamb]])

        if actdiff_lamb == 0 and disc_lamb == 0:
            save_acts = []  # Disable saving activations when unneeded.
        elif disc_lamb > 0:
            save_acts = [5]  # Adversarial loss only compatible with final layer.
        elif actdiff_lamb > 0:
            assert len(save_acts) > 0
        assert all([0 <= x <= 5 for x in save_acts])  # Valid save_acts locations.

        super(ResNetModel, self).__init__()

        if resnet_type == "18":
            self.encoder = resnet18(save_acts=save_acts)
        elif resnet_type == "34":
            self.encoder = resnet34(save_acts=save_acts)

        self.bce = nn.BCELoss()
        self.device = None
        self.op_counter = 0

        if disc_lamb > 0:
            _layers = [self.encoder.fc.in_features, 1024, 1024, 1024]
            # Self.D is only optimized using the internal optimizer. The grads
            # for these parameters are only allowed to flow during the forward
            # pass, so the external optimizer cannot influence the weights of
            # the discriminator.
            self.D = Discriminator(layers=_layers)
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr=disc_lr)
            self.disc_iter = disc_iter
        else:
            self.D = Dummy()
            self.D_opt = None
            self.disc_iter = 0

        self.actdiff_lamb = actdiff_lamb
        self.gradmask_lamb = gradmask_lamb
        self,rrr_lamb = rrr_lamb
        self.disc_lamb = disc_lamb
        self.criterion = torch.nn.CrossEntropyLoss()

    def _init_device(self):
        """ Runs only during the first forward pass."""
        if not self.device:
            self.device = next(self.parameters()).device

    def _grad_off(self, parameters):
        for p in parameters:
            p.requires_grad = False

    def _grad_on(self, parameters):
        for p in parameters:
            p.requires_grad = True

    def _assert_no_grads(self):
        """ Tests whether the model has any gradients attached to the tape."""
        assert sum([torch.sum(p.grad) for p in self.parameters()]) == 0

    def _iterate_op_counter(self):
        """ The external optimizer is allowed to backprop when counter=0."""
        self.op_counter += 1
        if self.op_counter >= self.disc_iter:
            self.op_counter = 0

    def _get_disc_loss(self, z_masked, z):
        """ Gets and estimate of the KLD between the encoded z and z_masked.
        loss_g can be used to ensure the encoded data from z_masked is more
        likely to confuse the discriminator. The discriminator D is only
        optimized by the internal optimizer.
        """
        self.D_opt.zero_grad()
        batch_size = z.shape[0]
        ones = torch.ones((batch_size, 1)).to(self.device)
        zero = torch.zeros((batch_size, 1)).to(self.device)

        # Discriminator loss: Real=unmasked, fake=masked, g=unmasked.
        # Match the fake to the real.
        real = self.D(z.detach())
        fake = self.D(z_masked.detach())
        loss_d = (self.bce(real, ones) + self.bce(fake, zero)) * self.disc_lamb
        loss_d.backward()
        self.D_opt.step()

        # Turns gradients of Discriminator off so that the parameters are not
        # updated by loss_g.
        self._grad_off(self.D.parameters())
        real = self.D(z)
        fake = self.D(z_masked)
        loss_g = (self.bce(real, zero) + self.bce(fake, ones)) * self.disc_lamb
        self._grad_on(self.D.parameters())

        return loss_d, loss_g

    def forward(self, X, seg):
        """ Forward pass of the model.
        Args:
          X: the input image [c, w, h].
          seg: mask of the class-discriminative image region.
        Returns:
          output dict of values required to calculate loss.
        """
        self._init_device()
        if self.disc_lamb > 0:
            self.D_opt.zero_grad()  # Zero grads of my disc optimizer.

        # Actdiff/Discriminator: save the activations for the masked pass.
        if self.training and any([self.actdiff_lamb > 0, self.disc_lamb > 0]):
            X_masked = shuffle_outside_mask(X, seg).detach()
            _ = self.encoder(X_masked)
            masked_activations = self.encoder.all_activations
        else:
            masked_activations = []

        y_pred = self.encoder(X)
        activations = self.encoder.all_activations

        return {'y_pred': y_pred,
                'X': X,
                'activations': activations,
                'masked_activations': masked_activations,
                'seg': seg}

    def loss(self, y, outputs):
        device = y.device
        clf_loss = self.criterion(outputs['y_pred'], y)
        actdiff_loss = torch.zeros(1)[0].to(device)
        grad_loss = torch.zeros(1)[0].to(device)
        disc_loss = torch.zeros(1)[0].to(device)
        gen_loss = torch.zeros(1)[0].to(device)

        if self.training and self.actdiff_lamb > 0:
            actdiff_loss = compare_activations(
                outputs['masked_activations'], outputs['activations'])

        if self.training and self.gradmask_lamb > 0:
            gradients = get_grad_contrast(outputs['X'], outputs['y_pred'])
            grad_loss = gradients * outputs['seg'].float()
            grad_loss = grad_loss.abs().sum() * self.gradmask_lamb

        if self.training and self.rrr_lamb > 0:
            gradients = get_grad_rrr(outputs['X'], outputs['y_pred'])
            grad_loss = (gradients * outputs['seg'].float())**2
            grad_loss = grad_loss.sum() * self.rrr_lamb

        if self.training and self.disc_lamb > 0:
            # TODO: gracefully handle indices (some way to do invariance at
            # multiple levels?)
            disc_loss, gen_loss = self._get_disc_loss(
                outputs['masked_activations'][-1], outputs['activations'][-1])

        self._iterate_op_counter()

        losses = {
            'clf_loss': clf_loss,
            'actdiff_loss': actdiff_loss,
            'gradmask_loss': grad_loss,
            'gen_loss': gen_loss}
        noop_losses = {
            'disc_loss': disc_loss.detach()}

        return (losses, noop_losses)
