"""Modified DenseNet in PyTorch from TorchXRayVison.
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
from activmask.models.loss import compare_activations, get_grad_contrast
from activmask.models.utils import shuffle_outside_mask, Dummy
from activmask.models.linear import Discriminator
sys.path.insert(0,"../../../torchxrayvision")
import torchxrayvision as xrv


@register.setmodelname("DenseNetModel")
class DenseNetModel(nn.Module):
    """
    actdiff: l2 distance b/t masked/unmasked data.
    gradmask: sum(abs(grad)) outside mask.
    disc: adversarially-driven invariance to masked/unmasked data.

    """
    def __init__(self, num_classes=2, actdiff_lamb=0, gradmask_lamb=0, disc_lamb=0, 
                 disc_lr=0.0001, disc_iter=0):

        assert actdiff_lamb >= 0
        assert gradmask_lamb >= 0
        assert disc_lamb >= 0
        assert disc_iter >= 0
        # Discriminator and actdiff penalty are incompatible.
        assert not all([x > 0 for x in [actdiff_lamb, disc_lamb]])

        super(ResNetModel, self).__init__()

        # The features are frozen, we only learn the final clasifier.
        self.encoder = xrv.models.DenseNet(weights="all")
        self._grad_off(self.encoder.features.parameters)
        self.fc = nn.Linear(self.encoder.classifier.in_features, num_classes)

        self.bce = nn.BCELoss()
        self.device = None
        self.op_counter = 0

        if disc_lamb > 0:
            _layers = [self.encoder.classifier.in_features, 1024, 1024, 1024]
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

        def _activate_and_pool(x):
            out = F.relu(x, inplace=True)
            return F.adaptive_avg_pool2d(out, (1, 1)).view(x.size(0), -1) 

        # Actdiff/Discriminator: save the activations for the masked pass.
        if self.training and any([self.actdiff_lamb > 0, self.disc_lamb > 0]):
            X_masked = shuffle_outside_mask(X, seg).detach()
            masked_activations = [_activate_and_pool(self.features(X_masked))]
        else:
            masked_activations = []

        activations = _activate_and_pool(self.features(X))
        y_pred = self.fc(activations)
        activations = [activations]

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
