import torch
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def shuffle_outside_mask(X, seg):
    """Mask data by randomizing the pixels outside of the mask."""
    X_masked = torch.clone(X)

    # Loop through batch images individually
    for b in range(seg.shape[0]):

        # Get all of the relevant values using this mask.
        b_seg = seg[b, :, :, :]
        tmp = X[b, b_seg]

        # Randomly permute those values.
        b_idx = torch.randperm(tmp.nelement())
        tmp = tmp[b_idx]
        X_masked[b, b_seg] = tmp

    return X_masked


def generate_feedforward_layers(n_init, n_layers, shrink_factor, min_size=1):
    """
    Given three parameters:
        n_initial: the size of the first hidden layer of the network.
        n_layers: the number of layers in the feedforward network.
        shrink_factor: the multiplier applied to layer l to determine the
                       size of layer n+1
    Return a python list defining the hidden layer sizes of the feedforward
    network.
    """
    architecture = []

    assert n_layers > 0

    # Ensures that the number of neurons in a layer is never smaller than
    # min_size neurons.
    for i in range(n_layers):
        architecture.append(max(min_size, math.ceil(n_init * shrink_factor**i)))

    return(architecture)


class Flatten(torch.nn.Module):
    """
    Helper class that allows you to perform a flattening op inside
    nn.Sequential.
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Dummy(torch.nn.Module):
    """
    Helper class that allows you to do nothing. Replaces `lambda x: x`.
    """
    def forward(self, x):
        return(x)


if __name__  == "__main__":
    assert generate_feedforward_layers(100, 1, 0.5) == [100]
    assert generate_feedforward_layers(100, 4, 0.5) == [100, 50, 25, 13]
    assert generate_feedforward_layers(20, 4, 0.5) == [20, 10, 10, 10]
