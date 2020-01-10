import torch

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
