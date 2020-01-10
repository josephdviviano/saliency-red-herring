from torch.utils.data import Dataset
import os
import pandas as pd
import skimage, skimage.transform
from skimage.morphology import square
import numpy as np
import utils.register as register
from PIL import Image
import torch
import torchvision.transforms.functional as TF

@register.setdatasetname("SyntheticDataset")
class SyntheticDataset(Dataset):
    def __init__(self, mode, dataroot, blur=0, seed=0, nsamples=32,
                 maxmasks=1, transform=None, new_size=28, distract_noise=0,
                 mask_all=False):

        assert 0 <= maxmasks <= 1

        self.root = dataroot
        self.mode = mode
        self.blur = blur
        self.distract_noise = distract_noise
        self.mask_all = mask_all
        self.maxmasks = maxmasks

        self._all_files = [f for f in os.listdir(self.root) if "seg" not in f and ".csv" not in f]
        self._seg_files = [f for f in os.listdir(self.root) if "seg" in f and ".csv" not in f]

        # random split based on seed
        np.random.seed(seed)
        np.random.shuffle(self._all_files)

        # get the files for each mode
        self.mode_file = mode

        # get the corresponding labels
        all_labels = pd.read_csv("{}/{}_labels.csv".format(self.root, self.mode_file))

        # randomly choose based on nsamples
        n_per_class = nsamples//2

        np.random.seed(seed)
        class0 = all_labels["file"].loc[all_labels["class"] == 0].values
        class1 = all_labels["file"].loc[all_labels["class"] == 1].values
        class0 = np.random.choice(class0, n_per_class, replace=False)
        class1 = np.random.choice(class1, n_per_class, replace=False)

        # get the corresponding segmentation files
        class0_seg = [f.replace("img","seg") for f in class0]
        class1_seg = [f.replace("img","seg") for f in class1]

        self.idx = np.append(class1, class0)
        self.mask_idx = np.append(class1_seg, class0_seg)
        self.labels = np.append(np.ones(len(class1)), np.zeros(len(class0)))

        # masks_selector is 1 for samples that should have a mask, else zero.
        self.masks_selector = np.ones(len(self.idx))
        if maxmasks < 1:
            n_masks_to_rm = round(n_per_class * (1-maxmasks))
            idx_masks_class0_to_rm = np.random.choice(
                np.arange(n_per_class), n_masks_to_rm, replace=False)
            idx_masks_class1_to_rm = np.random.choice(
                np.arange(n_per_class, n_per_class*2), n_masks_to_rm,
                replace=False)

            self.masks_selector[idx_masks_class0_to_rm] = 0
            self.masks_selector[idx_masks_class1_to_rm] = 0

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        img = Image.fromarray(np.load(self.root + "/" + self.idx[index]))
        seg = np.load(self.root + "/" + self.mask_idx[index])

        # Make the segmentation the entire image if it isn't in mask_selector.
        if not self.masks_selector[index]:
            seg = np.ones(seg.shape)

        # If there is a segmentation, blur it a bit.
        if (self.blur > 0) and (np.max(seg) != 0):
            seg = skimage.morphology.dilation(seg, selem=square(self.blur))

        seg = (seg > 0) * 1.

        img = TF.to_tensor(img)
        seg = TF.to_tensor(Image.fromarray(seg))
        label = self.labels[index]

        if self.mode == "train_flip":
            img = torch.flip(img,[2])
            seg = torch.flip(seg,[2])

        if self.distract_noise != 0:
            if np.random.choice([True,False], p=[self.distract_noise, 1-self.distract_noise]):
                img = torch.flip(img,[2])
                seg = torch.flip(seg,[2])

        # Control condition where we mask all data. Used to see if traditional
        # training works.
        if self.mask_all:
            img *= seg

        return (img, seg, int(label))

if __name__ == "__main__":

    import os,sys,inspect
    sys.path.insert(0,"..")
    import datasets, datasets.SyntheticDataset
    import json, medpy, collections, numpy as np, h5py
    import ntpath
    import matplotlib.pyplot as plt

    d = datasets.SyntheticDataset.SyntheticDataset(dataroot="../data/synth2/",
                                                   mode="distractor1",
                                                   blur=1, nsamples=10)

    import IPython; IPython.embed()
