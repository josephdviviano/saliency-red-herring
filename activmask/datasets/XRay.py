from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys
import pandas as pd
import pickle
import skimage
import skimage.draw
import tarfile, glob
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
import utils.register as register


def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    return sample


@register.setdatasetname("XRayDataset")
class JointDataset():
    def __init__(self, d1data, d1csv, d2data, d2csv, ratio=0.5, mode="train",
                 seed=0, transform=None, nsamples=None, maxmasks=None,
                 new_size=None):

        splits = np.array([0.5,0.25,0.25])
        assert mode in ['train', 'valid', 'test']
        assert np.sum(splits) == 1.0

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.new_size = new_size
        self.dataset1 = NIHXrayDataset(d1data, d1csv, seed=seed)
        self.dataset2 = PCXRayDataset(d2data, d2csv, seed=seed)

        all_imageids = np.concatenate([np.arange(len(self.dataset1)),
                                       np.arange(len(self.dataset2))]).astype(int)
        all_idx = np.arange(len(all_imageids)).astype(int)
        all_labels = np.concatenate([self.dataset1.labels,
                                     self.dataset2.labels]).astype(int)
        all_site = np.concatenate([np.zeros(len(self.dataset1)),
                                   np.ones(len(self.dataset2))]).astype(int)

        idx_sick = all_labels==1
        n_per_category = np.min([sum(idx_sick[all_site==0]),
                                 sum(idx_sick[all_site==1])])
        print("n_per_category={}".format(n_per_category))

        all_0_neg = all_idx[np.where((all_site==0) & (all_labels==0))]
        all_0_neg = np.random.choice(all_0_neg, n_per_category, replace=False)
        all_0_pos = all_idx[np.where((all_site==0) & (all_labels==1))]
        all_0_pos = np.random.choice(all_0_pos, n_per_category, replace=False)
        all_1_neg = all_idx[np.where((all_site==1) & (all_labels==0))]
        all_1_neg = np.random.choice(all_1_neg, n_per_category, replace=False)
        all_1_pos = all_idx[np.where((all_site==1) & (all_labels==1))]
        all_1_pos = np.random.choice(all_1_pos, n_per_category, replace=False)

        # TRAIN
        train_0_neg = np.random.choice(
            all_0_neg, int(n_per_category*ratio*splits[0]*2), replace=False)
        train_0_pos = np.random.choice(
            all_0_pos, int(n_per_category*(1-ratio)*splits[0]*2), replace=False)
        train_1_neg = np.random.choice(
            all_1_neg, int(n_per_category*(1-ratio)*splits[0]*2), replace=False)
        train_1_pos = np.random.choice(
            all_1_pos, int(n_per_category*ratio*splits[0]*2), replace=False)

        # REDUCE POST-TRAIN
        all_0_neg = np.setdiff1d(all_0_neg, train_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, train_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, train_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, train_1_pos)

        print("TRAIN: neg={}, pos={}".format(len(train_0_neg)+len(train_1_neg),
                                             len(train_0_pos)+len(train_1_pos)))

        # VALID
        valid_0_neg = np.random.choice(
            all_0_neg, int(n_per_category*(1-ratio)*splits[1]*2), replace=False)
        valid_0_pos = np.random.choice(
            all_0_pos, int(n_per_category*ratio*splits[1]*2), replace=False)
        valid_1_neg = np.random.choice(
            all_1_neg, int(n_per_category*ratio*splits[1]*2), replace=False)
        valid_1_pos = np.random.choice(
            all_1_pos, int(n_per_category*(1-ratio)*splits[1]*2), replace=False)

        # REDUCE POST-VALID
        all_0_neg = np.setdiff1d(all_0_neg, valid_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, valid_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, valid_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, valid_1_pos)

        print("VALID: neg={}, pos={}".format(len(valid_0_neg)+len(valid_1_neg),
                                             len(valid_0_pos)+len(valid_1_pos)))

        # TEST
        test_0_neg = all_0_neg
        test_0_pos = all_0_pos
        test_1_neg = all_1_neg
        test_1_pos = all_1_pos

        print("TEST: neg={}, pos={}".format(len(test_0_neg)+len(test_1_neg),
                                            len(test_0_pos)+len(test_1_pos)))

        if mode == "train":
            self.select_idx = np.concatenate([train_0_neg, train_0_pos,
                                              train_1_neg, train_1_pos])
        elif mode == "valid":
            self.select_idx = np.concatenate([valid_0_neg, valid_0_pos,
                                              valid_1_neg, valid_1_pos])
        elif mode == "test":
            self.select_idx = np.concatenate([test_0_neg, test_0_pos,
                                              test_1_neg, test_1_pos])
        else:
            raise Exception("unknown mode")

        self.imageids = all_imageids[self.select_idx]
        self.labels = all_labels[self.select_idx]
        self.site = all_site[self.select_idx]
        self.masks_selector = np.ones(len(self.site))

        # Image Resizing.
        FINAL_SIZE = (self.new_size, self.new_size)
        self.resize = XRayResizer(FINAL_SIZE)

        # Mask
        rr, cc = skimage.draw.ellipse(FINAL_SIZE[0]//2, FINAL_SIZE[1]//2,
                                      FINAL_SIZE[0]/2.5, FINAL_SIZE[1]/4)
        self.seg = np.zeros(FINAL_SIZE)
        self.seg[rr, cc] = 1

    def __len__(self):
        return len(self.imageids)

    def __getitem__(self, idx):

        if self.site[idx] == 0:
            dataset = self.dataset1
        else:
            dataset = self.dataset2

        img, _, _ = dataset[self.imageids[idx]]
        site = self.site[idx]
        seg = self.seg[None, :, :]

        # Convert to properly-sized tensors
        img = img[0, :, :]
        if self.new_size != img.shape[1]:
            img = self.resize(img)
        img = TF.to_tensor(img)
        seg = TF.to_tensor(seg).permute([1, 0, 2])

        return (img, seg), self.labels[idx], self.masks_selector[idx]


class NIHXrayDataset():

    def __init__(self, datadir, csvpath, transform=None, nrows=None, seed=0,
                 pure_labels=False):

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datadir = datadir
        self.transform = transform
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        # Load data
        self.csv = pd.read_csv(csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]

        # Get our two classes.
        idx_sick = self.csv["Finding Labels"].str.contains("Cardiomegaly")
        idx_heal = self.csv["Finding Labels"].str.contains("No Finding")

        # Exposed for our dataloader wrapper.
        self.csv['labels'] = 0
        self.csv.loc[idx_sick, 'labels'] = 1
        self.csv = self.csv[idx_sick | idx_heal]
        self.labels = self.csv['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im = imread(
            os.path.join(self.datadir, self.csv['Image Index'].iloc[idx]))
        im = normalize(im, self.MAXVAL)

        # Check that images are 2D arrays
        if len(im.shape) > 2:
            im = im[:, :, 0]
        if len(im.shape) < 2:
            print("error, dimension lower than 2 for image {}".format(
                self.Data['Image Index'][idx]))

        # Add color channel
        im = im[None, :, :]

        # Tranform
        if self.transform:
            im = self.transform(im)

        seg = np.ones(im.shape)

        # self.csv['Image Index'][idx]
        return im, seg, self.labels.iloc[idx]


class PCXRayDataset():

    def __init__(self, datadir, csvpath, transform=None, pretrained=False,
                 flat_dir=True, seed=0):
        # Removed "Dataset" super class...
        #super(PCXRayDataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.datadir = datadir
        self.transform = transform
        self.pretrained = pretrained
        self.flat_dir = flat_dir
        self.csv = pd.read_csv(csvpath)
        self.MAXVAL = 65535

        # Keep only the PA view.
        idx_pa = self.csv['ViewPosition_DICOM'].str.contains("POSTEROANTERIOR")
        idx_pa[idx_pa.isnull()] = False
        self.csv = self.csv[idx_pa]

        # Our two classes.
        idx_sick = self.csv['Labels'].str.contains('cardiomegaly')
        idx_sick[idx_sick.isnull()] = False
        idx_heal = self.csv['Labels'].str.contains('normal')
        idx_heal[idx_heal.isnull()] = False

        # Exposed for our dataloader wrapper.
        self.csv['labels'] = 0
        self.csv.loc[idx_sick, 'labels'] = 1
        self.csv = self.csv[idx_sick | idx_heal]
        self.labels = self.csv['labels']

    def __len__(self):
        return len(self.csv.labels)

    def __getitem__(self, idx):

        label = self.labels.iloc[idx]
        imgid = self.csv.iloc[idx]['ImageID']
        img_path = os.path.join(self.datadir, imgid)
        img = np.array(Image.open(img_path))[..., np.newaxis]
        img = normalize(img, self.MAXVAL)
        img = np.transpose(img, [-1, 0, 1])

        # Add color channel
        if self.pretrained:
            img = np.repeat(img, 3, axis=-1)

        if self.transform is not None:
            img = self.transform(img)

        seg = np.ones(img.shape)

        return img, seg, label


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        to_tensor = transforms.ToTensor()
        sample['PA'] = to_tensor(sample['PA'])
        sample['L'] = to_tensor(sample['L'])

        return sample


class XRayResizer(object):
    def __init__(self, size):
        self.to_pil = transforms.ToPILImage(mode="F")
        self.resizer = transforms.Resize(size)

    def __call__(self, x):
        x = self.to_pil(x)
        x = self.resizer(x)
        return(x)


class ToPILImage(object):
    """Convert ndarrays in sample to PIL images."""
    def __call__(self, x):
        to_pil = transforms.ToPILImage()
        return to_pil(x)



class GaussianNoise(object):
    """
    Adds Gaussian noise to the PA and L (mean 0, std 0.05)
    """
    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        pa_img += torch.randn_like(pa_img) * 0.05
        l_img += torch.randn_like(l_img) * 0.05

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


class RandomRotation(object):
    """
    Adds a random rotation to the PA and L (between -5 and +5).
    """
    def rot(self, img, r):
        new_img = scipy.ndimage.rotate(img,r, axes=[1,2], reshape=False, cval=np.min(img))
        return new_img
    
    def __call__(self, img, seg):
        rot_amount = np.random.rand() * 10.- 5.

        img = self.rot(img, rot_amount)
        seg = self.rot(seg, rot_amount)
        
        return img, seg
