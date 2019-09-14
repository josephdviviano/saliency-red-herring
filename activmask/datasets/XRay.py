from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import os,sys
import numpy as np
import pandas as pd
import pickle
import skimage
import tarfile, glob
import torch
import torch.nn.functional as F
import torchvision.models as models
import utils.register as register
import skimage.draw



def normalize(sample, maxval):
    sample = 2 * (sample.astype(np.float32) / maxval) - 1.
    return sample


@register.setdatasetname("XRayDataset")
class JointDataset():
    def __init__(self, d1data, d1csv, d2data, d2csv, ratio=0.5, mode="train",
                 seed=0, transform=None, nsamples=None, maxmasks=None,
                 new_size=None):
        self.dataset1 = NIHXrayDataset(d1data, d1csv)
        self.dataset2 = PCXRayDataset(d2data, d2csv)

        splits = np.array([0.5,0.25,0.25])

        np.random.seed(seed)
        all_imageids = np.concatenate([np.arange(len(self.dataset1)),
                                       np.arange(len(self.dataset2))]).astype(int)
        all_idx = np.arange(len(all_imageids)).astype(int)
        all_labels = np.concatenate([self.dataset1.labels,
                                     self.dataset2.labels]).astype(int)
        all_site = np.concatenate([np.zeros(len(self.dataset1)),
                                   np.ones(len(self.dataset2))]).astype(int)

        idx_sick = all_labels==1
        total_per_class = np.min([sum(idx_sick[all_site==0]),
                                  sum(idx_sick[all_site==1])])
        print("total_per_class", total_per_class)

        train_0_neg = all_idx[np.where((all_site==0) & (all_labels==0))]
        train_0_neg = np.random.choice(train_0_neg, total_per_class,
                                       replace=False)
        train_0_pos = all_idx[np.where((all_site==0) & (all_labels==1))]
        train_0_pos = np.random.choice(train_0_pos, total_per_class,
                                       replace=False)
        train_1_neg = all_idx[np.where((all_site==1) & (all_labels==0))]
        train_1_neg = np.random.choice(train_1_neg, total_per_class,
                                       replace=False)
        train_1_pos = all_idx[np.where((all_site==1) & (all_labels==1))]
        train_1_pos = np.random.choice(train_1_pos, total_per_class,
                                       replace=False)

        splits = (splits*total_per_class).astype(int)
        if mode == "train":
            train_0_neg = train_0_neg[:splits[0]]
            train_0_pos = train_0_pos[:splits[0]]
            train_1_neg = train_1_neg[:splits[0]]
            train_1_pos = train_1_pos[:splits[0]]
        elif mode == "valid":
            train_0_neg = train_0_neg[splits[0]:splits[0]+splits[1]]
            train_0_pos = train_0_pos[splits[0]:splits[0]+splits[1]]
            train_1_neg = train_1_neg[splits[0]:splits[0]+splits[1]]
            train_1_pos = train_1_pos[splits[0]:splits[0]+splits[1]]
        elif mode == "test":
            train_0_neg = train_0_neg[splits[1]:splits[1]+splits[2]]
            train_0_pos = train_0_pos[splits[1]:splits[1]+splits[2]]
            train_1_neg = train_1_neg[splits[1]:splits[1]+splits[2]]
            train_1_pos = train_1_pos[splits[1]:splits[1]+splits[2]]
        else:
            raise Exception("unknown mode")

        #print(train_0_neg)
        train_0_neg = np.random.choice(
            train_0_neg, int(len(train_0_neg)*ratio), replace=False)
        train_0_pos = np.random.choice(
            train_0_pos, int(len(train_0_pos)*(1-ratio)), replace=False)
        train_1_neg = np.random.choice(
            train_1_neg, int(len(train_1_neg)*(1-ratio)), replace=False)
        train_1_pos = np.random.choice(
            train_1_pos, int(len(train_1_pos)*ratio), replace=False)

        self.select_idx = np.concatenate([train_0_neg, train_0_pos, train_1_neg, train_1_pos])
        self.imageids = all_imageids[self.select_idx]
        self.labels = all_labels[self.select_idx]
        self.site = all_site[self.select_idx]
        self.masks_selector = np.ones(len(self.site))

        # Mask
        rr, cc = skimage.draw.ellipse(112, 112, 100, 90)
        self.seg = np.zeros((224, 224))
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

        return (img, seg), self.labels[idx], self.masks_selector[idx]


class NIHXrayDataset():

    def __init__(self, datadir, csvpath, transform=None, nrows=None):

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
        #self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]

        # Get our two classes.
        idx_sick = self.csv["Finding Labels"].str.contains("Pneumonia")
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


class PCXRayDataset(Dataset):

    def __init__(self, datadir, csvpath, transform=None, dataset='train',
                 pretrained=False, min_patients_per_label=50,
                 exclude_labels=["other", "normal", "no finding"],
                 flat_dir=True):
        """
        Data reader. Only selects labels that at least min_patients_per_label
        patients have.
        """
        super(PCXRayDataset, self).__init__()

        assert dataset in ['train', 'val', 'test']

        self.datadir = datadir
        self.transform = transform
        self.pretrained = pretrained
        self.threshold = min_patients_per_label
        self.exclude_labels = exclude_labels
        self.flat_dir = flat_dir
        self.csv = pd.read_csv(csvpath)
        self.MAXVAL = 65535

        # Keep only the PA view.
        idx_pa = self.csv['ViewPosition_DICOM'].str.contains("POSTEROANTERIOR")
        idx_pa[idx_pa.isnull()] = False
        self.csv = self.csv[idx_pa]

        # Our two classes.
        idx_sick = self.csv['Labels'].str.contains('pneumonia')
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


class ToPILImage(object):
    """
    Convert ndarrays in sample to PIL images.
    """
    def __call__(self, sample):
        to_pil = transforms.ToPILImage()
        sample['PA'] = to_pil(sample['PA'])
        sample['L'] = to_pil(sample['L'])

        return sample


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
    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        rot_amount = np.random.rand() * 5.
        rot = transforms.RandomRotation(rot_amount)
        pa_img = rot(pa_img)
        l_img = rot(l_img)

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample
