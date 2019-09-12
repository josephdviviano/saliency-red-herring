from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import skimage
import tarfile, glob
import torch
import torch.nn.functional as F
import torchvision.models as models
import utils.register as register

@register.setdatasetname("XRayDataset")
class JointDataset():
    def __init__(self, d1data, d1csv, d2data, d2csv, ratio=0.5, mode="train",
                 seed=0, transform=None, nsamples=None, maxmasks=None,
                 new_size=None):
        self.dataset1 = NIHXrayDataset(d1data, d1csv)
        self.dataset2 = PCXRayDataset(d2data, d2csv)

        splits = np.array([0.5,0.25,0.25])

        np.random.seed(seed)
        all_imageids = np.concatenate([np.arange(len(dataset1)),
                                       np.arange(len(dataset2))]).astype(int)
        all_idx = np.arange(len(all_imageids)).astype(int)
        all_labels = np.concatenate([dataset1.labels,
                                     dataset2.labels]).astype(int)
        all_site = np.concatenate([np.zeros(len(dataset1)),
                                   np.ones(len(dataset2))]).astype(int)

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
            train_0_pos, int(len(train_0_pos)*ratio), replace=False)
        train_1_neg = np.random.choice(
            train_1_neg, int(len(train_1_neg)*(1-ratio)), replace=False)
        train_1_pos = np.random.choice(
            train_1_pos, int(len(train_1_pos)*(1-ratio)), replace=False)

        self.select_idx = np.concatenate([train_0_neg, train_0_pos, train_1_neg, train_1_pos])
        self.imageids = all_imageids[self.select_idx]
        self.labels = all_labels[self.select_idx]
        self.site = all_site[self.select_idx]

    def __len__(self):
        return len(self.imageids)

    def __getitem__(self, idx):

        if self.site[idx] == 0:
            dataset = self.dataset1
        else:
            dataset = self.dataset2

        return dataset[self.imageids[idx]], self.labels[idx], self.site[idx]


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
        return len(self.Data)

    def __getitem__(self, idx):
        im = misc.imread(
            os.path.join(self.datadir, self.csv['Image Index'][idx]))
        # For the ChestXRay dataset, range is [0, 255]

        # Check that images are 2D arrays
        if len(im.shape) > 2:
            im = im[:, :, 0]
        if len(im.shape) < 2:
            print("error, dimension lower than 2 for image {}".format(
                self.Data['Image Index'][idx]))

        # Add color channel
        im = im[:, :, None]

        # Tranform
        if self.transform:
            im = self.transform(im)

        # self.csv['Image Index'][idx]
        return im, self.labels[idx]


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

        self.idx2pt = {idx:x for idx, x in enumerate(self.csv.PatientID.unique())}

    @property
    def targets(self):
        targets = [self.metadata[pt]['Labels'] for pt in self.idx2pt.values()]
        return self.mb.transform(targets)

    @property
    def data(self):
        files = []
        for pt in self.idx2pt.values():
            data = self.metadata[pt]
            pa_dir = str(int(data['ImageDir']['PA'])) if not self.flat_dir else ''
            pa_path = join(self.datadir, pa_dir, data['ImageID']['PA'])
            files.append(pa_path)

        print("Reading files")
        imgs = np.stack([np.array(Image.open(path)) for path in tqdm(files)])
        imgs = np.expand_dims(imgs, -1)
        return imgs

    def __len__(self):
        return len(self.csv.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]

        pa_dir = str(int(data['ImageDir']['PA'])) if not self.flat_dir else ''
        pa_path = join(self.datadir, pa_dir, data['ImageID']['PA'])
        pa_img = np.array(Image.open(pa_path))[..., np.newaxis]

        l_dir = str(int(data['ImageDir']['L'])) if not self.flat_dir else ''
        l_path = join(self.datadir, l_dir, data['ImageID']['L'])
        l_img = np.array(Image.open(l_path))[..., np.newaxis]

        if self.pretrained:
            # Add color channel
            pa_img = np.repeat(pa_img, 3, axis=-1)
            l_img = np.repeat(l_img, 3, axis=-1)

        sample = {'PA': pa_img, 'L': l_img}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample['PA'], self.labels[idx]


class Normalize(object):
    """
    Changes images values to be between -1 and 1.
    """
    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        pa_img = 2 * (pa_img / 65536) - 1.
        pa_img = pa_img.astype(np.float32)
        l_img = 2 * (l_img / 65536) - 1.
        l_img = l_img.astype(np.float32)

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


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
