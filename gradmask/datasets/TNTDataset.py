from torch.utils.data import Dataset
import os
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters
import utils.register as register
import torch
import numpy as np
import collections
import torchvision.transforms
import torchvision.transforms.functional as TF

def transform(image, mask, is_train, new_size):

        if is_train:
            # Resize
            resize = torchvision.transforms.Resize(size=(new_size+10, new_size+10))
            image = resize(image)
            mask = resize(mask)
            
            # Random crop
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                image, output_size=(new_size, new_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if np.random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        else:
            # Resize
            if new_size:
                resize = torchvision.transforms.Resize(size=(new_size, new_size))
                image = resize(image)
                mask = resize(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


@register.setdatasetname("TNTDataset")
class TNTDataset(Dataset):

    'Tumor-NoTumor Dataset loader for PyTorch'

    ## the folder all is a combination of train and holdout
    def __init__(self, tntpath='/network/data1/brats2013_tumor-notumor/all', mode="train", nsamples=32, maxmasks=32,  blur=0, new_size=100, seed=0, transform=False):
        self.tntpath = tntpath
        self.mode = mode
        self.datapath = self.tntpath
        self._all_files_full = sorted(os.listdir(self.datapath + "/flair"))
        self.seed = seed    
        self.transform = transform
        self.blur = blur
        self.new_size = new_size
        
        self._all_files = np.unique([filename.split("-")[0] for filename in self._all_files_full])
        
        np.random.seed(self.seed)
        np.random.shuffle(self._all_files)
        
        file_ratio = int(len(self._all_files)*0.3)
        print("mode=" + self.mode)
        if self.mode == "train":
            self.files = self._all_files[:file_ratio]
        elif self.mode == "valid":
            self.files = self._all_files[file_ratio:file_ratio*2]
        elif self.mode == "test":
            self.files = self._all_files[file_ratio*2:]
        else:
            raise Exception("Unknown mode")
            
        print("Loading {} files:".format(len(self.files)) + str(self.files))
        self.samples = [filename for filename in self._all_files_full if (filename.split("-")[0] in self.files)]
        self.samples = np.asarray(self.samples)
        
        self.labels = ["True" in filename for filename in self.samples]
        self.labels = np.asarray(self.labels)
            
        self.idx = np.arange(self.labels.shape[0])
        self.labels = self.labels[self.idx]
        
        np.random.seed(seed)
        class0 = np.where(self.labels == 0)[0]
        class1 = np.where(self.labels == 1)[0]
        class0 = np.random.choice(class0, nsamples//2, replace=False)
        class1 = np.random.choice(class1, nsamples//2, replace=False)
        self.idx = np.append(class1, class0)
        
        #these should be in order
        self.samples = self.samples[self.idx]
        self.labels = self.labels[self.idx]
        
        print ("This dataloader contains:" + str(collections.Counter(self.labels)))
        
        # the samples start with labelled ones. past half there should be no labels
        self.mask_idx = self.idx[:maxmasks]
            
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Select sample
        filename = self.samples[index]
        label = self.labels[index]

        image = imread(os.path.join(self.datapath, "flair", filename))
        image = Image.fromarray(image)

        #         t1 = imread(self.datapath + "/t1/" + filename)
        #         t1 = Image.fromarray(t1)
        #         if self.transform != None:
        #             t1 = self.transform(t1)

        seg = imread(self.datapath + "/segmentation/" + filename)
        seg = ((seg >= 30)) * 256.

        if self.blur > 0:
            seg = skimage.filters.gaussian(seg, self.blur)
            seg = seg / seg.max()

        seg = (seg > 0) * 1.

        seg = Image.fromarray(seg)

        #has_tumor = ("True" in filename)
        
        if self.mode == "train":
            image, seg = transform(image, seg, True, self.new_size)
        else:
            image, seg = transform(image, seg, False, self.new_size)

        return (image, seg), int(label), float(index in self.mask_idx)
    
    
    
