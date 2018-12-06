from torch.utils.data import Dataset
import os
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters
import json
import medpy, medpy.io 
import numpy as np
import collections

def extract_samples(data, image_path, label_path):
    image_data, _ = medpy.io.load(image_path)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    labels = seg_data.sum((1,2)) > 1
    
    for i in range(image_data.shape[0]):
        data.append((image_data[i],seg_data[i],labels[i]))

class MSDDataset(Dataset):

    def __init__(self, dataroot, mode, max_files = 10, transform=None, blur=0, seed=0, nsamples=32, maxmasks=32):
        
        self.dataroot = dataroot
        self.dataset = json.load(open(dataroot + "dataset.json"))
        
        files = sorted(self.dataset["training"])
        
        print("mode=" + mode)
        if mode == "train":
            self.files = files[:max_files]
        elif mode == "valid":
            self.files = files[max_files:max_files*2]
        elif mode == "test":
            self.files = files[max_files*2:max_files*3]
        else:
            raise Exception("Unknown mode")
        
        self.samples = []
        for i, p in enumerate(self.files):
            print(p["image"], p["label"])

            extract_samples(self.samples, self.dataroot + p["image"], self.dataroot + p["label"])

        self.labels = np.asarray([s[2] for s in self.samples])
        self.transform = transform
        self.blur = blur
        
        print (collections.Counter(self.labels))
        
        self.idx = np.arange(self.labels.shape[0])
        
        np.random.seed(seed)
        class0 = np.where(self.labels == 0)[0]
        class1 = np.where(self.labels == 1)[0]
        class0 = np.random.choice(class0, nsamples/2, replace=False)
        class1 = np.random.choice(class1, nsamples/2, replace=False)
        self.idx = np.append(class1, class0)
        
        #these should be in order
        self.labels = self.labels[self.idx]
        
        # the samples start with labelled ones. past half there should be no labels
        self.mask_idx = self.idx[:maxmasks]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        
        image,seg,label = self.samples[self.idx[index]]

        image = Image.fromarray(image)
        if self.transform != None:
            image = self.transform(image)

        if (self.blur > 0) and (seg.max() != 0):
            seg = skimage.filters.gaussian(seg, self.blur)
            seg = seg / seg.max()

        seg = (seg > 0) * 1.

        seg = Image.fromarray(seg)
        if self.transform != None:
            seg = self.transform(seg)


        return (image, seg), int(label), float(self.idx[index] in self.mask_idx)