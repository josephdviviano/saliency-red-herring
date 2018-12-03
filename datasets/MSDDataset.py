from torch.utils.data import Dataset
import os
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters
import json
import medpy, medpy.io 
import numpy as np

def extract_samples(data, image_path, label_path):
    image_data, _ = medpy.io.load(image_path)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    labels = seg_data.sum((1,2)) > 1
    
    for i in range(image_data.shape[0]):
        data.append((image_data[i],seg_data[i],labels[i]))

class MSDDataset(Dataset):

    def __init__(self, dataroot, max_files = 10, mask_idx=[], transform=None, blur=0):
        
        self.dataroot = dataroot
        self.dataset = json.load(open(dataroot + "dataset.json"))
        
        self.samples = []
        for i, p in enumerate(sorted(self.dataset["training"])):
            if i >= max_files: break;
            print(p["image"], p["label"])

            extract_samples(self.samples, self.dataroot + p["image"], self.dataroot + p["label"])

        self.labels = np.asarray([s[2] for s in self.samples])
        self.transform = transform
        self.blur = blur
        self.mask_idx = set(mask_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        image,seg,label = self.samples[index]

        image = Image.fromarray(image)
        if self.transform != None:
            image = self.transform(image)

        if self.blur > 0:
            seg = skimage.filters.gaussian(seg, self.blur)
            seg = seg / seg.max()

        seg = (seg > 0) * 1.

        seg = Image.fromarray(seg)
        if self.transform != None:
            seg = self.transform(seg)


        return (image, seg), int(label), float(index in self.mask_idx)