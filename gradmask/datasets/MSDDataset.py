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
import torchvision.transforms
import torchvision.transforms.functional as TF
import gradmask.utils.register as register

def extract_samples(data, image_path, label_path):
    image_data, _ = medpy.io.load(image_path)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    labels = seg_data.sum((1,2)) > 1
    
    print (collections.Counter(labels))
    
    for i in range(image_data.shape[0]):
        data.append((image_data[i],seg_data[i],labels[i]))

#https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606
def transform(image, mask, is_train):

        if is_train:
            # Resize
            resize = torchvision.transforms.Resize(size=(110, 110))
            image = resize(image)
            mask = resize(mask)
            
            # Random crop
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                image, output_size=(100, 100))
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
            resize = torchvision.transforms.Resize(size=(100, 100))
            image = resize(image)
            mask = resize(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

class MSDDataset(Dataset):

    def __init__(self, mode, dataroot='/network/data1/MSD/MSD/Task06_Lung/', max_files = 10, blur=0, seed=0, nsamples=32, maxmasks=32, transform=None):
        
        self.mode = mode
        self.dataroot = dataroot
        self.dataset = json.load(open(dataroot + "dataset.json"))
       
        #import ipdb; ipdb.set_trace()
        #files = sorted(self.dataset["training"])
        files = self.dataset['training']

        np.random.seed(seed)
        np.random.shuffle(files)
        
        print("mode=" + self.mode)
        if self.mode == "train":
            self.files = files[:max_files]
        elif self.mode == "valid":
            self.files = files[-max_files*2:-max_files]
        elif self.mode == "test":
            self.files = files[-max_files:]
        else:
            raise Exception("Unknown mode")
        
        self.samples = []
        for i, p in enumerate(self.files):
            print(p["image"], p["label"])

            extract_samples(self.samples, self.dataroot + p["image"], self.dataroot + p["label"])

        self.labels = np.asarray([s[2] for s in self.samples])
        #self.transform = transform
        self.blur = blur
        
        print (collections.Counter(self.labels))
        
        self.idx = np.arange(self.labels.shape[0])
        
        np.random.seed(seed)
        class0 = np.where(self.labels == 0)[0]
        class1 = np.where(self.labels == 1)[0]
        class0 = np.random.choice(class0, nsamples//2, replace=False)
        class1 = np.random.choice(class1, nsamples//2, replace=False)
        self.idx = np.append(class1, class0)
        
        #these should be in order
        self.labels = self.labels[self.idx]
        
        # the samples start with labelled ones. past half there should be no labels
        self.mask_idx = self.idx[:maxmasks]
        # transform does nothing, it's pass in parameter only to make it compatible with everything else.

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        
        image,seg,label = self.samples[self.idx[index]]

        image = Image.fromarray(image)
        #if self.transform != None:
        #    image = self.transform(image)

        if (self.blur > 0) and (seg.max() != 0):
            seg = skimage.filters.gaussian(seg, self.blur)
            seg = seg / seg.max()

        seg = (seg > 0) * 1.

        seg = Image.fromarray(seg)
        #if self.transform != None:
            #seg = self.transform(seg)
            
        if self.mode == "train":
            image, seg = transform(image, seg, True)
        else:
            image, seg = transform(image, seg, False)

        return (image, seg), int(label), float(self.idx[index] in self.mask_idx)

@register.setdatasetname('LungMSDDataset')
class LungMSDDataset(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(dataroot='/network/data1/MSD/MSD/Task06_Lung/', **kwargs)

@register.setdatasetname('ColonMSDDataset')
class ColonMSDDataset(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(dataroot='/network/data1/MSD/MSD/Task10_Colon/', **kwargs)





