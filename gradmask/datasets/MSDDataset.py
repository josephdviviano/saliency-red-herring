from torch.utils.data import Dataset
import os, os.path
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
import h5py, ntpath
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
        
def extract_samples2(data, labels, image_path, label_path):
    image_data, _ = medpy.io.load(image_path)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    these_labels = seg_data.sum((1,2)) > 1
    
    print (collections.Counter(these_labels))
    
    for i in range(image_data.shape[0]):
        data.append([image_data[i],seg_data[i]])
        labels.append(these_labels[i])

        
def compute_hdf5(dataroot, files, hdf5_name):
    
    with h5py.File(hdf5_name,"w") as hf:
        files = sorted(files, key=lambda k: k["image"])
        for i, p in enumerate(files):
            print(p["image"], p["label"])
            name = ntpath.basename(p["image"])

            grp = hf.create_group(name)
            grp.attrs['name'] = name
            grp.attrs['author'] = "jpc"

            samples = []
            labels = []

            extract_samples2(samples, labels, dataroot + p["image"], dataroot + p["label"])

            grp_slices = grp.create_group("slices")
            for idx, zlice in enumerate(samples):
                print(".", end=" ")
                grp_slices.create_dataset(str(idx),data=zlice, compression='gzip')
            print(".")
            grp.create_dataset("labels",data=labels)

#https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606
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
            resize = torchvision.transforms.Resize(size=(new_size, new_size))
            image = resize(image)
            mask = resize(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


cached_msd_ref = {}
    
class MSDDataset(Dataset):

    def __init__(self, mode, dataroot, blur=0, seed=0, nsamples=32, maxmasks=32, transform=None, new_size=100):
        
        self.mode = mode
        self.dataroot = dataroot
        self.new_size = new_size
        
        filename = self.dataroot + "msd_gz_new.hdf5"
        if not os.path.isfile(filename):
            print("Computing hdf5 file of the data")
            dataset = json.load(open(self.dataroot + "dataset.json"))
            files = dataset['training']
            compute_hdf5(self.dataroot, files, filename)
            
        #store cached reference so we can load the valid and test faster
        if not dataroot in cached_msd_ref:
            cached_msd_ref[dataroot] = h5py.File(filename,"r")
        self.dataset = cached_msd_ref[dataroot]
        
        self._all_files = sorted(list(self.dataset.keys()))
        
        all_labels = np.concatenate([self.dataset[i]["labels"] for i in self._all_files])
        print ("Full dataset contains: " + str(collections.Counter(all_labels)))
        
        np.random.seed(seed)
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
        #self.samples = np.concatenate([self.dataset[i]["slices"] for i in self.files])
        self.samples = []
        for file in self.files:
            for sli in range(len(self.dataset[file]["slices"])):
                self.samples.append((file, sli))
        self.labels = np.concatenate([self.dataset[i]["labels"] for i in self.files])
        #self.transform = transform
        self.blur = blur
        
        print ("Loaded images contain:" + str(collections.Counter(self.labels)))
        
        self.idx = np.arange(self.labels.shape[0])
        
        np.random.seed(seed)
        class0 = np.where(self.labels == 0)[0]
        class1 = np.where(self.labels == 1)[0]
        class0 = np.random.choice(class0, nsamples//2, replace=False)
        class1 = np.random.choice(class1, nsamples//2, replace=False)
        self.idx = np.append(class1, class0)
        
        #these should be in order
        #self.samples = self.samples[self.idx]
        self.labels = self.labels[self.idx]
        
        print ("This dataloader contains:" + str(collections.Counter(self.labels)))
        
        # the samples start with labelled ones. past half there should be no labels
        self.mask_idx = self.idx[:maxmasks]
        # transform does nothing, it's pass in parameter only to make it compatible with everything else.

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        
        key = self.samples[self.idx[index]]
        image,seg = self.dataset[key[0]]["slices"][str(key[1])]
        label = self.labels[index]

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
            image, seg = transform(image, seg, True, self.new_size)
        else:
            image, seg = transform(image, seg, False, self.new_size)

        return (image, seg), int(label), float(self.idx[index] in self.mask_idx)

@register.setdatasetname("LungMSDDataset")
class LungMSDDataset(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(dataroot='/network/data1/MSD/MSD/Task06_Lung/', **kwargs)

@register.setdatasetname("ColonMSDDataset")
class ColonMSDDataset(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(dataroot='/network/data1/MSD/MSD/Task10_Colon/', **kwargs)

@register.setdatasetname("LiverMSDDataset")
class LiverMSDDataset(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(dataroot='/network/data1/MSD/MSD/Task03_Liver/', **kwargs)

@register.setdatasetname("ProstateMSDDataset")
class ProstateMSDDataset(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(dataroot='/network/data1/MSD/MSD/Task05_Prostate/', **kwargs)

