from torch.utils.data import Dataset
import os
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters

class TNTDataset(Dataset):

    'Tumor-NoTumor Dataset loader for PyTorch'

    def __init__(self, tntpath, subset="train", maxmasks=999999, transform=None, blur=0):
        self.tntpath = tntpath
        self.subset = subset
        self.datapath = self.tntpath + "/" + self.subset + "/"
        self.imgs = sorted(os.listdir(self.datapath + "/flair"))
        self.transform = transform
        self.blur = blur

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Select sample
        filename = self.imgs[index]

        flair = imread(self.datapath + "/flair/" + filename)
        flair = Image.fromarray(flair)
        if self.transform != None:
            flair = self.transform(flair)

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
        if self.transform != None:
            seg = self.transform(seg)

        has_tumor = ("True" in filename)

        return (flair, flair, seg), has_tumor