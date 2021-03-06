from PIL import Image
from skimage.io import imread, imsave
from skimage.morphology import square
from torch.utils.data import Dataset
import activmask.utils.register as register
import collections
import h5py
import json
import medpy, medpy.io
import numpy as np
import os
import skimage, skimage.transform
import skimage.filters
import torchvision.transforms
import torchvision.transforms.functional as TF
import torch

def extract_samples(image_path, label_path):
    """Extracts samples / labels from each .nii"""
    def _load(file_path):
        d, _ = medpy.io.load(file_path)
        return d.transpose(2,0,1)

    image_data = _load(image_path)
    seg_data = _load(label_path)
    these_labels = seg_data.sum((1, 2)) > 1  # True = segmentation in z_slice.


    data, labels = [], []
    for i in range(image_data.shape[0]):
        data.append([image_data[i], seg_data[i]])
        labels.append(these_labels[i])

    return (data, labels)


def compute_hdf5(dataroot, files, hdf5_name):

    with h5py.File(hdf5_name, "w") as hf:
        files = sorted(files, key=lambda k: k["image"])

        for i, p in enumerate(files):

            def _process(d, x):
                return os.path.join(d, x.lstrip('./'))

            image = _process(dataroot, p['image'])
            label = _process(dataroot, p['label'])
            name = os.path.basename(image)

            print('[{}] -- adding image={}, label={}'.format(
                hdf5_name, os.path.basename(image), os.path.basename(label)))

            grp = hf.create_group(name)
            grp.attrs['name'] = name
            grp.attrs['author'] = 'jdv'

            samples, labels = extract_samples(image, label)

            grp_slices = grp.create_group("slices")
            for idx, z_slice in enumerate(samples):
                grp_slices.create_dataset(
                    str(idx), data=z_slice, compression='gzip')
            grp.create_dataset("labels", data=labels)


def scale(image, max_range=1024):
    """Assumes that maxvalue and minvalue are the same."""
    minimum, maximum = torch.min(image), torch.max(image)
    image = (image - minimum) / (maximum - minimum)  # [0, 1]
    image -= 0.5                                     # [-0.5, 0.5]
    image *= 2*max_range                             # [-max_range, max_range]

    return(image)


#https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606
def transform(image, mask, is_train, size):

        if is_train:

            resize = torchvision.transforms.Resize(size=(size+10, size+10))
            image = resize(image)
            mask = resize(mask)

            # Random crop
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                image, output_size=(size, size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if np.random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        else:
            resize = torchvision.transforms.Resize(size=(size, size))
            image = resize(image)
            mask = resize(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = scale(image)  # Scale image to ~[-1024 1024]

        return image, mask


class MSDDataset(Dataset):

    def __init__(self, mode, dataroot, blur=0, nsamples=32, maxmasks=1,
                 transform=None, new_size=100, mask_all=False, seed=1234,
                 verbose=False):

        assert 0 <= maxmasks <= 1

        #self.transform = transform
        self.blur = blur
        self.dataroot = dataroot
        self.filename = os.path.join(self.dataroot, "msd_gz.hdf5")
        self.mask_all = mask_all
        self.maxmasks = maxmasks
        self.mode = mode
        self.new_size = new_size
        self.nsamples = nsamples
        self.samples = []
        self.seed = seed

        # If the hdf5 file does not exist, create it.
        if not os.path.isfile(self.filename):
            print("Computing hdf5 file of the data")
            dataset = json.load(
                open(os.path.join(self.dataroot, "dataset.json")))
            files = dataset['training']
            compute_hdf5(self.dataroot, files, self.filename)

        self.hdf5 = h5py.File(self.filename, "r")
        self._all_files = sorted(list(self.hdf5.keys()))
        file_ratio = int(len(self._all_files)*0.3)

        # Get the subset of the input niftis.
        if self.mode == "train":
            self.files = self._all_files[:file_ratio]
        elif self.mode == "valid":
            self.files = self._all_files[file_ratio:file_ratio*2]
        elif self.mode == "test":
            self.files = self._all_files[file_ratio*2:]
        else:
            raise Exception("Unknown mode")

        np.random.seed(self.seed)
        np.random.shuffle(self._all_files)

        for fname in self.files:
            for sli in range(len(self.hdf5[fname]["slices"])):
                self.samples.append((fname, sli))

        # Labels are in numpy format.
        self.labels = np.concatenate([self.hdf5[i]["labels"] for i in self.files])
        self.idx = np.arange(self.labels.shape[0])

        # Randomly choose based on nsamples
        n_per_class = self.nsamples//2
        np.random.seed(self.seed)
        class0 = np.where(self.labels == 0)[0]
        class1 = np.where(self.labels == 1)[0]
        class0 = np.random.choice(class0, n_per_class, replace=False)
        class1 = np.random.choice(class1, n_per_class, replace=False)
        self.idx = np.append(class1, class0)
        self.labels = self.labels[self.idx]  # These should be in order.

        # Masks_selector is 1 for samples that should have a mask, else zero.
        self.masks_selector = np.ones(len(self.idx))
        if self.maxmasks < 1:
            n_masks_to_rm = round(n_per_class * (1-self.maxmasks))
            idx_masks_class1_to_rm = np.random.choice(
                np.arange(n_per_class), n_masks_to_rm, replace=False)
            idx_masks_class0_to_rm = np.random.choice(
                np.arange(n_per_class, n_per_class*2), n_masks_to_rm,
                replace=False)

            self.masks_selector[idx_masks_class0_to_rm] = 0
            self.masks_selector[idx_masks_class1_to_rm] = 0

        if verbose:
            print ("This {} dataloader contains: {}".format(
                self.mode, str(collections.Counter(self.labels))))

        # NB: transform does nothing, only exists for compatibility.
        # Now build an array of the data for RAM access
        self.dataset = {'images': None, 'segmentations': None, 'labels': None}

        images, segmentations, labels = [], [], []
        for i, idx in enumerate(self.idx):
            key = self.samples[idx]
            image, seg = self.hdf5[key[0]]["slices"][str(key[1])]
            label = self.labels[i]

            # Make the segmentation the entire image if it is the negative class
            # or it is not in the masks_selector.
            if not self.masks_selector[i] or int(label) == 0:
                seg = np.ones(seg.shape)

            # If there is a segmentation, blur it a bit.
            if (self.blur > 0) and (seg.max() != 0):
                #seg = skimage.morphology.dilation(seg, out=seg)
                seg = skimage.filters.gaussian(seg, self.blur, output=seg)
                seg /= seg.max()
                seg = (seg > 0) * 1.

            images.append(image)
            segmentations.append(seg)
            labels.append(int(label))

        self.dataset['images'] = np.stack(images, axis=0)
        self.dataset['segmentations'] = np.stack(segmentations, axis=0)
        self.dataset['labels'] = np.array(labels)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        image = self.dataset['images'][index, ...]
        seg = self.dataset['segmentations'][index, ...]
        label = self.dataset['labels'][index]

        image = Image.fromarray(image)

        seg = Image.fromarray(seg)

        image, seg = transform(image, seg, self.mode == 'train', self.new_size)

        # Control condition where we mask all data. Used to see if traditional
        # training works.
        if self.mask_all:
            image *= seg

        #self.masks_selector[index]
        return (image, seg, int(label))


@register.setdatasetname("LungMSDDataset")
class LungMSDDataset(MSDDataset):
    def __init__(self, base_path, **kwargs):
        super().__init__(dataroot=os.path.join(base_path, 'Task06_Lung'), **kwargs)

@register.setdatasetname("ColonMSDDataset")
class ColonMSDDataset(MSDDataset):
    def __init__(self, base_path, **kwargs):
        super().__init__(dataroot=os.path.join(base_path, 'Task10_Colon'), **kwargs)

@register.setdatasetname("LiverMSDDataset")
class LiverMSDDataset(MSDDataset):
    def __init__(self, base_path, **kwargs):
        super().__init__(dataroot=os.path.join(base_path, 'Task03_Liver'), **kwargs)

@register.setdatasetname("PancreasMSDDataset")
class PancreasMSDDataset(MSDDataset):
    def __init__(self, base_path, **kwargs):
        super().__init__(dataroot=os.path.join(base_path, 'Task07_Pancreas'), **kwargs)

@register.setdatasetname("ProstateMSDDataset")
class ProstateMSDDataset(MSDDataset):
    def __init__(self, base_path, **kwargs):
        super().__init__(dataroot=os.path.join(base_path, 'Task05_Prostate'), **kwargs)

@register.setdatasetname("HeartMSDDataset")
class HeartMSDDataset(MSDDataset):
    def __init__(self, base_path, **kwargs):
        super().__init__(dataroot=os.path.join(base_path, 'Task02_Heart'), **kwargs)
