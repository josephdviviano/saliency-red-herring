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


def scale(image, maxval=1024):
    """Assumes that maxvalue and minvalue are the same."""
    image += maxval # minimum value is now 0
    image /= maxval*2

    return(image)


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

        # Scale image to (maximum) ~[0 1]
        #image = scale(image)
        return image, mask


class MSDDataset(Dataset):

    def __init__(self, mode, dataroot, blur=0, seed=0, nsamples=32,
                 maxmasks=1, transform=None, new_size=100, mask_all=False):

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

    def open_hdf5(self):
        """In order for hdf5 to work in a multiprocess setting, we need to open
        the HDF5 file once for each process.
        """
        self.dataset = h5py.File(self.filename, "r")
        self._all_files = sorted(list(self.dataset.keys()))
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
            for sli in range(len(self.dataset[fname]["slices"])):
                self.samples.append((fname, sli))
        self.labels = np.concatenate([self.dataset[i]["labels"] for i in self.files])
        self.idx = np.arange(self.labels.shape[0])

        # randomly choose based on nsamples
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

        print ("This dataloader contains: {}".format(
            str(collections.Counter(self.labels))))

        # NB: transform does nothing, only exists for compatibility.

    def __len__(self):

        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        return len(self.idx)

    def __getitem__(self, index):

        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        key = self.samples[self.idx[index]]
        image, seg = self.dataset[key[0]]["slices"][str(key[1])]
        label = self.labels[index]

        image = Image.fromarray(image)

        # Make the segmentation the entire image if it isn't in masks_selector.
        if not self.masks_selector[index]:
            seg = np.ones(seg.shape)

        # Make the segmentation the entire image if it is the negative class.
        if int(label) == 0:
            seg = np.ones(seg.shape)

        # If there is a segmentation, blur it a bit.
        if (self.blur > 0) and (seg.max() != 0):
            seg = skimage.morphology.dilation(seg, selem=square(self.blur))

        seg = (seg > 0) * 1.
        seg = Image.fromarray(seg)

        if self.mode == "train":
            image, seg = transform(image, seg, True, self.new_size)
        else:
            image, seg = transform(image, seg, False, self.new_size)

        # Control condition where we mask all data. Used to see if traditional
        # training works.
        if self.mask_all:
            image *= seg

        return (image, seg, int(label)) #self.masks_selector[index]

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
