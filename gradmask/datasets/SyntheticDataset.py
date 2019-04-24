from torch.utils.data import Dataset
import os
import pandas as pd

class SytheticDataset(Dataset):
    def __init__(self, mode, dataroot="/network/data1/GM", blur=0, seed=0, nsamples=32, maxmasks=32, transform=None, new_size=28):

        self.root = root
        self.mode = mode
        
        self._all_files = [f for f in os.listdir(root) if "seg" not in f and ".csv" not in f]
        self._seg_files = [f for f in os.listdir(root) if "seg" in f and ".csv" not in f]
        # random split based on seed
        np.random.seed(seed)
        np.random.shuffle(self._all_files)
        
        # get the files for each mode
        if self.mode == "train":
            self.files = [f for f in self._all_files if "train" in f]
        elif self.mode == "valid":
            self.files = [f for f in self._all_files if "valid" in f]
        elif self.mode == "test":
            self.files = [f for f in self._all_files if "test" in f]
        else:
            raise Exception("Unknown mode")
        
        # get the corresponding labels
        all_labels = pd.read_csv("{}/{}_labels.csv".format(root, mode))
        
        # randomly choose based on nsamples
        np.random.seed(seed)
        class0 = all_labels["file"].loc[all_labels["class"] == 0].values
        class1 = all_labels["file"].loc[all_labels["class"] == 1].values
        class0 = np.random.choice(class0, nsamples//2, replace=False)
        class1 = np.random.choice(class1, nsamples//2, replace=False)
        
        # get the corresponding segmentation files
        class0_seg = [f.replace("img","seg") for f in class0]
        class1_seg = [f.replace("img","seg") for f in class1]
        
        self.idx = np.append(class1, class0)
        self.mask_idx = np.append(class1_seg, class0_seg)
        self.labels = np.append(np.ones(len(class1)), np.zeros(len(class0)))
        
        # TODO: add in selector for maxmasks

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        img = np.load(self.idx[index])
        img_seg = np.load(self.mask_idx[index])
        
        label = self.labels[index]
        # TODO: fix maxmasks so that the 1 returned here is whether the img mask was selected to be used
        return (img, img_seg), label, 1
