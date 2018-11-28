
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import sklearn, sklearn.model_selection
import random
import os, sys
import pickle

import sys
import argparse
from torch.utils import data
import os
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters
import datasets.TNTDataset
import models.simple_cnn

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, nargs='?', default=0, help='random seed for split and init')
    parser.add_argument('-nsamples', type=int, nargs='?', default=64, help='Number of samples for train')
    parser.add_argument('-maxmasks', type=int, nargs='?', default=0, help='Number of masks to use for train')
    parser.add_argument('-thing', default=False, action='store_true', help='Do the thing')

    args = parser.parse_args()

    exp_id = str(args).replace(" ","").replace("Namespace(","").replace(")","").replace(",","-").replace("=","")
    print(exp_id)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cuda = torch.cuda.is_available()

    BATCH_SIZE = 128

    mytransform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(100),
        torchvision.transforms.ToTensor()])

    train = datasets.TNTDataset.TNTDataset("/data/lisa/data/brats2013_tumor-notumor/",
                       transform=mytransform,
                       blur=3,
                       maxmasks=args.maxmasks)

    tosplit = np.asarray([("True" in name) for name in train.imgs])
    idx = range(tosplit.shape[0])
    train_idx, valid_idx = sklearn.model_selection.train_test_split(idx, stratify=tosplit, train_size=0.75,
                                                                    random_state=args.seed)

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=train, batch_size=len(valid_idx),
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_idx),
                                               num_workers=0)

    cnn = models.simple_cnn.CNN()
    if cuda:
        cnn = cnn.cuda()

    print(cnn)

    use_gradmask = args.thing
    stats = []

    for epoch in range(500):
        batch_loss = []
        for step, (x, y) in enumerate(train_loader):

            b_x = Variable(x[0], requires_grad=True)
            b_y = Variable(y)
            seg_x = x[2]

            if cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                seg_x = seg_x.cuda()

            cnn.train()
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            if use_gradmask:
                input_grads = \
                torch.autograd.grad(outputs=torch.abs(output[:, 1]).sum(),  # loss,#torch.abs(output).sum(),
                                    inputs=b_x,
                                    create_graph=True)[0]

                # only apply to positive examples
                input_grads = b_y.float().reshape(-1, 1, 1, 1) * input_grads

                res = input_grads * (1 - seg_x.float())
                gradmask_loss = epoch * (res ** 2)

                # Simulate that we only have some masks
                gradmask_loss = masks_avail.reshape(-1, 1, 1, 1) * gradmask_loss

                gradmask_loss = gradmask_loss.sum()
                loss = loss + gradmask_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            batch_loss.append(loss.data[0])
            # print (loss)

        cnn.eval()
        test_output, last_layer = cnn(valid_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        auc = sklearn.metrics.roc_auc_score(valid_y, pred_y.cpu())
        stat = {"epoch": epoch,
                "trainloss": np.asarray(batch_loss).mean(),
                "validauc": auc}
        stat.update(vars(args))
        stats.append(stat)
        print('Epoch: ', epoch, '| train loss: %.4f' % np.asarray(batch_loss).mean(), '| valid auc: %.2f' % auc)
        # os.mkdir("stats")
        pickle.dump(stats, open("stats/" + exp_id + ".pkl", "wb"))
