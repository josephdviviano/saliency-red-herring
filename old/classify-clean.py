
# coding: utf-8

# In[ ]:




# In[1]:

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
import matplotlib
matplotlib.use('Agg');
import matplotlib.pyplot as plt
#plt.set_cmap('jet');
import random
import os, sys
import pickle



# In[ ]:




# In[296]:

import sys
import argparse

if len(sys.argv) == 3 and sys.argv[1] == "-f": #on jupyter
    sys.argv = ['a','-thing','-maxmasks','0']
    
parser = argparse.ArgumentParser()

parser.add_argument('-seed', type=int, nargs='?',default=0, help='random seed for split and init')
parser.add_argument('-nsamples', type=int, nargs='?',default=64, help='Number of samples for train')
parser.add_argument('-maxmasks', type=int, nargs='?',default=0, help='Number of masks to use for train')
parser.add_argument('-thing', default=False, action='store_true', help='Do the thing')


args = parser.parse_args()


# In[297]:

print(args)


# In[298]:

exp_id = str(args).replace(" ","").replace("Namespace(","").replace(")","").replace(",","-").replace("=","")
print(exp_id)


# In[299]:

torch.manual_seed(args.seed);
random.seed(args.seed)


# In[300]:

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


# In[301]:

cuda = torch.cuda.is_available()


# In[302]:

BATCH_SIZE = 128


# In[303]:

from torch.utils import data
import os
import skimage, skimage.transform
from skimage.io import imread, imsave


# In[304]:

from PIL import Image
import skimage.filters


# In[ ]:




# In[ ]:




# In[ ]:




# In[305]:

class TNTDataset(data.Dataset):
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
        seg = ((seg >= 30))*256.

        if self.blur > 0:
            seg = skimage.filters.gaussian(seg, self.blur)
            seg = seg/seg.max()

        seg = (seg > 0)*1.

        seg = Image.fromarray(seg)
        if self.transform != None:
            seg = self.transform(seg)
        
        has_tumor = ("True" in filename)

        return (flair, flair, seg), has_tumor 


# In[ ]:




# In[ ]:




# In[306]:

mytransform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(100),
    torchvision.transforms.ToTensor()])


# In[307]:

train = TNTDataset("/data/lisa/data/brats2013_tumor-notumor/", 
                   transform=mytransform,
                   blur=3,
                   maxmasks=args.maxmasks)


# In[308]:

plt.imshow(train[5][0][0][0]);
plt.title(train[5][1]);
plt.show()
plt.imshow(train[5][0][2][0]);


# In[ ]:




# In[309]:

plt.imshow(train[250][0][0][0]);
plt.title(train[250][1]);
plt.show()
plt.imshow(train[250][0][2][0]);


# In[ ]:




# In[ ]:




# In[ ]:




# In[310]:

# to debug the maxmasks
# a = list(train)
# b = [k[2] for k in a]
# for i in b:
#     if i[0].sum()> 0:
#         plt.imshow(i[0])
#         plt.show()
    


# In[ ]:




# In[311]:

# def f(x):
#     plt.imshow(train[x][0][1][0]);
#     plt.title(train[x][1]);
#     plt.show()
#     plt.imshow(train[x][0][2][0]);

# interact(f, x=(0,len(train),1));


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[312]:

tosplit = np.asarray([("True" in name) for name in train.imgs])
idx = range(tosplit.shape[0])
train_idx, valid_idx = sklearn.model_selection.train_test_split(idx, stratify=tosplit, train_size=0.75, random_state=args.seed)


# In[313]:

import collections
collections.Counter(tosplit)


# In[314]:

print ("train_idx", len(train_idx))
print ("valid_idx", len(valid_idx))


# In[315]:

#reduce samples
train_idx = train_idx[:args.nsamples]


# In[316]:

print ("train_idx", len(train_idx))
print ("valid_idx", len(valid_idx))


# In[ ]:




# In[ ]:




# In[317]:

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE, 
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
                                          num_workers=8)
valid_loader = torch.utils.data.DataLoader(dataset=train, batch_size=len(valid_idx), 
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_idx),
                                          num_workers=8)


# In[ ]:




# In[ ]:




# In[318]:

valid_data = list(valid_loader)
valid_x = Variable(valid_data[0][0][0]).cuda()
valid_y = valid_data[0][1].cuda()


# In[ ]:




# In[ ]:




# In[319]:

plt.imshow(valid_data[0][0][0][60][0]);
plt.title(valid_y[60]);
plt.show()
plt.imshow(valid_data[0][0][2][60][0]);


# In[ ]:




# In[ ]:




# In[320]:

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=2,   
                padding=0,        
            ),     
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,   
                padding=0,        
            ),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,   
                padding=0,        
            ),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=2,   
                padding=0,        
            ),
            nn.ReLU(),
        )
        self.out = nn.Linear(440, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


# In[321]:

cnn = CNN()
if cuda:
    cnn = cnn.cuda()

print(cnn)


# In[322]:

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


# In[ ]:




# In[323]:

masks_avail = np.arange(args.nsamples) < args.maxmasks
masks_avail = torch.FloatTensor(masks_avail*1.).cuda()
masks_avail


# In[ ]:




# In[334]:

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
            input_grads = torch.autograd.grad(outputs=torch.abs(output[:,1]).sum(), #loss,#torch.abs(output).sum(), 
                                       inputs=b_x, 
                                       create_graph=True)[0]
            
            #only apply to positive examples
            input_grads = b_y.float().reshape(-1,1,1,1)*input_grads
            
            res = input_grads * (1-seg_x.float())
            gradmask_loss = epoch*(res**2)
            
            #Simulate that we only have some masks
            gradmask_loss = masks_avail.reshape(-1,1,1,1) * gradmask_loss
            
            gradmask_loss = gradmask_loss.sum()
            loss = loss + gradmask_loss
            
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        batch_loss.append(loss.data[0])
        #print (loss)
    
    cnn.eval()
    test_output, last_layer = cnn(valid_x)
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    auc = sklearn.metrics.roc_auc_score(valid_y, pred_y.cpu())
    stat = {"epoch":epoch,
            "trainloss":np.asarray(batch_loss).mean(),
            "validauc": auc}
    stat.update(vars(args))
    stats.append(stat)
    print('Epoch: ', epoch, '| train loss: %.4f' % np.asarray(batch_loss).mean(), '| valid auc: %.2f' % auc)
    #os.mkdir("stats")
    pickle.dump(stats, open("stats/" + exp_id + ".pkl","wb"))     


# In[1]:

gradmask_loss.shape


# In[ ]:

#loss*10000*(res**2).sum()


# In[ ]:




# In[1]:

import pickle
#os.mkdir("stats")
pickle.dump(stats, open("stats/" + exp_id + ".pkl","w"))   


# In[46]:

sys.exit()


# In[ ]:




# In[ ]:




# In[ ]:

##############################################
##############################################
##############################################
##############################################
## below is code to debug the model


# In[ ]:




# In[ ]:




# In[221]:




# In[35]:

x, y = train[250]


# In[36]:

x[0].shape


# In[37]:

plt.imshow(x[0][0], cmap="gray");
plt.title(y);
plt.show()
plt.imshow(x[2][0]);


# In[38]:

x_var = Variable(x[0].unsqueeze(0).cuda(), requires_grad=True)
pred = cnn(x_var)[0]


# In[39]:

pred


# In[40]:

input_grads = torch.autograd.grad(outputs=torch.abs(pred[:,1]).sum(), 
                                       inputs=x_var,
                                       create_graph=True)[0]


# In[ ]:




# In[41]:

input_grads = input_grads[0][0].cpu().detach().numpy()


# In[ ]:




# In[42]:

plt.imshow(np.abs(input_grads), cmap="jet");


# In[ ]:




# In[43]:

plt.imshow(x[2][0])


# In[ ]:




# In[ ]:




# In[44]:

#these are the only allowed grads
masked_grads = np.abs(input_grads)*(x[2][0])
masked_grads[0][0] = torch.FloatTensor(np.abs(input_grads)).max()
plt.imshow(masked_grads, cmap="jet")


# In[45]:

#We can regularize to reduce this
masked_grads = np.abs(input_grads)*(1-x[2][0])
masked_grads[0][0] = torch.FloatTensor(np.abs(input_grads)).max()
plt.imshow(masked_grads, cmap="jet")


# In[ ]:




# In[ ]:




# In[ ]:

import skimage.filters
enlarged_mask = skimage.filters.gaussian(x[2][0].numpy(),15)
enlarged_mask = enlarged_mask/enlarged_mask.max()
enlarged_mask.max()


# In[ ]:




# In[ ]:

plt.imshow(enlarged_mask)


# In[59]:

#assuming the segmentations are sloppy we can enlarge the segmentations
masked_grads = input_grads*(1-enlarged_mask)
plt.imshow(masked_grads)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

#torch.save(cnn.state_dict(), "./classifier_model.pth")


# In[18]:

# checkpoint = torch.load("./classifier_model.pth")
# cnn.load_state_dict(checkpoint)


# In[ ]:

# cnn.eval()
# test_output, last_layer = cnn(valid_x)
# pred_y = torch.max(test_output, 1)[1].data.squeeze()
# accuracy = float((pred_y == valid_y).sum()) / float(valid_y.size(0))
# print 'valid accuracy: %.2f' % accuracy


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



