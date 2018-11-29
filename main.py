
from __future__ import print_function
import torch
from torch.autograd import Variable
import torchvision
import numpy as np
import sklearn, sklearn.model_selection
import random
import pickle
import argparse
import datasets.TNTDataset
import models.simple_cnn
from torch import nn

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, nargs='?', default=0, help='random seed for split and init')
    parser.add_argument('-nsamples', type=int, nargs='?', default=64, help='Number of samples for train')
    parser.add_argument('-maxmasks', type=int, nargs='?', default=0, help='Number of masks to use for train')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-thing', default=False, action='store_true', help='Do the thing')
    parser.add_argument('-data-path', default=None, help='Path to data.')

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

    train = datasets.TNTDataset.TNTDataset(args.data_path,
                       transform=mytransform,
                       blur=3)

    tosplit = np.asarray([("True" in name) for name in train.imgs])
    idx = range(tosplit.shape[0])
    train_idx, valid_idx = sklearn.model_selection.train_test_split(idx, stratify=tosplit, train_size=0.75,
                                                                    random_state=args.seed)
    train_idx = train_idx[:args.nsamples]
    mask_idx = train_idx[:args.maxmasks]

    train.mask_idx = set(mask_idx)


    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=train, batch_size=len(valid_idx),
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_idx),
                                               num_workers=0)

    valid_data = list(valid_loader)
    valid_x = Variable(valid_data[0][0][0]).cuda()
    valid_y = valid_data[0][1].cuda()

    cnn = models.simple_cnn.CNN()
    if cuda:
        cnn = cnn.cuda()

    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()


    use_gradmask = args.thing
    stats = []

    for epoch in range(500):
        batch_loss = []
        for step, (x, y, use_mask) in enumerate(train_loader):

            b_x = Variable(x[0], requires_grad=True)
            b_y = Variable(y)
            use_mask = Variable(use_mask)
            seg_x = x[2]

            if cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                seg_x = seg_x.cuda()
                use_mask = use_mask.cuda()

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
                #gradmask_loss = res ** 2

                # Simulate that we only have some masks
                #import ipdb; ipdb.set_trace()
                gradmask_loss = use_mask.reshape(-1, 1).float() * gradmask_loss.float().reshape(-1, np.prod(gradmask_loss.shape[1:]))

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
