import torch
from torch.autograd import Variable
import torchvision
from torch import nn
import numpy as np
import sklearn, sklearn.model_selection
import random, argparse, pickle, collections
import datasets
import models.simple_cnn
import sys, os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, nargs='?', default=0, help='random seed for split and init')
    parser.add_argument('-nsamples', type=int, nargs='?', default=128, help='Number of samples for train')
    parser.add_argument('-maxmasks', type=int, nargs='?', default=128, help='Number of masks to use for train')
    parser.add_argument('-maskblur', type=int, nargs='?', default=3, help='std for blur applied to each mask')
    parser.add_argument('-annealinglambda', type=float, default=1.0, help='Annealing')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-batchsize', type=int, default=32)
    parser.add_argument('-thing', default=False, action='store_true', help='Do the thing')
    parser.add_argument('-thingstyle', type=int, default=1, help='Do the thing style')
    parser.add_argument('-dataset', type=str, default="lung", help='name of dataset')
    #parser.add_argument('-data-path', default=None, help='Path to data.')

    args = parser.parse_args()

    exp_id = str(args).replace(" ","").replace("Namespace(","").replace(")","").replace(",","-").replace("=","").replace("'","")
    print(exp_id)

    #don't run twice
    import os.path
    if (os.path.isfile("stats/" + exp_id + ".pkl")):
        print("Already processed: " + exp_id)
        sys.exit()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cuda = torch.cuda.is_available()

    mytransform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(100),
        torchvision.transforms.ToTensor()])

    if args.dataset == "tnt":
        train, valid, test = [datasets.TNTDataset('/data/lisa/data/brats2013_tumor-notumor/',
                           mode=thismode,
                           #transform=mytransform,
                           blur=args.maskblur,
                           nsamples=args.nsamples,
                           seed=args.seed, 
                           maxmasks=args.maxmasks)
                           for thismode in ["train", "valid", "test"]]
    elif args.dataset == "lung":
        train, valid, test = [datasets.MSDDataset('/data/lisa/data/MSD/MSD/Task06_Lung/',
                           mode=thismode,
                           #transform=mytransform,
                           blur=args.maskblur,
                           nsamples=args.nsamples,
                           seed=args.seed, 
                           maxmasks=args.maxmasks,
                           max_files=20)
                           for thismode in ["train", "valid", "test"]]
    elif args.dataset == "colon":
        train, valid, test = [datasets.MSDDataset('/data/lisa/data/MSD/MSD/Task10_Colon/',
                           mode=thismode,
                           #transform=mytransform,
                           blur=args.maskblur,
                           nsamples=args.nsamples,
                           seed=args.seed, 
                           maxmasks=args.maxmasks, 
                           max_files=15)
                           for thismode in ["train", "valid", "test"]]
        

#    tosplit = train.labels#np.asarray([("True" in name) for name in train.imgs])
#    idx = range(tosplit.shape[0])
    
#    #balance
#    idx,tosplit = balanced_subsample(np.asarray(idx), tosplit) 

    
#    train_idx, valid_idx = sklearn.model_selection.train_test_split(idx, 
#                                                                    stratify=tosplit, 
#                                                                    train_size=args.nsamples, 
#                                                                    test_size=100,
#                                                                    random_state=args.seed)
    #train_idx = train_idx[:args.nsamples]
#    mask_idx = train_idx[:args.maxmasks]

#    train.mask_idx = set(mask_idx)

    print("classes train", collections.Counter(train.labels))
    print("classes valid", collections.Counter(valid.labels))
    print("classes test", collections.Counter(test.labels))

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=args.batchsize,num_workers=8,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=len(valid),num_workers=8,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=len(test), num_workers=8,shuffle=True)

    print("Building valid set")
    valid_data = list(valid_loader)
    valid_x = Variable(valid_data[0][0][0]).cuda()
    valid_y = valid_data[0][1].cuda()
    print("done")
    
    print("Building test set")
    test_data = list(test_loader)
    test_x = Variable(test_data[0][0][0]).cuda()
    test_y = test_data[0][1].cuda()
    print("done")

    cnn = models.simple_cnn.CNN(train[0][0][0])
    cnn = cnn.cuda()

    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()


    use_gradmask = args.thing
    stats = []

    for epoch in range(800):
        batch_loss = []
        for step, (x, y, use_mask) in enumerate(train_loader):

            b_x = Variable(x[0], requires_grad=True)
            b_y = Variable(y)
            use_mask = Variable(use_mask)
            seg_x = x[1]

            if cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                seg_x = seg_x.cuda()
                use_mask = use_mask.cuda()

            cnn.train()
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            if use_gradmask:
                if args.thingstyle == 1:
                    input_grads = \
                    torch.autograd.grad(outputs=torch.abs(output[:, 1]).sum(),
                                        inputs=b_x,
                                        create_graph=True)[0]

                elif args.thingstyle == 2:
                    input_grads = \
                    torch.autograd.grad(outputs=(torch.abs(output[:,1]).sum()),
                                        inputs=b_x,
                                        create_graph=True)[0]
                    input_grads2 = \
                    torch.autograd.grad(outputs=(torch.abs(output[:,0]).sum()),
                                        inputs=b_x,
                                        create_graph=True)[0]
                    input_grads = torch.abs(input_grads) - torch.abs(input_grads2)
                
                
                # only apply to positive examples
                input_grads = b_y.float().reshape(-1, 1, 1, 1) * input_grads

                res = input_grads * (1 - seg_x.float())
                gradmask_loss = (res ** 2)

                # Simulate that we only have some masks
                gradmask_loss = use_mask.reshape(-1, 1).float() * gradmask_loss.float().reshape(-1, np.prod(gradmask_loss.shape[1:]))

                gradmask_loss = gradmask_loss.sum()
                gradmask_loss = epoch * args.annealinglambda * gradmask_loss
                loss = loss + gradmask_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            batch_loss.append(loss.data[0])
            # print (loss)

        cnn.eval()

        valid_output, last_layer = cnn(valid_x)
        pred_valid_y = torch.max(valid_output, 1)[1].data.squeeze()
        valid_auc = sklearn.metrics.roc_auc_score(valid_y, pred_valid_y.cpu())
        
        test_output, last_layer = cnn(test_x)
        pred_test_y = torch.max(test_output, 1)[1].data.squeeze()
        test_auc = sklearn.metrics.roc_auc_score(test_y, pred_test_y.cpu())
        
        stat = {"epoch": epoch,
                "trainloss": np.asarray(batch_loss).mean(),
                "validauc": valid_auc,
                "testauc": test_auc}
        stat.update(vars(args))
        stats.append(stat)
        print('Epoch: ', epoch, '| train loss: %.4f' % np.asarray(batch_loss).mean(), '| valid auc: %.2f' % valid_auc, '| test auc: %.2f' % test_auc)
        # os.mkdir("stats")
        if (epoch % 20) == 0: # 20 times faster 
            pickle.dump(stats, open("stats/" + exp_id + ".pkl", "wb"))
            
    pickle.dump(stats, open("stats/" + exp_id + ".pkl", "w"))
    print("script complete")
