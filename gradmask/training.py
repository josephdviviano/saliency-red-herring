from collections import OrderedDict
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import copy
import itertools
import logging
import manager.mlflow.logger as mlflow_logger
import notebooks.auto_ipynb as auto_ipynb
import numpy as np
import numpy as np
import pprint
import random
import time, os, sys
import torch
import torch.nn as nn
import utils.configuration as configuration
import utils.monitoring as monitoring

# Fix backend so I can print images on the cluster.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_LOG = logging.getLogger(__name__)

@mlflow_logger.log_experiment(nested=True)
@mlflow_logger.log_metric('best_metric', 'testauc_for_best_validauc')
def train(cfg, dataset_train=None, dataset_valid=None, dataset_test=None, recompile=True):

    print("Our config:")
    pprint.pprint(cfg)

    # Get information from configuration.
    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epochs = cfg['num_epochs']
    # maxmasks = cfg['maxmasks']
    penalise_grad = cfg['penalise_grad']
    penalise_grad_usemasks = cfg.get('penalise_grad_usemasks')
    conditional_reg = cfg.get('conditional_reg', False)

    device = 'cuda' if cuda else 'cpu'

    # Setting the seed.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cuda:
        torch.cuda.manual_seed_all(seed)

    # Dataset
    # transform
    tr_train = configuration.setup_transform(cfg, 'train')
    tr_valid = configuration.setup_transform(cfg, 'valid')
    tr_test= configuration.setup_transform(cfg, 'test')

    # The dataset
    if recompile:
        dataset_train = configuration.setup_dataset(cfg, 'train')(tr_train)
        dataset_valid = configuration.setup_dataset(cfg, 'valid')(tr_valid)
        dataset_test = configuration.setup_dataset(cfg, 'test')(tr_test)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'],
                                                num_workers=0, pin_memory=cuda)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'],
                                                num_workers=0, pin_memory=cuda)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'],
                                                num_workers=0, pin_memory=cuda)

    model = configuration.setup_model(cfg).to(device)
    print(model)
    # TODO: checkpointing

    # Optimizer
    optim = configuration.setup_optimizer(cfg)(model.parameters())
    print(optim)

    criterion = torch.nn.CrossEntropyLoss()

    # Aaaaaannnnnd, here we go!
    best_metric = 0.
    testauc_for_best_validauc = 0.
    metrics = []

    # Wrap the function for mlflow. Don't worry buddy, if you don't have mlflow it will still work.
    valid_wrap_epoch = mlflow_logger.log_metric('valid_acc')(test_epoch)
    test_wrap_epoch = mlflow_logger.log_metric('test_acc')(test_epoch)

    img_viz0 = dataset_train[2]
    img_viz1 = dataset_valid[2]

    for epoch in range(num_epochs):

        if cfg['viz']:
            print("CUDA: ", cuda)
            processImageSmall("0",epoch, img_viz0, model, cuda)
            processImageSmall("1",epoch, img_viz1, model, cuda)

        penalise_grad_epoch = penalise_grad

        avg_loss = train_epoch(epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion,
                               penalise_grad=penalise_grad_epoch,
                               penalise_grad_usemasks=penalise_grad_usemasks,
                               conditional_reg=conditional_reg,
                               gradmask_lambda=cfg['gradmask_lambda'],
                               bre_lambda=cfg['bre_lambda'],
                               recon_lambda=cfg['recon_lambda'])

        auc_train = valid_wrap_epoch(name="train",
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=train_loader,
                                     criterion=criterion)

        auc_valid = valid_wrap_epoch(name="valid",
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion)

        if auc_valid > best_metric:
            best_metric = auc_valid

            # only compute when we need to
            auc_test = test_wrap_epoch(name="test",
                                   epoch=epoch,
                                   model=model,
                                   device=device,
                                   data_loader=test_loader,
                                   criterion=criterion)
            testauc_for_best_validauc = auc_test

        stat = {"epoch": epoch,
                "trainloss": avg_loss,
                "validauc": auc_valid,
                "trainauc": auc_train,
                "testauc": auc_test,
                "testauc_for_best_validauc": testauc_for_best_validauc}
        stat.update(configuration.process_config(cfg))

        metrics.append(stat)

    monitoring.log_experiment_csv(cfg, [best_metric])

    return metrics, best_metric, testauc_for_best_validauc, {'dataset_train': dataset_train,
                                                    'dataset_valid': dataset_valid,
                                                    'dataset_test': dataset_test}

@mlflow_logger.log_metric('train_loss')
def train_epoch(epoch, model, device, train_loader, optimizer,
                criterion, penalise_grad, penalise_grad_usemasks,
                conditional_reg, gradmask_lambda, bre_lambda, recon_lambda):

    model.train()

    recon_criterion = nn.MSELoss()

    # losses: cross-entropy + reconstruction + gradmask + bre
    avg_clf_loss = []
    avg_gradmask_loss = []
    avg_recon_loss = []
    avg_bre_loss = []
    avg_loss = []

    t = tqdm(train_loader)
    for batch_idx, (data, target, use_mask) in enumerate(t):

        #use_mask = torch.ones((len(target))) # TODO change here.
        optimizer.zero_grad()

        x, seg = data
        x, seg, target, use_mask = x.to(device), seg.to(device), target.to(device), use_mask.to(device)
        x.requires_grad = True

        class_output, reconstruction = model(x)
        clf_loss = criterion(class_output, target)

        # Reconstruction Loss.
        if recon_lambda > 0:
            recon_loss = recon_criterion(x, reconstruction)
            recon_loss *= recon_lambda
        else:
            recon_loss = torch.zeros(1).to(device)

        # Gradmask Loss.
        # TODO: slow! Optimized using advance indexing or something?
        gradmask_loss = torch.Tensor([0]).to(device)  # default

        if penalise_grad != "False":

            input_grads = get_gradmask_loss(
                x, class_output, model, target, penalise_grad)

            # only apply to positive examples
            if penalise_grad != "diff_from_ref":
                # only do it here because the masking happens elsewhere in
                # the diff_from_ref architecture
                # print("target mask: ", target.float().reshape(-1, 1, 1, 1).shape)
                input_grads = target.float().reshape(-1, 1, 1, 1) * input_grads

            if conditional_reg:
                temp_softmax = torch.softmax(class_output, dim=1).detach()
                certainty_mask = 1 - torch.argmax(
                    (temp_softmax > 0.95).float(), dim=1)
                input_grads *= certainty_mask.float().reshape(-1, 1, 1, 1)

            if penalise_grad_usemasks:
                input_grads = input_grads * (1 - seg.float())
            else:
                input_grads = input_grads

            gradmask_loss = input_grads

            # Gradmask loss increases over epochs.
            #gradmask_loss = epoch * (gradmask_loss ** 2)
            #n_iter = len(train_loader) * epoch + batch_idx

            # Simulate that we only have some masks
            gradmask_loss = use_mask.reshape(-1, 1).float() * \
                            gradmask_loss.float().reshape(
                                -1, np.prod(gradmask_loss.shape[1:]))
            gradmask_loss = gradmask_loss.abs().sum()
            gradmask_loss *= gradmask_lambda

        # BRE loss.
        bre_loss, me_loss, ac_loss, me_stats, ac_stats = get_bre_loss(model)
        bre_loss *= bre_lambda

        # Final loss is three terms combined.
        loss = clf_loss + gradmask_loss + recon_loss + bre_loss

        # Reporting.
        avg_clf_loss.append(clf_loss.detach().cpu().numpy())
        avg_recon_loss.append(recon_loss.detach().cpu().numpy())
        avg_gradmask_loss.append(gradmask_loss.detach().cpu().numpy())
        avg_bre_loss.append(bre_loss.detach().cpu().numpy())
        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description((penalise_grad +
            ' Train (clf={:4.4f} recon={:4.4f} mask={:4.4f}  bre={:4.4f} total={:4.4f})'.format(
                np.mean(avg_clf_loss),
                np.mean(avg_recon_loss),
                np.mean(avg_gradmask_loss),
                np.mean(avg_bre_loss),
                np.mean(avg_loss))))

        # Learning.
        if np.isnan(loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
        loss.backward()
        optimizer.step()

    return np.mean(avg_loss)


def get_gradmask_loss(x, class_output, model, target, penalise_grad):
    if penalise_grad == "contrast":
        # d(y_0-y_1)/dx
        input_grads = torch.autograd.grad(outputs=torch.abs(class_output[:, 0]-class_output[:, 1]).sum(),
                                inputs=x, allow_unused=True,
                                create_graph=True)[0]
    elif penalise_grad == "nonhealthy":
        # select the non healthy class d(y_1)/dx
        input_grads = torch.autograd.grad(outputs=torch.abs(class_output[:, 1]).sum(),
                                inputs=x, allow_unused=True,
                                create_graph=True)[0]
    elif penalise_grad == "diff_from_ref":
        # do the deep lift style ref update and diff-to-ref calculations here

        # update the reference stuff
        # 1) mask all_activations to get healthy only

        try:
            target = target.to(x.get_device())
        except:
            pass

        healthy_mask = target.float().reshape(-1, 1, 1, 1).clone()
        healthy_mask[target.float().reshape(-1, 1, 1, 1) == 0] = 1
        healthy_mask[target.float().reshape(-1, 1, 1, 1) != 0] = 0

        diff = torch.FloatTensor()

        try:
            diff = diff.to(x.get_device())
        except:
            pass

        # print("Activation lengths: ", len(model.all_activations))
        for i in range(len(model.all_activations)):
            a = model.all_activations[i]
            # print("Activation shape: ", a.shape)

            if len(a.shape) < 4:
                # activations are from the last layers (shape [batch, FC layer_size])
                new_mask = target.float().reshape(-1, 1)
                new_mask[target.float().reshape(-1, 1) == 0] = 1
                new_mask[target.float().reshape(-1, 1) != 0] = 0
                healthy_batch = new_mask * a
            else:
                healthy_batch = healthy_mask * a
#                     print("healthy mask shape: ", healthy_mask.shape, "activation shape: ", a.shape, "len: ", len(a.shape))

            # 2) detach grads for the healthy samples
            healthy_batch = healthy_batch.detach()
#                     print("Healthy batch shape: ", healthy_batch.shape)

            # 3) get batch-wise average of activations per layer
            batch_avg_healthy = torch.mean(healthy_batch, dim=0)
#                     print("Healthy batch avg shape: ", batch_avg_healthy.shape)

            # 4) update global reference layer average in model's deep_lift_ref attr
            if len(model.ref) < len(model.all_activations):
                # for the first iteration, just make the model.ref == batch_avg_healthy for that layer
                model.ref.append(batch_avg_healthy)
            else:
                # otherwise, a rolling average
#                         print("ref shape: ", model.ref[i].shape)
                model.ref[i] = model.ref[i] * 0.8 + batch_avg_healthy

        # 5) TODO: somehow incorporate std to allowing regions of variance in the healthy images

        # use the reference layers to get the diff-to-ref of each layer and output contribution scores
        # contribution scores should be the input_grads? Should be a single matrix of values for how each input
        # pixel contributes to the output layer, no? Like all the layer-wise diff-from-ref get condensed into
        # one thing based on sum(contribution_scores of (delta_x_i, delta_t)) = delta_t
        # 1) for each layer, t - t0, then mask the unhealthy ones
        # 2) flatten, 3) stack or join together somehow?, 4) L1 norm, 5) input grads
            diff = torch.cat((diff, torch.flatten((a - model.ref[i]) * target.float().reshape(-1, 1, 1, 1))))

        input_grads = torch.autograd.grad(outputs=torch.abs(diff).sum(),
                                         inputs=x, allow_unused=True, create_graph=True)[0]

    elif penalise_grad == "masd_style":
        # In the style of the Model-Agnostic Saliency Detector paper: https://arxiv.org/pdf/1807.07784.pdf

        print(class_output.shape, representation.shape)
        # outputs should now be: abs(diff(area_seg - area_saliency_map)) +
        # WIP
    else:
        raise Exception("Unknown style of penalise_grad. Options are: contrast, nonhealthy, diff_from_ref")

    return input_grads


def dim2margin(d, s=3.):
    """
    s is a hyperparameter which tunes how smooth the transition is from
    +1 to -1.
    """
    _base_var = 1.  # .25 is for 0-1 bernoulli, this is +/- bernoulli
    return(s*np.sqrt(_base_var / d))


def softsign(h, epsilon=1e-3):
    """
    A somewhat-smooth representation of a binarized vector, which allows
    for gradients to flow.
    """
    _mu = abs(h).mean()
    h_epsilon = np.float32(epsilon) * _mu
    h_epsilon = h_epsilon.detach()
    act = (h / (abs(h) + h_epsilon))
    return(act)


def get_bre_loss(model, epsilon=1e-3, binarizer="softsign", s=3.0):
    """
    Improving GAN training via binarized representation entropy regularization.
    Cao et al. ICLR 2018.

    """

    me_value, me_stats = me_term(
        model.all_activations, epsilon=epsilon, binarizer=binarizer)
    ac_value, ac_stats = ac_term(
        model.all_activations, epsilon=epsilon, binarizer=binarizer, s=s)
    bre_loss = me_value + ac_value
    return(bre_loss, me_value, ac_value, me_stats, ac_stats)


def me_term(hs, epsilon=1e-3, binarizer='softsign'):
    """
    Gets the marginal of the joint entropy of the binarized activations
    for each layer, averaging across layers.
    """
    h_me = []
    stats = []
    binarizer = globals()[binarizer]

    for h in hs:

        # Flatten each member of the minibatch.
        h = torch.flatten(h, 1)

        # Debug mean of absoloute activations.
        stats.append(abs(h).mean())

        act = binarizer(h, epsilon)

        # Mean of the i-th activations across the minibatch k, squared, then
        # mean of those i activations.
        h_me.append(torch.mean(act, 0).pow(2).mean())

    # Mean of the me term across all layers.
    hs_me = sum(h_me) /  np.float32(len(hs))
    stats = sum(stats) / np.float32(len(stats))

    return(hs_me, stats)


def ac_term(hs, epsilon=1e-3, s=3., binarizer='softsign'):
    """
    Get the mean correlation of the activation vectors.
    """
    assert s >= 1.

    C = 0.0
    stats_C = 0.0
    stats_abs_mean = []  # Keeps track of the mean(abs()) value.
    stats_sat90_ratio = []  # Keeps track of the % of values over abs(0.90).
    stats_sat99_ratio = []  # Keeps track of the % of values over abs(0.99).
    binarizer = globals()[binarizer]

    for h in hs:
        act = binarizer(h, epsilon)

        stats_abs_mean.append(torch.mean(abs(act)))
        stats_sat90_ratio.append(torch.mean((abs(act) > .90).double()))
        stats_sat99_ratio.append(torch.mean((abs(act) > .99).double()))

        act = torch.flatten(act, 1)

        # Dot product of activations with transpose.
        l_C = torch.mm(act, act.t()) / act.shape[1]
        stats_C += l_C

        # Store the max of (0, l_C) in C.
        C_tmp = abs(l_C) - dim2margin(act.shape[1], s)
        C_tmp[C_tmp < 0] = 0
        C += C_tmp

    # Calculate final mean correlation across all layers.
    C /= np.float32(len(hs))
    C -= torch.diag(torch.diag(C))
    C = torch.mean(C)

    # Calculate statistics for debugging and reporting.
    stats_abs_mean = sum(stats_abs_mean) / np.float32(len(hs))
    stats_sat90_ratio = sum(stats_sat90_ratio) / np.float32(len(hs))
    stats_sat99_ratio = sum(stats_sat99_ratio) / np.float32(len(hs))
    stats_C /= np.float32(len(hs))
    stats_Cmin = stats_C.min(1)[0]  # collect before we 0 the diagonal
    stats_C -= torch.diag(torch.diag(stats_C))
    stats_Cmean = stats_C.sum(0) / stats_C.shape[1]-1
    stats_Cmax = stats_C.max(0)

    stats = {"cmin": stats_Cmin,
             "cmean": stats_Cmean,
             "cmax": stats_Cmax,
             "abs_mean": stats_abs_mean,
             "sat90_ratio": stats_sat90_ratio,
             "sat99_ratio": stats_sat99_ratio}

    return(C, stats)


def test_epoch(name, epoch, model, device, data_loader, criterion):

    model.eval()
    data_loss = 0

    targets = []
    predictions = []
    predictions_bin = []  # Keep track of actual predictions.

    with torch.no_grad():
        for (data, target, use_mask) in data_loader:

            x, seg, target = data[0].to(device), data[1].to(device), target.to(device)
            class_output, _ = model(x)

            # sum up batch loss
            data_loss += criterion(class_output, target).sum().item()

            targets.append(target.cpu().data.numpy())
            predictions.append(class_output.cpu().data.numpy())
            predictions_bin.append(
                torch.max(class_output, 1)[1].detach().cpu().numpy())

    auc = accuracy_score(np.concatenate(targets), np.concatenate(predictions).argmax(axis=1))
    predictions_bin = np.hstack(predictions_bin)

    data_loss /= len(data_loader)
    print(epoch, 'Average {} loss: {:.4f}, AUC: {:.4f}, pred: {}/{}'.format(
        name, data_loss, auc,
        np.sum(predictions_bin==0), np.sum(predictions_bin==1)))

    return auc

def processImage(text, i, sample, model, cuda=True):
    fig = plt.Figure(figsize=(20, 10), dpi=160)
    gcf = plt.gcf()
    gcf.set_size_inches(20, 10)
    fig.set_canvas(gcf.canvas)
    gridsize = (3,6)
    x, target, use_mask = sample

    if cuda:
        x_var = torch.autograd.Variable(x[0].unsqueeze(0).cuda(), requires_grad=True)
    else:
        x_var = torch.autograd.Variable(x[0].unsqueeze(0), requires_grad=True)
    model.eval()
    class_output, res = model(x_var)

    ax2 = plt.subplot2grid(gridsize, (1, 0), colspan=1)
    ax3 = plt.subplot2grid(gridsize, (1, 1), colspan=1)
    ax4 = plt.subplot2grid(gridsize, (0, 0), colspan=1)
    ax5 = plt.subplot2grid(gridsize, (0, 1), rowspan=1)
    ax6 = plt.subplot2grid(gridsize, (0, 2), rowspan=1)
    ax7 = plt.subplot2grid(gridsize, (0, 3), rowspan=1)

    ax2.set_title(str(i) + "Input Image")
    ax2.imshow(x[0][0].cpu().numpy(), interpolation='none', cmap='Greys_r')
    ax3.set_title("Masked input")
    ax3.imshow((x[1][0]*x[0][0]).cpu().numpy(), interpolation='none', cmap='Greys_r')

    ax4.set_title("nonhealthy")
    gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "nonhealthy").detach().cpu().numpy()[0][0]
    ax4.imshow(np.abs(gradmask), cmap="jet", interpolation='none')
    ax5.set_title("nonhealthy masked")
    ax5.imshow(np.abs(gradmask)*x[1][0].cpu().numpy(), cmap="jet", interpolation='none')

    ax6.set_title("contrast")
    gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "contrast").detach().cpu().numpy()[0][0]
    ax6.imshow(np.abs(gradmask), cmap="jet", interpolation='none')

    ax7.set_title("diff_from_ref")
    gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "diff_from_ref").detach().cpu().numpy()[0][0]
    ax7.imshow(np.abs(gradmask), cmap="jet", interpolation='none')

    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig('images/image-' + text + "-" + str(i) + '.png', bbox_inches='tight', pad_inches=0)

#to make video: ffmpeg -y -i images/image-test-%d.png -vcodec libx264 aout.mp4
def processImageSmall(text, i, sample, model, cuda=True):
    fig = plt.Figure(figsize=(20, 10), dpi=160)
    gcf = plt.gcf()
    gcf.set_size_inches(20, 10)
    fig.set_canvas(gcf.canvas)
    gridsize = (2,3)
    x, target, use_mask = sample

    if cuda:
        x_var = torch.autograd.Variable(x[0].unsqueeze(0).cuda(), requires_grad=True)
    else:
        x_var = torch.autograd.Variable(x[0].unsqueeze(0), requires_grad=True)

    model.eval()
    class_output, res = model(x_var)

    ax2 = plt.subplot2grid(gridsize, (1, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 1))
#     ax4 = plt.subplot2grid(gridsize, (0, 0))
#     ax5 = plt.subplot2grid(gridsize, (0, 1))
    ax6 = plt.subplot2grid(gridsize, (0, 0))
    ax7 = plt.subplot2grid(gridsize, (0, 1))

    ax2.set_title(str(i) + " Input Image")
    ax2.imshow(x[0][0].cpu().numpy(), interpolation='none', cmap='Greys_r')
    ax3.set_title("Mask")
    ax3.imshow((x[1][0]).cpu().numpy(), interpolation='none', cmap='Greys_r')

#     ax4.set_title("nonhealthy")
#     gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "nonhealthy").detach().cpu().numpy()[0][0]
#     ax4.imshow(np.abs(gradmask), cmap="jet", interpolation='none')
#     ax5.set_title("nonhealthy masked")
#     ax5.imshow(np.abs(gradmask)*x[1][0].cpu().numpy(), cmap="jet", interpolation='none')

    ax6.set_title("nonhealthy d|y|/dx")
    gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "nonhealthy").detach().cpu().numpy()[0][0]
    ax6.imshow(np.abs(gradmask), cmap="jet", interpolation='none')

    ax7.set_title("contrast d|y0-y1|/dx")
    gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "contrast").detach().cpu().numpy()[0][0]
    ax7.imshow(np.abs(gradmask), cmap="jet", interpolation='none')

#     try:
#         ax7.set_title("diff_from_ref")
#         gradmask = get_gradmask_loss(x_var, class_output, model, torch.tensor(1.), "diff_from_ref").detach().cpu().numpy()[0][0]
#         ax7.imshow(np.abs(gradmask), cmap="jet", interpolation='none')
#     except:
#         pass

    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig('images/image-' + text + "-" + str(i) + '.png', bbox_inches='tight', pad_inches=0)


@mlflow_logger.log_experiment(nested=False)
def train_skopt(cfg, n_iter, base_estimator, n_initial_points, random_state, new_size, train_function=train):

    """
    Do a Bayesian hyperparameter optimization.

    :param cfg: Configuration file.
    :param n_iter: Number of Bayesien optimization steps.
    :param base_estimator: skopt Optimization procedure.
    :param n_initial_points: Number of random search before starting the optimization.
    :param random_state: seed.
    :param train_function: The training procedure to optimize. The function should take a dict as input and return a metric maximize.
    :return:
    """

    import skopt
    from skopt.space import Real, Integer, Categorical

    # Helper function to help us sparse the yaml config file.
    def parse_dict(d_, prefix='', l=[]):
        """
        Find the keys in the config dict that are to be optimized.
        """
        if isinstance(d_, dict):
            for key in d_.keys():
                temp = parse_dict(d_[key], prefix + '.' + key, [])
                if temp:
                    l += temp
            return l
        else:
            try:
                x = eval(d_)
                if isinstance(x, (Real, Integer, Categorical)):
                    l.append((prefix, x))
            except:
                pass
            return l

    # Helper functions to hack in the config and change the right parameter.
    def set_key(dic, key, value):
        """
        Aux function to set the value of a key in a dict
        """
        k1 = key.split(".")
        k1 = list(filter(lambda l: len(l) > 0, k1))
        if len(k1) == 1:
            dic[k1[0]] = value
        else:
            set_key(dic[k1[0]], ".".join(k1[1:]), value)

    def generate_config(config, keys, new_values):
        new_config = copy.deepcopy(config)
        for i, key in enumerate(list(keys.keys())):
            set_key(new_config, key, new_values[i].item())
        return new_config

    # Sparse the parameters that we want to optimize
    skopt_args = OrderedDict(parse_dict(cfg))

    # Create the optimizer
    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator=base_estimator,
                                n_initial_points=n_initial_points,
                                random_state=random_state)

    opt_results = None
    state = {}
    best_metric = 0
    best_metrics = None
    for i in range(n_iter):

        # Do a bunch of loops.
        suggestion = optimizer.ask()
        this_cfg = generate_config(cfg, skopt_args, suggestion)
        opt_dict = this_cfg['optimizer']
        #opt_dict[list(opt_dict.keys())[0]]['lr'] = 10**float(-opt_dict[list(opt_dict.keys())[0]]['lr'])

        # optimizer.tell(suggestion, - train_function(this_cfg)[0]) # We minimize the negative accuracy/AUC
        metrics, metric, metric_test, state = train_function(this_cfg, recompile=state == {}, **state)

        print(i, "metric",metric,"metric_test",metric_test,"this_cfg",this_cfg)
        opt_results = optimizer.tell(suggestion, - metric) # We minimize the negative accuracy/AUC

        #record metrics to write and plot
        if best_metric < metric:
            best_metric = metric
            best_metrics = metrics
            print(i, "New best metric: ", best_metric, "Best metric test: ", metric_test)

    print("The best metric: ", best_metric, "Best bmetric test: ", metric_test)
    # Done! Hyperparameters tuning has never been this easy.

    # Saving the skopt results.
    try:
        import mlflow
        import mlflow.sklearn
        # mlflow.sklearn.log_model(opt_results, 'skopt')
        dimensions = list(skopt_args.keys())
        auto_ipynb.to_ipynb(auto_ipynb.plot_optimizer, True, run_uuid=mlflow.active_run()._info.run_uuid, dimensions=dimensions, path='skopt')
    except:
        # sorry buddy, the feature you are seeking is not available.
        pass

    return best_metrics
