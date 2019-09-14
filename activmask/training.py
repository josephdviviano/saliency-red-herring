from collections import OrderedDict
from loss import compare_activations, get_bre_loss, get_gradmask_loss
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import copy
import itertools
import logging
import manager.mlflow.logger as mlflow_logger
import notebooks.auto_ipynb as auto_ipynb
import numpy as np
import pprint
import random
import time, os, sys
import torch
import torch.nn as nn
import utils.configuration as configuration
import utils.monitoring as monitoring
import gc

# Fix backend so one can generate plots without Display set.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_LOG = logging.getLogger(__name__)
VIZ_IDX = 10


@mlflow_logger.log_experiment(nested=True)
@mlflow_logger.log_metric('best_valid_auc', 'best_test_auc')
def train(cfg, dataset_train=None, dataset_valid=None, dataset_test=None,
          recompile=True):

    print("Our config:")
    pprint.pprint(cfg)

    # Get information from configuration.
    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epochs = cfg['num_epochs']
    exp_name = cfg['experiment_name']
    recon_masked = cfg['recon_masked']
    recon_continuous = cfg['recon_continuous']

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
    scheduler = ReduceLROnPlateau(optim, mode='max')
    print(optim)

    criterion = torch.nn.CrossEntropyLoss()

    # Stats for the table.
    best_epoch, best_train_auc, best_valid_auc, best_test_auc = -1, -1, -1, -1
    metrics = []
    auc_valid = 0

    # Wrap the function for mlflow (optional).
    valid_wrap_epoch = mlflow_logger.log_metric('valid_acc')(test_epoch)
    test_wrap_epoch = mlflow_logger.log_metric('test_acc')(test_epoch)

    img_viz_train = dataset_train[VIZ_IDX]
    img_viz_valid = dataset_valid[VIZ_IDX]

    print("CUDA: ", cuda)
    for epoch in range(num_epochs):

        if cfg['viz']:
            render_img("train", epoch, img_viz_train, model, exp_name, cuda)
            render_img("valid", epoch, img_viz_valid, model, exp_name, cuda)

        # scheduler.step(auc_valid)

        avg_loss = train_epoch(epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion,
                               bre_lambda=cfg['bre_lambda'],
                               recon_lambda=cfg['recon_lambda'],
                               actdiff_lambda=cfg['actdiff_lambda'],
                               gradmask_lambda=cfg['gradmask_lambda'],
                               recon_masked=recon_masked,
                               recon_continuous=recon_continuous)

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

        # Early Stopping: compute best test_auc when we beat best valid score.
        if auc_valid > best_valid_auc:

            auc_test = test_wrap_epoch(name="test",
                                   epoch=epoch,
                                   model=model,
                                   device=device,
                                   data_loader=test_loader,
                                   criterion=criterion)

            best_train_auc = auc_train
            best_valid_auc = auc_valid
            best_test_auc = auc_test
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        # Update the stat dictionary with each epoch, append to metrics list.
        stat = {"epoch": epoch,
                "train_loss": avg_loss,
                "valid_auc": auc_valid,
                "train_auc": auc_train,
                "test_auc": auc_test,
                "best_train_auc": best_train_auc,
                "best_valid_auc": best_valid_auc,
                "best_test_auc": best_test_auc,
                "best_epoch": best_epoch}
        stat.update(configuration.process_config(cfg))
        metrics.append(stat)

    monitoring.log_experiment_csv(cfg, [best_valid_auc])

    results_dict = {'dataset_train': dataset_train,
                    'dataset_valid': dataset_valid,
                    'dataset_test': dataset_test}

    # Render gradients from the best model.
    if cfg['viz']:
        render_img(
            "best_train", best_epoch, img_viz_train, best_model, exp_name, cuda)
        render_img(
            "best_valid", best_epoch, img_viz_valid, best_model, exp_name, cuda)

    # Write best model to disk.
    output_dir = os.path.join('checkpoints', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Saves maxmasks and seed in the name.
    output_name = "best_model_{}_{}.pth.tar".format(
        cfg['seed'], cfg['maxmasks_train'])
    torch.save(best_model, os.path.join(output_dir, output_name))

    return (best_valid_auc, best_test_auc, metrics, results_dict)


@mlflow_logger.log_metric('train_loss')
def train_epoch(epoch, model, device, train_loader, optimizer,
                criterion, bre_lambda=0, recon_lambda=0, actdiff_lambda=0,
                gradmask_lambda=0, recon_masked=False, recon_continuous=False):

    model.train()

    # Simple simages should use recon_criterion=False
    if recon_continuous:
        recon_criterion = nn.MSELoss()
    else:
        recon_criterion = nn.BCELoss()

    # losses: cross-entropy + reconstruction + bre + actdiff + gradmask
    avg_clf_loss = []
    avg_actdiff_loss, avg_recon_loss, avg_bre_loss, avg_gradmask_loss = [], [], [], []
    avg_loss = []

    t = tqdm(train_loader)
    for batch_idx, (data, target, use_mask) in enumerate(t):

        optimizer.zero_grad()

        X, seg = data
        X, seg = X.to(device), seg.to(device)
        target, use_mask = target.to(device), use_mask.to(device)
        X.requires_grad = True

        # Get the inverse of the mask (1 *outside* ROI).
        # NB: If no mask for an image, the entire image should start out masked.
        #     The seg will be inverted to be all 0, and therefore the actdiff
        #     and gradmask loss will not be applied.
        seg = torch.abs(seg-1).byte()

        # Mask the data by shuffling pixels outside of the mask.
        if actdiff_lambda > 0 or recon_masked:

            X_masked = torch.clone(X)

            # Loop through batch images individually
            for b in range(seg.shape[0]):

                # Get all of the relevant values using this mask.
                b_seg = seg[b, :, :, :]
                tmp = X[b, b_seg]

                # Randomly permute those values.
                b_idx = torch.randperm(tmp.nelement())
                tmp = tmp[b_idx]
                X_masked[b, b_seg] = tmp

        # Activation Difference loss: activations of model when passed masked X.
        if actdiff_lambda > 0:

            # Sanity check!!
            # X_masked = X

            _, _  = model(X_masked)
            masked_activations = model.all_activations
        else:
            actdiff_loss = torch.zeros(1).to(device)

        # Classification loss: Do a forward pass with the data.
        y_pred, X_recon = model(X)
        clf_loss = criterion(y_pred, target)

        # Activation difference loss: Calculate l2 norm between the activations
        # using masked and unmasked data.
        if actdiff_lambda > 0:
            actdiff_loss = compare_activations(
                masked_activations, model.all_activations)

            actdiff_loss *= actdiff_lambda

        # Reconstruction Loss: The target reconstruction can be raw or masked.
        if recon_lambda > 0:

            if recon_masked:
                X_target = X_masked.detach()
            else:
                X_target = X.detach()

            recon_loss = recon_criterion(X_recon, X_target)
            recon_loss *= recon_lambda
        else:
            recon_loss = torch.zeros(1).to(device)

        # BRE loss. TODO: Remove?
        if bre_lambda > 0:
            bre_loss, me_loss, ac_loss, me_stats, ac_stats = get_bre_loss(model)
            bre_loss *= bre_lambda
        else:
            bre_loss = torch.zeros(1).to(device)

        # Gradmask loss: Get all gradients, and mask them for the target class.
        if gradmask_lambda > 0:
            input_grads = get_gradmask_loss(X, y_pred, model, target)

            # Used to apply gradmask only to the positive class.
            #input_grads *= target.float().reshape(-1, 1, 1, 1)

            input_grads *= seg.float()
            gradmask_loss = input_grads.abs().sum() * gradmask_lambda
        else:
            gradmask_loss = torch.zeros(1).to(device)

        # Final loss is all terms combined.
        loss = clf_loss + actdiff_loss + recon_loss + bre_loss + gradmask_loss

        # Reporting.
        avg_clf_loss.append(clf_loss.detach().cpu().numpy())
        avg_actdiff_loss.append(actdiff_loss.detach().cpu().numpy())
        avg_recon_loss.append(recon_loss.detach().cpu().numpy())
        avg_bre_loss.append(bre_loss.detach().cpu().numpy())
        avg_gradmask_loss.append(gradmask_loss.detach().cpu().numpy())
        avg_loss.append(loss.detach().cpu().numpy())

        t.set_description((
            ' train (clf={:4.4f} actdif={:4.4f} recon={:4.4f} bre={:4.4f} grd={:4.4f} tot={:4.4f})'.format(
                np.mean(avg_clf_loss),
                np.mean(avg_actdiff_loss),
                np.mean(avg_recon_loss),
                np.mean(avg_bre_loss),
                np.mean(avg_gradmask_loss),
                np.mean(avg_loss))))

        # Learning.
        if np.isnan(loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()

        loss.backward()
        optimizer.step()
        gc.collect()

    return np.mean(avg_loss)


def test_epoch(name, epoch, model, device, data_loader, criterion):

    model.eval()
    data_loss = 0

    targets = []
    predictions = []
    predictions_bin = []  # Keep track of actual predictions.

    with torch.no_grad():
        for (data, target, use_mask) in data_loader:

            x, seg = data[0].to(device), data[1].to(device)
            target = target.to(device)
            class_output, _ = model(x)

            # sum up batch loss
            data_loss += criterion(class_output, target).sum().item()

            targets.append(target.cpu().data.numpy())
            predictions.append(class_output.cpu().data.numpy())
            predictions_bin.append(
                torch.max(class_output, 1)[1].detach().cpu().numpy())

    auc = accuracy_score(np.concatenate(targets),
                         np.concatenate(predictions).argmax(axis=1))
    predictions_bin = np.hstack(predictions_bin)

    data_loss /= len(data_loader)
    print(epoch, 'Average {} loss: {:.4f}, AUC: {:.4f}, pred: {}/{}'.format(
        name, data_loss, auc,
        np.sum(predictions_bin==0), np.sum(predictions_bin==1)))

    return auc


def render_img(text, i, sample, model, exp_name, cuda=True):
    """
    Video protip: ffmpeg -y -i images/image-test-%d.png -vcodec libx264 aout.mp4
    """
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4,
                                             figsize=(12, 48), dpi=72)
    x, target, use_mask = sample

    if cuda:
        x_var = torch.autograd.Variable(x[0].unsqueeze(0).cuda(),
            requires_grad=True)
    else:
        x_var = torch.autograd.Variable(x[0].unsqueeze(0),
            requires_grad=True)

    model.eval()
    y_prime, x_prime = model(x_var)

    ax0.set_title(str(i) + " Masked Image")
    img = x[0][0].cpu().numpy()
    img = img / np.max(img)  # Scales the input image so that the maximum=1.
    seg = x[1][0].cpu().numpy() #* 0.5  # Makes mask bright, but not too bright.
    ax0.imshow(img + seg, interpolation='none', cmap='Greys_r')
    ax0.axis('off')

    ax1.set_title("nonhealthy d|y|/dx")
    gradmask = get_gradmask_loss(x_var, y_prime, model, torch.tensor(1.),
                                 "nonhealthy").detach().cpu().numpy()[0][0]
    ax1.imshow(np.abs(gradmask), cmap="jet", interpolation='none')
    ax1.axis('off')

    ax2.set_title("contrast d|y0-y1|/dx")
    gradmask = get_gradmask_loss(x_var, y_prime, model, torch.tensor(1.),
                                 "contrast").detach().cpu().numpy()[0][0]
    ax2.imshow(np.abs(gradmask), cmap="jet", interpolation='none')
    ax2.axis('off')

    ax3.set_title("Reconstruction")
    # Fails for models that output a nonsense reconstruction (CNN, ResNet).
    if isinstance(x_prime, torch.Tensor):
        ax3.imshow(x_prime[0][0].detach().cpu().numpy(),
                   interpolation='none', cmap='Greys_r')
        ax3.axis('off')
    else:
        ax3.remove()

    plt.tight_layout()
    output_dir = os.path.join('images', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig('{0}/image-{1}-{2:03d}.png'.format(output_dir, text, i),
                bbox_inches='tight', pad_inches=0)
    plt.close("all")


@mlflow_logger.log_experiment(nested=False)
def train_skopt(cfg, n_iter, base_estimator, n_initial_points, random_state,
                new_size, train_function=train):
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
    results_dict = {}
    best_valid_auc = 0
    best_metrics = None

    for i in range(n_iter):

        suggestion = optimizer.ask()
        this_cfg = generate_config(cfg, skopt_args, suggestion)
        opt_dict = this_cfg['optimizer']

        #metrics, metric, metric_test, state = train_function(this_cfg, recompile=state == {}, **state)
        this_valid_auc, this_test_auc, these_metrics, results_dict = train_function(
            this_cfg, recompile=results_dict == {}, **results_dict)

        print("{}: valid auc={}, test auc={}, cfg={}".format(
            i, this_valid_auc, this_test_auc, this_cfg))

        # We minimize the negative accuracy/AUC
        opt_results = optimizer.tell(suggestion, - this_valid_auc)

        #record metrics to write and plot
        if this_valid_auc > best_valid_auc:
            best_valid_auc = this_valid_auc
            best_test_auc = this_test_auc
            best_metrics = these_metrics
            print("{}: **New best valid/test AUC**: {}/{}".format(
                i, this_valid_auc, this_test_auc))

    print("{}: **Final best valid/test AUC**: {}/{}".format(
        i, best_valid_auc, best_test_auc))

    return best_metrics
