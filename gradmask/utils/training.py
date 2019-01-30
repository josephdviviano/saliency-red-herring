from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import torch
import numpy as np
import random
from collections import OrderedDict
import copy
import gradmask.utils.configuration as configuration
import gradmask.utils.monitoring as monitoring
import time


_LOG = logging.getLogger(__name__)

def train(cfg):

    print("Our config:", cfg)
    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epoch = cfg['epoch']
    nsamples = cfg['nsamples']
    maxmasks = cfg['maxmasks']
    penalise_grad = cfg['penalise_grad']
    log_folder = "logs/{}".format(time.asctime())

    device = 'cuda' if cuda else 'cpu'

    # Setting the seed.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if cuda:
        torch.cuda.manual_seed_all(seed)

    # Dataset
    # transform
    tr_train = configuration.setup_transform(cfg, 'train')
    tr_valid = configuration.setup_transform(cfg, 'valid')
    tr_test= configuration.setup_transform(cfg, 'test')

    # The dataset
    dataset_train = configuration.setup_dataset(cfg, 'train')(tr_train)
    dataset_valid = configuration.setup_dataset(cfg, 'valid')(tr_valid)
    dataset_test = configuration.setup_dataset(cfg, 'test')(tr_test)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'])
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'])

    test_loader = torch.utils.data.DataLoader(dataset_test, 
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'])

    
    model = configuration.setup_model(cfg).to(device)
    print(model)
    # TODO: checkpointing

    # Optimizer
    optim = configuration.setup_optimizer(cfg)(model.parameters())
    print(optim)

    criterion = torch.nn.CrossEntropyLoss()

    # Aaaaaannnnnd, here we go!
    best_metric = 0.
    metrics = []

    for epoch in range(num_epoch):

        train_epoch(epoch=epoch,
                    model=model,
                    device=device,
                    optimizer=optim,
                    train_loader=train_loader,
                    criterion=criterion,
                    penalise_grad=penalise_grad)

        auc_valid = test_epoch(model=model,
                      device=device,
                      data_loader=valid_loader,
                      criterion=criterion)

        if auc_valid > best_metric:
            best_metric = auc_valid

        # Save monitor the auc/loss, etc.
        auc_test = test_epoch(model=model,
                      device=device,
                      data_loader=test_loader,
                      criterion=criterion)

        stat = {"epoch": epoch,
                "trainloss": -1,# TODO np.asarray(batch_loss).mean(),
                "validauc": auc_valid,
                "testauc": auc_test}
        stat.update(cfg)

        metrics.append(stat)

        if epoch % 20 == 0:
            monitoring.save_metrics(metrics, folder="{}/stats".format(log_folder))
        

    monitoring.save_metrics(metrics, folder="{}/stats".format(log_folder))
    monitoring.log_experiment_csv(cfg, [best_metric])
    return best_metric

def train_epoch(epoch, model, device, train_loader, optimizer, criterion, penalise_grad):

    model.train()

    for batch_idx, (data, target, use_mask) in enumerate(tqdm(train_loader)):

        #use_mask = torch.ones((len(target))) # TODO change here.
        optimizer.zero_grad()
        
        x, seg = data
        x, seg, target, use_mask = x.to(device), seg.to(device), target.to(device), use_mask.to(device)
        x.requires_grad=True

        class_output, representation = model(x)

        loss = criterion(class_output, target)

        # TODO: this place if suuuuper slow. Should be optimized by using advance indexing or something.
        if penalise_grad:
            input_grads = torch.autograd.grad(outputs=torch.abs(class_output[:, 1]).sum(),  # select the non healthy class
                                    inputs=x, allow_unused=True,
                                    create_graph=True)[0]

            # only apply to positive examples
            input_grads = target.float().reshape(-1, 1, 1, 1) * input_grads

            res = input_grads * (1 - seg.float())
            gradmask_loss = epoch * (res ** 2)

            # Simulate that we only have some masks
            gradmask_loss = use_mask.reshape(-1, 1).float() * \
                            gradmask_loss.float().reshape(-1, np.prod(gradmask_loss.shape[1:]))

            gradmask_loss = gradmask_loss.sum()
            loss = loss + gradmask_loss

        loss.backward()

        optimizer.step()

def test_epoch(model, device, data_loader, criterion):

    model.eval()
    data_loss = 0

    targets = []
    predictions = []

    with torch.no_grad():
        for (data, target, use_mask) in data_loader:

            x, seg, target = data[0].to(device), data[1].to(device), target.to(device)
            class_output, representation = model(x)
            
            data_loss += criterion(class_output, target).sum().item() # sum up batch loss
            
            targets.append(target.cpu().data.numpy())
            predictions.append(class_output.cpu().data.numpy())

    auc = accuracy_score(np.concatenate(targets), np.concatenate(predictions).argmax(axis=1))

    data_loss /= len(data_loader.dataset)
    print('\nAverage loss: {:.4f}, AUC: {:.4f}\n'.format(data_loss, auc))
    return auc

def train_skopt(cfg, n_iter, base_estimator, n_initial_points, random_state, train_function=train):

    """
    Do a Bayesian hyperparameter optimization.

    :param cfg: Configuration file.
    :param n_iter: Number of Bayesien optimization steps.
    :param base_estimator: skopt Optimization procedure.
    :param n_initial_points: Number of random search before starting the optimization.
    :param random_state: seed.
    :param train_function: The trainig procedure to optimize. The function should take a dict as input and return a metric maximize.
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
            set_key(new_config, key, new_values[i])
        return new_config

    # Sparse the parameters that we want to optimize
    skopt_args = OrderedDict(parse_dict(cfg))

    # Create the optimizer
    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator=base_estimator,
                                n_initial_points=n_initial_points,
                                random_state=random_state)

    for _ in range(n_iter):

        # Do a bunch of loops.
        suggestion = optimizer.ask()
        this_cfg = generate_config(cfg, skopt_args, suggestion)
        optimizer.tell(suggestion, - train_function(this_cfg)) # We minimize the negative accuracy/AUC

    # Done! Hyperparameters tuning has never been this easy.

