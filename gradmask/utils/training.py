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
import utils.configuration as configuration
import utils.monitoring as monitoring
import time, os, sys


_LOG = logging.getLogger(__name__)


def process_config(config):
    """
        Here to deconstruct the config file from nested dicts into a flat structure
    """
    output_dict = {}
    for mode in ['train', 'valid','test']:
        for key, item in config.items():
            if type(config[key]) == dict:
                # if the 'value' is actually a dict, iterate through and collect train/valid/test values
                try:
                    # config is of the form main_key: train/test/valid: key_val: more_key_val_pairs
                    sub_dict = config[key][mode]
                    main_key_value = list(sub_dict.keys())[0]
                    output_dict["{}_{}".format(mode, key)] = main_key_value
                    sub_sub_dict = sub_dict[main_key_value] # e.g. name of optimiser, name of dataset
                    for k, i in sub_sub_dict.items():
                        output_dict["{}_{}_{}".format(mode, key, k)] = i # so we don't have e.g. train_dataset_MSD_mode
                except:
                    # config is of the form main_key: key_val: more_key_val_pairs e.g. optimiser: Adam: lr: 0.001
                    sub_dict = config[key]
                    main_key_value = list(sub_dict.keys())[0]
                    output_dict[key] = main_key_value
                    sub_sub_dict = sub_dict[main_key_value] # e.g. name of optimiser, name of dataset
                    for k, i in sub_sub_dict.items():
                        output_dict["{}_{}".format(key, k)] = i
            else:
                # standard key: val pair
                output_dict[key] = item
    return output_dict


def train(cfg):

    print("Our config:", cfg)
    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epochs = cfg['num_epochs']
    nsamples = cfg['nsamples']
    maxmasks = cfg['maxmasks']
    penalise_grad = cfg['penalise_grad']
    penalise_grad_usemask = cfg.get('penalise_grad_usemask', False)
    conditional_reg = cfg.get('conditional_reg', False)
    
    ncfg = dict(cfg)
    del ncfg["cuda"]
    del ncfg["num_epochs"]
    del ncfg["transform"]
    ncfg["dataset"] = list(ncfg["dataset"]["train"].keys())[0]
    log_folder = "logs/" + str(ncfg).replace("'","").replace(" ","").replace("{","_").replace("}","_")
    print("Log folder:" + log_folder)
    if os.path.isdir(log_folder):
        print("Log folder exists. Will exit.")
        
        sys.exit(0)
    
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
    best_testauc_for_validauc = 0.
    metrics = []

    for epoch in range(num_epochs):

        avg_loss = train_epoch( epoch=epoch,
                                model=model,
                                device=device,
                                optimizer=optim,
                                train_loader=train_loader,
                                criterion=criterion,
                                penalise_grad=penalise_grad,
                                penalise_grad_usemask=penalise_grad_usemask,
                                conditional_reg=conditional_reg
                              )

        auc_valid = test_epoch(epoch=epoch,
                               model=model,
                               device=device,
                               data_loader=valid_loader,
                               criterion=criterion)

        # Save monitor the auc/loss, etc.
        auc_test = test_epoch(epoch=epoch,
                              model=model,
                              device=device,
                              data_loader=test_loader,
                              criterion=criterion)
        
        if auc_valid > best_metric:
            best_metric = auc_valid
            best_testauc_for_validauc = auc_test

        stat = {"epoch": epoch,
                "trainloss": avg_loss, 
                "validauc": auc_valid,
                "testauc": auc_test,
                "best_testauc_for_validauc": best_testauc_for_validauc}
        stat.update(process_config(cfg))

        metrics.append(stat)

#         if epoch % 20 == 0:
#             monitoring.save_metrics(metrics, folder="{}/stats".format(log_folder))
        

    monitoring.save_metrics(metrics, folder="{}/stats".format(log_folder))
    monitoring.log_experiment_csv(cfg, [best_metric])
    return best_metric, best_testauc_for_validauc


def train_epoch(epoch, model, device, train_loader, optimizer, criterion, penalise_grad, penalise_grad_usemask, conditional_reg):

    model.train()
    avg_loss = []
    for batch_idx, (data, target, use_mask) in enumerate(tqdm(train_loader)):

        #use_mask = torch.ones((len(target))) # TODO change here.
        optimizer.zero_grad()
        
        x, seg = data
        x, seg, target, use_mask = x.to(device), seg.to(device), target.to(device), use_mask.to(device)
        x.requires_grad=True

        class_output, representation = model(x)

        loss = criterion(class_output, target)

        # TODO: this place is suuuuper slow. Should be optimized by using advance indexing or something.
        if penalise_grad != "False":
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
                healthy_mask = target.float().reshape(-1, 1, 1, 1).clone()
                healthy_mask[target.float().reshape(-1, 1, 1, 1) == 0] = 1
                healthy_mask[target.float().reshape(-1, 1, 1, 1) != 0] = 0
                
                diff = torch.FloatTensor()
                diff = diff.to(device)
                
                print("Activation lengths: ", len(model.all_activations))
                for i in range(len(model.all_activations)):
                    a = model.all_activations[i]
                    # print("Activation shape: ", a.shape)
                    
                    healthy_batch = healthy_mask * a
                    # print("Healthy batch: ", healthy_batch.shape)
                    
                    # 2) detach grads for the healthy samples
                    healthy_batch = healthy_batch.detach()
                
                    # 3) get batch-wise average of activations per layer
                    batch_avg_healthy = torch.mean(healthy_batch, dim=0)
                    # print("Healthy batch avg: ", batch_avg_healthy.shape)
                    
                    # 4) update global reference layer average in model's deep_lift_ref attr
                    if len(model.ref) < len(model.all_activations):
                        # for the first iteration, just make the model.ref == batch_avg_healthy for that layer
                        model.ref.append(batch_avg_healthy)
                    else:
                        # otherwise, a rolling average
                        model.ref[i] = model.ref[i] * 0.8 + batch_avg_healthy
                
                # 5) TODO: somehow incorporate std to allowing regions of variance in the healthy images
                
                # use the reference layers to get the diff-to-ref of each layer and output contribution scores
                # contribution scores should be the input_grads? Should be a single matrix of values for how each input
                # pixel contributes to the output layer, no? Like all the layer-wise diff-from-ref get condensed into
                # one thing based on sum(contribution_scores of (delta_x_i, delta_t)) = delta_t
                # 1) for each layer, t - t0, then mask the unhealthy ones
                # 2) flatten, 3) stack or join together somehow?, 4) L1 norm, 5) input grads
                    diff = torch.cat((diff, torch.flatten((a - model.ref[i]) * target.float().reshape(-1, 1, 1, 1))))
                
                print(diff.shape)
                input_grads = torch.autograd.grad(outputs=torch.abs(diff).sum(),
                                                 inputs=x, allow_unused=True, create_graph=True)[0]
            
            elif penalise_grad == "masd_style":
                # In the style of the Model-Agnostic Saliency Detector paper: https://arxiv.org/pdf/1807.07784.pdf
                
                print(class_output.shape, representation.shape)
                # outputs should now be: abs(diff(area_seg - area_saliency_map)) + 
                # WIP
            else:
                raise Exception("Unknown style of penalise_grad. Options are: contrast, nonhealthy, diff_from_ref")

            # only apply to positive examples
            if penalise_grad != "diff_from_ref":
                # only do it here because the masking happens elsewhere in the diff_from_ref architecture
                print("target mask: ", target.float().reshape(-1, 1, 1, 1).shape)
                input_grads = target.float().reshape(-1, 1, 1, 1) * input_grads

            if conditional_reg:
                temp_softmax = torch.softmax(class_output, dim=1).detach()
                certainty_mask = 1 - torch.argmax((temp_softmax > 0.95).float(), dim=1)
                input_grads = certainty_mask.float().reshape(-1, 1, 1, 1) * input_grads
            
            if penalise_grad_usemask:
                res = input_grads * (1 - seg.float())
            else:
                res = input_grads

            #res = input_grads * (1 - seg.float())

            gradmask_loss = epoch * (res ** 2)

            # Simulate that we only have some masks
            gradmask_loss = use_mask.reshape(-1, 1).float() * \
                            gradmask_loss.float().reshape(-1, np.prod(gradmask_loss.shape[1:]))

            gradmask_loss = gradmask_loss.sum()
            loss = loss + loss*gradmask_loss

        avg_loss.append(loss.detach().cpu().numpy())
        loss.backward()

        optimizer.step()
    return np.mean(avg_loss)


def test_epoch(epoch, model, device, data_loader, criterion):

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
    print(epoch, 'Average test loss: {:.4f}, Test AUC: {:.4f}'.format(data_loss, auc))
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
        optimizer.tell(suggestion, - train_function(this_cfg)[0]) # We minimize the negative accuracy/AUC

    # Done! Hyperparameters tuning has never been this easy.

