import click
import training as training
import utils.configuration as configuration
import utils.monitoring as monitoring
import os, sys

# agg backend so I can use matplotlib without x server.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@click.group()
def run():
    pass

@run.command()
@click.option('--config', '-cgf',
              type=click.Path(exists=True, resolve_path=True),
              help='Configuration file.')
@click.option('-seed', type=int, help='Seed for split and model')
@click.option('-nsamples_train', type=int, help='nsamples_train')
@click.option('-new_size', type=int, help='new_size')
@click.option('-maxmasks_train', type=float, default=1, help='maxmasks_train')
@click.option('-num_epochs', type=int, help='num_epochs')
@click.option('-viz', type=bool, default=False, help='plot images')
@click.option('-lr', type=float, default=False, help='learning rate')
def train(config, seed, nsamples_train, num_epochs, new_size,
          maxmasks_train, viz, lr):

    cfg = configuration.load_config(config)
    if not seed is None:
        cfg["seed"] = seed

    dataset = cfg["dataset"]["train"]
    optimizer = cfg["optimizer"]
    this_optimizer = list(optimizer.keys())[0]

    if not cfg["nsamples_train"] is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = cfg["nsamples_train"]
        del cfg["nsamples_train"]

    if not nsamples_train is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = nsamples_train

    # maxmasks is always set.
    dataset[list(dataset.keys())[0]]["maxmasks"] = maxmasks_train
    cfg["maxmasks_train"] = maxmasks_train

    if not new_size is None:
        # new_size passed in from the command line
        print("USING NEW SIZE from CMD: ", new_size)
        dataset[list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["valid"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["test"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["model"][list(cfg["model"].keys())[0]]["img_size"] = int(new_size)
        cfg["new_size"] = int(new_size)
    elif not cfg.get("new_size", None): # if it hasn't been set in the config
        new_size = 100
        print("USING NEW SIZE DEFAULT: ", new_size)
        cfg["new_size"] = new_size # default
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
        cfg["dataset"]["valid"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["test"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["model"][list(cfg["model"].keys())[0]]["img_size"] = new_size
    else:
        new_size = cfg["new_size"]
        print("USING NEW SIZE from CFG: ", new_size)
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
        cfg["dataset"]["valid"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["test"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["model"][list(cfg["model"].keys())[0]]["img_size"] = new_size

    if not num_epochs is None:
        cfg["num_epochs"] = num_epochs

    if not lr is None:
        cfg["optimizer"][this_optimizer]["lr"] = lr

    cfg["viz"] = viz

    log_folder = get_log_folder_name(cfg)
    log_folder = "logs-single/" + str(hash(log_folder)).replace("-","_")
    print("Log folder:" + log_folder)

    if os.path.isdir(log_folder):
        print("Log folder exists. Will exit.")
        sys.exit(0)

    best_metric, testauc_for_best_validauc, metrics, state = training.train(cfg)

    # Plot train / valid AUC.
    train_auc, valid_auc = [], []

    for metric in metrics:
        train_auc.append(metric['train_auc'])
        valid_auc.append(metric['valid_auc'])

    plt.plot(train_auc)
    plt.plot(valid_auc)
    plt.legend(['train', 'valid'])
    plt.xlabel('epoch')
    plt.ylabel('AUC')

    # take best log and write it
    monitoring.save_metrics(metrics, folder="{}/stats".format(log_folder))
    plt.savefig("{}/auc.jpg".format(log_folder))

@run.command()
@click.option('--config', '-cgf',
              type=click.Path(exists=True, resolve_path=True),
              help='Configuration file.')
@click.option('-seed', type=int, help='Seed for split and model')
@click.option('-nsamples_train', type=int, help='nsamples_train')
@click.option('--n_iter', type=int, default=10, help='Configuration file.')
@click.option('--base_estimator',
              type=click.Choice(["GP", "RF", "ET", "GBRT"]),
              default="GP", help='Estimator.')
@click.option('--n_initial_points',
              type=int, default=5,
              help='Number of evaluations of func with initialization points before approximating it with base_estimator.')
@click.option('--train_function',
              type=str, default="train",
              help='Training function to optimize over.')
@click.option('-new_size', type=int, help='new_size')
@click.option('-maxmasks_train', type=int, default=1, help='maxmasks_train')
@click.option('-num_epochs', type=int, help='num_epochs')
@click.option('-viz', type=bool, default=False, help='plot images')
def train_skopt(config, seed, nsamples_train, n_iter, base_estimator,
                n_initial_points, train_function, new_size, maxmasks_train,
                num_epochs, viz):

    cfg = configuration.load_config(config)
    cfg["skopt"] = True
    if not seed is None:
        cfg["seed"] = seed

    dataset = cfg["dataset"]["train"]
    if not cfg["nsamples_train"] is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = cfg["nsamples_train"]
        del cfg["nsamples_train"]
    if not nsamples_train is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = nsamples_train

    # maxmasks is always set.
    dataset[list(dataset.keys())[0]]["maxmasks"] = maxmasks_train
    cfg["maxmasks_train"] = maxmasks_train

    if not new_size is None:
        # new_size passed in from the command line
        print("USING NEW SIZE from CMD: ", new_size)
        dataset[list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["valid"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["test"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["model"][list(cfg["model"].keys())[0]]["img_size"] = int(new_size)
        cfg["new_size"] = int(new_size)
    elif not cfg.get("new_size", None): # if it hasn't been set in the config
        new_size = 100
        print("USING NEW SIZE DEFAULT: ", new_size)
        cfg["new_size"] = new_size # default
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
        cfg["dataset"]["valid"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["test"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["model"][list(cfg["model"].keys())[0]]["img_size"] = new_size
    else:
        new_size = cfg["new_size"]
        print("USING NEW SIZE from CFG: ", new_size)
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
        cfg["dataset"]["valid"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["dataset"]["test"][list(dataset.keys())[0]]["new_size"] = int(new_size)
        cfg["model"][list(cfg["model"].keys())[0]]["img_size"] = new_size

    if not num_epochs is None:
        cfg["num_epochs"] = num_epochs

    # do logging stuff and break if already done

    cfg['viz'] = viz
    log_folder = get_log_folder_name(cfg)
    log_folder = "logs/" + str(hash(log_folder)).replace("-","_")
    print("Log folder:" + log_folder)

    if os.path.isdir(log_folder):
        print("Log folder exists. Will exit.")
        sys.exit(0)

    metrics_bestrun = training.train_skopt( cfg, n_iter=n_iter,
                                            base_estimator=base_estimator,
                                            n_initial_points=n_initial_points,
                                            random_state=seed,
                                            new_size=cfg["new_size"],
                                            train_function=getattr(training, train_function))

    # take best log and write it
    monitoring.save_metrics(metrics_bestrun, folder="{}/stats".format(log_folder))


def get_log_folder_name(cfg):

    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epochs = cfg['num_epochs']
    # maxmasks = cfg['maxmasks']
    conditional_reg = cfg.get('conditional_reg', False)

    ncfg = dict(cfg)
    del ncfg["cuda"]
    del ncfg["num_epochs"]
    del ncfg["transform"]
    dataset_cfg = cfg["dataset"]["train"]
    #print(dataset_cfg[list(dataset_cfg.keys())[0]])
    ncfg["nsamples_train"] = dataset_cfg[list(dataset_cfg.keys())[0]]["nsamples"]
    ncfg["maxmasks"] = dataset_cfg[list(dataset_cfg.keys())[0]]["maxmasks"]
    ncfg["dataset"] = list(ncfg["dataset"]["train"].keys())[0]

    log_ncfg_tmp = configuration.process_config(ncfg)
    log_ncfg = {}
    # here to trim the folder name length or the whole thing borks
    for k, i in log_ncfg_tmp.items():
        if "lambda" in k:
            # because we are JUST over the file/folder name limit of 255 char lol
            log_ncfg[k.replace("penalise_grad_","")] = i
        elif all([w not in k for w in ["n_iter", "manager", "skopt", "shuffle"]]):
            log_ncfg[k] = i
        elif "conditional" in k:
            log_ncfg["cndreg"] = i

    log_folder = "logs/" + str(log_ncfg)
    for s in ["'", " ", "{","}",",",":","_","*","(",")","\""]:
        log_folder = log_folder.replace(s, "")
    return log_folder

def main():
    run()

if __name__ == '__main__':
    main()

