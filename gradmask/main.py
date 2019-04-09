import click
import training as training
import utils.configuration as configuration
import utils.monitoring as monitoring
import os, sys

penalise_grad_choices = ["contrast", "diff_from_ref", "nonhealthy"]

@click.group()
def run():
    pass

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('-seed', type=int, help='Seed for split and model')
@click.option('-penalise_grad', type=str, help='penalise_grad')
@click.option('-penalise_grad_usemasks', type=bool, help='penalise_grad_usemasks')
@click.option('-conditional_reg', type=bool, help='conditional_reg')
@click.option('-nsamples_train', type=int, help='nsamples_train')
@click.option('-new_size', type=int, help='new_size')
@click.option('-maxmasks_train', type=int, help='maxmasks_train')
@click.option('-num_epochs', type=int, help='num_epochs')
@click.option('-viz', type=bool, default=False, help='plot images')
def train(config, seed, penalise_grad, nsamples_train, num_epochs, penalise_grad_usemasks, conditional_reg, new_size, maxmasks_train, viz):
    cfg = configuration.load_config(config)
    if not seed is None:
        cfg["seed"] = seed

    if not penalise_grad is None:
        cfg["penalise_grad"] = penalise_grad

    if not penalise_grad_usemasks is None:
        cfg["penalise_grad_usemasks"] = penalise_grad_usemasks
        
    dataset = cfg["dataset"]["train"]
    if not cfg["nsamples_train"] is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = cfg["nsamples_train"]
        del cfg["nsamples_train"]
    if not nsamples_train is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = nsamples_train
      
    if not cfg["maxmasks_train"] is None:
        dataset[list(dataset.keys())[0]]["maxmasks"] = cfg["maxmasks_train"]
        del cfg["maxmasks_train"]
    if not maxmasks_train is None:
        dataset[list(dataset.keys())[0]]["maxmasks"] = maxmasks_train
        
    if not new_size is None:
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
        
    if not conditional_reg is None:
        cfg["conditional_reg"] = conditional_reg
    
    if not num_epochs is None:
        cfg["num_epochs"] = num_epochs

    cfg["viz"] = viz
    
    log_folder = get_log_folder_name(cfg)
    log_folder = "logs-single/" + str(hash(log_folder)).replace("-","_")
    print("Log folder:" + log_folder)
    
    if os.path.isdir(log_folder):
        print("Log folder exists. Will exit.")
        sys.exit(0)

    metrics, best_metric, testauc_for_best_validauc, state = training.train(cfg)
    
    # take best log and write it
    monitoring.save_metrics(metrics, folder="{}/stats".format(log_folder))
    

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('-seed', type=int, help='Seed for split and model')
@click.option('-penalise_grad', type=str, help='penalise_grad')
@click.option('-nsamples_train', type=int, help='nsamples_train')
@click.option('--n_iter', type=int, default=10, help='Configuration file.')
@click.option('--base_estimator', type=click.Choice(["GP", "RF", "ET", "GBRT"]), default="GP", help='Estimator.')
@click.option('--n_initial_points', type=int, default=5, help='Number of evaluations of func with initialization points before approximating it with base_estimator.')
@click.option('--train_function', type=str, default="train", help='Training function to optimize over.')
@click.option('-penalise_grad_usemasks', type=bool, help='penalise_grad_usemasks')
@click.option('-conditional_reg', type=bool, help='conditional_reg')
@click.option('-new_size', type=int, help='new_size')
@click.option('-maxmasks_train', type=int, help='maxmasks_train')
@click.option('-num_epochs', type=int, help='num_epochs')
@click.option('-viz', type=bool, default=False, help='plot images')
def train_skopt(config, seed, penalise_grad, nsamples_train, n_iter, base_estimator, n_initial_points, train_function,
               penalise_grad_usemasks, conditional_reg, new_size, maxmasks_train, num_epochs, viz):
    cfg = configuration.load_config(config)
    cfg["skopt"] = True
    if not seed is None:
        cfg["seed"] = seed

    if not penalise_grad is None:
        cfg["penalise_grad"] = penalise_grad

    if not penalise_grad_usemasks is None:
        cfg["penalise_grad_usemasks"] = penalise_grad_usemasks
        
    dataset = cfg["dataset"]["train"]
    if not cfg["nsamples_train"] is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = cfg["nsamples_train"]
        del cfg["nsamples_train"]
    if not nsamples_train is None:
        dataset[list(dataset.keys())[0]]["nsamples"] = nsamples_train
      
    if not cfg["maxmasks_train"] is None:
        dataset[list(dataset.keys())[0]]["maxmasks"] = cfg["maxmasks_train"]
        del cfg["maxmasks_train"]
    if not maxmasks_train is None:
        dataset[list(dataset.keys())[0]]["maxmasks"] = maxmasks_train
        
    if not new_size is None:
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
        
    if not conditional_reg is None:
        cfg["conditional_reg"] = conditional_reg
    
    if not num_epochs is None:
        cfg["num_epochs"] = num_epochs
       
    if not new_size is None:
        dataset = cfg["dataset"]["train"]
        dataset[list(dataset.keys())[0]]["new_size"] = new_size
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
                                            train_function=getattr(training, train_function))
    
    # take best log and write it
    monitoring.save_metrics(metrics_bestrun, folder="{}/stats".format(log_folder))
    
    
def get_log_folder_name(cfg):
        
    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epochs = cfg['num_epochs']
    # maxmasks = cfg['maxmasks']
    penalise_grad = cfg['penalise_grad']
    penalise_grad_usemask = cfg.get('penalise_grad_usemask', False)
    conditional_reg = cfg.get('conditional_reg', False)
    penalise_grad_lambdas = [cfg['penalise_grad_lambda_1'], cfg['penalise_grad_lambda_2']]

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
        elif "penalise_grad_" in k:
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

