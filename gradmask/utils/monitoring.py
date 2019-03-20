import time
import json
import os
from collections import OrderedDict
import pickle as pkl

# A very basic logging setup to get a minimum of monitoring.
# You can do a better one.
def log_experiment_csv(config, stuff, folder='logs', file_name="experiment_table.csv"):

    import csv
    import git

    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(os.path.join(folder, file_name), 'a') as csvfile:

        writer = csv.writer(csvfile, delimiter='\t', quotechar='\'', quoting=csv.QUOTE_MINIMAL)

        # Get the time
        tt = time.ctime()

        # Get the git hash
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
        except git.exc.InvalidGitRepositoryError:
            sha = 0

        # Get the config file
        cc = json.dumps(config)

        # Add the metrics and whatnot
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
                            if type(item) == float:
                                i = round(i, 4)
                            output_dict["{}_{}_{}".format(mode, key, k)] = i # so we don't have e.g. train_dataset_MSD_mode
                    except:
                        # config is of the form main_key: key_val: more_key_val_pairs e.g. optimiser: Adam: lr: 0.001
                        sub_dict = config[key]
                        main_key_value = list(sub_dict.keys())[0]
                        output_dict[key] = main_key_value
                        sub_sub_dict = sub_dict[main_key_value] # e.g. name of optimiser, name of dataset
                        for k, i in sub_sub_dict.items():
                            if type(item) == float:
                                i = round(i, 4)
                            output_dict["{}_{}".format(key, k)] = i
                else:
                    # standard key: val pair
                    if type(item) == float:
                        item = round(item, 4)
                        
                    output_dict[key] = item

        # line = [tt, sha] + list(stuff) + [cc] + list(OrderedDict(output_dict).values())
        line = [tt, sha] + list(stuff) + list(OrderedDict(output_dict))
        writer.writerow(line)

def save_metrics(metrics, folder='logs', file_name='metrics.pkl'):

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    pkl.dump(metrics, open(os.path.join(folder, file_name), 'wb'))

