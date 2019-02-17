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
        line = [tt, sha] + list(stuff) + [cc] + list(OrderedDict(config).values())
        writer.writerow(line)

def save_metrics(metrics, folder='logs', file_name='metrics.pkl'):

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    pkl.dump(metrics, open(os.path.join(folder, file_name), 'wb'))

