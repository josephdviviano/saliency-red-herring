# gradmask

## Installation

Create a virtual environment:
```
conda create -n gradmask python=3.5
source activate gradmask # Need to run this everytime we want to run the project.
```

then install the requirements as well as the project:
```
sudo pip install -e .
sudo pip -r requirements.txt
```

Alternatively, you can run the installation script by doing `source install.sh`.


## Training
For now, a config file needs to be defined to run the experiments. An exemple can be found in `config/mnist.yml`
To run the code, simply launch
```
gradmask train --config gradmask/config/mnist.yml
```

## Monitoring
The code right now will log all experiments in the `logs/experiments.csv` file. The time of saving, git hash, config, and best accuracy on the valid set is saved.

## Adding my own model and dataset
To add your new fancy model/dataset that is not already in torchvision, here are the steps to follow:
1. Add you class in `models/datasets` either in a new file or a preexisting one.
2. Register the model/dataset with the appropriate function (ex: `@register.setmodelname("cnn_example")`)

## Bayesian hyperparameters optimization with skopt
Steps to launch bayesian hyperparameters search:
1. In your `.yml` config file, choose the parameters you want to optimize (i.e. learning rate).
2. Replace the value with  the search parameters. For example:
    ```
    # Optimizer
    optimizer:
      Adam:
        lr: "Real(10**-4, 10**-2, 'log-uniform')"
    ```
    search the learning rate in the range (0.01, 0.0001), on a log scale. For more examples, please refer to [the official documentation](https://scikit-optimize.github.io/#skopt.BayesSearchCV).
3. Launch your config file with `iia.gradmask train-skopt --config your/config/path.yml`

An config example can be found in `config/mnist_skopt.yml`

