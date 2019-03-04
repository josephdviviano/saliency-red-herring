#!/bin/bash

source $HOME/.bashrc
source activate gradmask

gradmask train --config gradmask/config/lungmsd.yml

echo 'DONE'
