#!/bin/bash
hostname
source $HOME/.bashrc
#source activate gradmask

python3 -u main.py train-skopt --config config/lungmsd_skopt.yml $@

rc=$?
echo "Exit code is $?"
if  [[ $rc == 0 ]]; then
    echo "Command succeeded"
else
    echo "Command failed"
fi

#for i in {0..5}; do
#    python3 -u main.py train --config config/lungmsd.yml -seed $i -penalise_grad=False
#    python3 -u main.py train --config config/lungmsd.yml -seed $i -penalise_grad=nonhealthy
#done
