#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask

python -u main.py --config config/synth.yml $@

rc=$?
echo "Exit code is $?"
if  [[ $rc == 0 ]]; then
    echo "Command succeeded"
else
    echo "Command failed"
fi
