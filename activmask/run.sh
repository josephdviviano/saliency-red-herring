#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc
#source activate activmask

EXPERIMENTS="synth livermsd"

for i in {1..10}; do
    echo "ITERATION ${i}:"

    for exp in ${EXPERIMENTS}; do
        for file in $(ls config/${exp}/${exp}_*); do
            python -u main.py train --config ${file}
        done
    done
done
