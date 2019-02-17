#!/bin/bash
hostname
#source ~/.bashrc

#for s in {0..4};
#do
#    echo "Doing seed $s"
python -u main.py ${@} 2>&1
#done
