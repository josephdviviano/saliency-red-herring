#!/bin/bash
#Install script. setup the virtual environment and everything

conda create -n gradmask python=3.5
source activate gradmask # Need to run this everytime we want to run the project.
sudo $(which pip) install -e .
sudo $(which pip) install -r requirements.txt

export LC_ALL=C.UTF-8 
export LANG=C.UTF-8

