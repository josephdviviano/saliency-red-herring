#!/bin/bash
#Install script. setup the virtual environment and everything

virtualenv .env
. .env/bin/activate # Need to run this everytime we want to run the project. 
sudo $(which pip) install -e .
sudo $(which pip) install -r requirements.txt

export LC_ALL=C.UTF-8 
export LANG=C.UTF-8

