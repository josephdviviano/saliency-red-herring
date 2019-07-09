#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

conda create -n activmask python=3.5
source activate activmask

$(which pip) install -e .
$(which pip) install -r requirements.txt
