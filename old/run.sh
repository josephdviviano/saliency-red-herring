#!/bin/bash
hostname
#source ~/.bashrc
jupyter nbconvert --to python $1  
FILENAME=$1
echo ${FILENAME%%.*}
cat ${FILENAME%%.*}.py | sed -e "s/get_ipython().magic(u'matplotlib inline')//"  > ${FILENAME%%.*}-clean.py
python -u ${FILENAME%%.*}-clean.py ${@:2} 2>&1
