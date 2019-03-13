#!/bin/bash
hostname
python3 -u main.py train --config config/lungmsd.yml $@

echo "Exit code is $?"
rc=$?
if  [[ $rc == 0 ]]; then
    echo "Command succeeded"
else
    echo "Command failed"
fi

#for i in {0..5}; do
#    python3 -u main.py train --config config/lungmsd.yml -seed $i -penalise_grad=False
#    python3 -u main.py train --config config/lungmsd.yml -seed $i -penalise_grad=nonhealthy
#done
