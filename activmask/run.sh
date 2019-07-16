#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask

# Main experiment grid!
python -u main.py train --config config/synth_ae_activmask.yml #$@
python -u main.py train --config config/synth_ae_clf.yml
python -u main.py train --config config/synth_ae_noactdiff.yml
python -u main.py train --config config/synth_ae_norecon.yml
python -u main.py train --config config/synth_ae_reconmasked.yml

python -u main.py train --config config/synth_unet_activmask.yml
python -u main.py train --config config/synth_unet_clf.yml
python -u main.py train --config config/synth_ae_noactdiff.yml
python -u main.py train --config config/synth_unet_norecon.yml
python -u main.py train --config config/synth_ae_reconmasked.yml

python -u main.py train --config config/synth_cnn_clf.yml
python -u main.py train --config config/synth_cnn_norecon.yml
python -u main.py train --config config/synth_resnet_clf.yml
python -u main.py train --config config/synth_resnet_norecon.yml

# Control experiment: classification using masked training data.
python -u main.py train --config config/synth_cnn_maskonly.yml
python -u main.py train --config config/synth_resnet_maskonly.yml

rc=$?
echo "Exit code is $?"
if  [[ $rc == 0 ]]; then
    echo "Command succeeded"
else
    echo "Command failed"
fi
