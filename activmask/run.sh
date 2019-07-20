#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask

# Main experiment grid!
python -u main.py train --config config/synth_ae_actdiff+recon.yml
python -u main.py train --config config/synth_ae_actdiff.yml
python -u main.py train --config config/synth_ae_clf.yml
python -u main.py train --config config/synth_ae_clfmasked.yml
python -u main.py train --config config/synth_ae_recon.yml
python -u main.py train --config config/synth_ae_reconmasked.yml

python -u main.py train --config config/synth_cnn_actdiff.yml
python -u main.py train --config config/synth_cnn_clf.yml
python -u main.py train --config config/synth_cnn_clfmasked.yml

python -u main.py train --config config/synth_resnet_actdiff.yml
python -u main.py train --config config/synth_resnet_clf.yml
python -u main.py train --config config/synth_resnet_clfmasked.yml

python -u main.py train --config config/synth_unet_actdiff+recon.yml
python -u main.py train --config config/synth_unet_actdiff.yml
python -u main.py train --config config/synth_unet_clf.yml
python -u main.py train --config config/synth_unet_clfmasked.yml
python -u main.py train --config config/synth_unet_recon.yml
python -u main.py train --config config/synth_unet_reconmasked.yml

rc=$?
echo "Exit code is $?"
if  [[ $rc == 0 ]]; then
    echo "Command succeeded"
else
    echo "Command failed"
fi
