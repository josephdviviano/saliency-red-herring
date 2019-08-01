#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask


for i in {1..10}; do

    echo "ITERATION ${i}:"

    # Synthetic dataset experiments.
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

    # Liver Experiments
    python -u main.py train --config config/livermsd_ae_actdiff+recon.yml
    python -u main.py train --config config/livermsd_ae_actdiff.yml
    python -u main.py train --config config/livermsd_ae_clf.yml
    python -u main.py train --config config/livermsd_cnn_actdiff.yml
    python -u main.py train --config config/livermsd_cnn_clf.yml
    python -u main.py train --config config/livermsd_resnet_actdiff.yml
    python -u main.py train --config config/livermsd_resnet_clf.yml
    python -u main.py train --config config/livermsd_unet_actdiff+recon.yml
    python -u main.py train --config config/livermsd_unet_actdiff.yml
    python -u main.py train --config config/livermsd_unet_clf.yml

done
