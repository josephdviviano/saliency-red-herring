#!/bin/bash

mkdir -p cluster_logs

EXPERIMENTS="rsna-search synth-search xray-search"
SEED=1234
BASE_DIR="/home/cohenjos/scratch"


for exp in ${EXPERIMENTS}; do
    for file in $(ls activmask/config/${exp}/${exp}*); do

        filename=$(basename ${file})
        filename="${filename%.*}"
        filename="${filename}_${SEED}"
        runscript="${BASE_DIR}/${filename}.pbs"

        # Generates a job script.
        cat << EOF > ${runscript}
#!/bin/bash
#SBATCH --job-name=${filename}
#SBATCH --output=../cluster_logs/${filename}_out.txt
#SBATCH --error=../cluster_logs/${filename}_err.txt
#SBATCH --time=12:00:00
#SBATCH --mem=8Gb
#SBATCH --account=def-marzyeh
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8

hostname
export LANG=C.UTF-8
source $HOME/.bashrc
python -u ../activmask/main.py train-skopt --config ${file} --seed=${SEED}
EOF
    sbatch ${runscript}
    rm ${runscript}
    done
done
