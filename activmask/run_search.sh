#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="cardiacmsd-search livermsd-search pancreasmsd-search"
SEEDS=(1234)
LR=(0.01 0.001 0.0001)

for seed in "${SEEDS[@]}"; do
    echo "SEED ${seed}:"

    for lr in "${LR[@]}"; do
    echo "LR ${lr}:"

    for exp in ${EXPERIMENTS}; do
        for file in $(ls config/${exp}/${exp}_*); do

            filename=$(basename ${file})
            filename="${filename%.*}"
            filename="${filename}_${seed}_${lr}"
            runscript="${filename}.pbs"

            # Generates a job script.
            cat << EOF > ${runscript}
#!/bin/bash
#SBATCH --job-name=${filename}
#SBATCH --output=cluster_logs/${filename}_out.txt
#SBATCH --error=cluster_logs/${filename}_err.txt
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:titanxp:1

hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask
python -u main.py train --config ${file} -seed=${seed} -viz=${viz} -lr=${lr}
EOF

        # Only run jobs that don't already have an output log.
        if [ ! -f cluster_logs/${filename}_out.txt ]; then
            echo "submitting ${filename}"
            sbatch ${runscript}
        fi

        rm ${runscript}
        done
    done
    done
done
