#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="xray-seeds rsna-seeds synth-seeds"
SEEDS=(1234 3232 3221 9856 1290 1987 1111 6400 8888 0451)

for exp in ${EXPERIMENTS}; do
    for file in $(ls activmask/config/${exp}/${exp}*); do
        for seed in "${SEEDS[@]}"; do

            filename=$(basename ${file})
            filename="${filename%.*}"
            filename="${filename}_${seed}"

            # Generates a job script.
            runscript="${filename}.pbs"
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
source activate activmask
python -u ../activmask/main.py train --config ${file} --seed=${seed}
EOF

        sbatch ${runscript}
        rm ${runscript}

        done
    done
done
