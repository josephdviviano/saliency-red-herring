#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="xray-seeds" #"livermsd-seeds synth-seeds colonmsd-seeds cardiacmsd-seeds pancreasmsd-seeds" # xray-seeds"
SEEDS=(1234 3232 3221 9856 1290 1987 1111 6400 8888 0451)  # (3232 3221 9856 1290 1987)

for exp in ${EXPERIMENTS}; do
    for file in $(ls activmask/config/${exp}/${exp}*bal_actdiff*); do
        for seed in "${SEEDS[@]}"; do

            filename=$(basename ${file})
            filename="${filename%.*}"
            filename="${filename}_${seed}"

            # Generates a job script.
            runscript="${filename}.pbs"
            cat << EOF > ${runscript}
#!/bin/bash
#SBATCH --job-name=${filename}
#SBATCH --output=cluster_logs/${filename}_out.txt
#SBATCH --error=cluster_logs/${filename}_err.txt
#SBATCH --time=12:00:00
#SBATCH --mem=8Gb
#SBATCH --account=def-marzyeh
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8

hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask
python -u activmask/main.py train --config ${file} --seed=${seed}
EOF

        bash ${runscript}
        rm ${runscript}

        done
    done
done
