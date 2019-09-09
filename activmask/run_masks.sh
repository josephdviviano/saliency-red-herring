#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="synth-masks livermsd-masks cardiacmsd-masks pancreasmsd-masks"
SEEDS=(1234 3232 3221 9856 1290 1987 3200 6400 8888 0451)
MASKS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for seed in "${SEEDS[@]}"; do
    for mask in "${MASKS[@]}"; do

        echo "SEED: ${seed} MASK ${mask}:"

        for exp in ${EXPERIMENTS}; do
            for file in $(ls config/${exp}/${exp}_*); do

                filename=$(basename ${file})
                filename="${filename%.*}"
                filename="${filename}_${seed}_${mask}"
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
python -u main.py train --config ${file} -seed=${seed} -maxmasks_train=${mask}
EOF

            # Only run jobs that don't already have an output log.
            if [ ! -f cluster_logs/${filename}_out.txt ]; then
                sbatch ${runscript}
                sleep 1
            fi

            rm ${runscript}
            done
        done
    done
done
