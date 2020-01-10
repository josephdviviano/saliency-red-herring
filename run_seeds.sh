#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="synth livermsd cardiacmsd pancreasmsd"
SEEDS=(1234 3232 3221 9856 1290)
#SEEDS=(1987 3200 6400 8888 0451)

for seed in "${SEEDS[@]}"; do
    echo "SEED ${seed}:"

    # Only generate visualizations for the first seed.
    if [ ${seed} -eq 1234 ]; then
        viz="True"
    else
        viz="False"
    fi

    for exp in ${EXPERIMENTS}; do
        for file in $(ls config/${exp}/${exp}_*); do

            filename=$(basename ${file})
            filename="${filename%.*}"
            filename="${filename}_${seed}"
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
python -u main.py train --config ${file} --seed=${seed}
EOF

        # Only run jobs that don't already have an output log.
        if [ ! -f cluster_logs/${filename}_out.txt ]; then
            sbatch ${runscript}
        fi

        rm ${runscript}
        done
    done
done
