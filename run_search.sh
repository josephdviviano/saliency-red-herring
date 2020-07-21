#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="synth-search livermsd-search cardiacmsd-search pancreasmsd-search" #xray-search"
SEED=1234

for exp in ${EXPERIMENTS}; do
    for file in $(ls activmask/config/${exp}/${exp}_*); do

        filename=$(basename ${file})
        filename="${filename%.*}"
        filename="${filename}_${seed}"
        runscript="${filename}.pbs"

        # Generates a job script.
        cat << EOF > ${runscript}
#!/bin/bash
#!/bin/bash
#SBATCH --job-name=${filename}
#SBATCH --output=cluster_logs/${filename}_out.txt
#SBATCH --error=cluster_logs/${filename}_err.txt
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=8Gb
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1

hostname
export LANG=C.UTF-8
source $HOME/.bashrc
python -u activmask/main.py train-skopt --config ${file} --seed=${SEED}
EOF

    bash ${runscript}
    rm ${runscript}
    done
done
