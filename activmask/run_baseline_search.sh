#!/bin/bash

mkdir cluster_logs

EXPERIMENTS="livermsd-search cardiacmsd-search pancreasmsd-search"
SEED=1234

for exp in ${EXPERIMENTS}; do
    for file in $(ls config/${exp}/*_baseline-search.yml); do

        filename=$(basename ${file})
        filename="${filename%.*}"
        filename="${filename}_${SEED}"
        runscript="${filename}.pbs"

        # Generates a job script.
        cat << EOF > ${runscript}
#!/bin/bash
#SBATCH --job-name=${filename}
#SBATCH --output=cluster_logs/${filename}_out.txt
#SBATCH --error=cluster_logs/${filename}_err.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:titanx:1


hostname
export LANG=C.UTF-8
source $HOME/.bashrc
source activate activmask
python -u main.py train-skopt --config ${file} -seed=${SEED} --n_iter=10
EOF

    # Only run jobs that don't already have an output log.
    if [ ! -f cluster_logs/${filename}_out.txt ]; then
        sbatch ${runscript} --account=rpp-bengioy
    fi
    rm ${runscript}
    done
done

