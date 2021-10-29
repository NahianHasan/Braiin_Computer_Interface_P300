#!/bin/bash
# FILENAME:  cluster_run

#SBATCH -A standby

#SBATCH --array=[1-4,6-9]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --export=ALL
#SBATCH --job-name 'P300'
#SBATCH --output='./slurm_output/slurm-%A_%a.out'
#SBATCH --error='./slurm_output/slurm-%A_%a.out'

echo "I ran on:" "${SLURM_ARRAY_TASK_ID}"

python3 Main.py -s ${SLURM_ARRAY_TASK_ID}
