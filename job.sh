#!/usr/bin/zsh

#SBATCH --ntasks=96

#SBATCH --nodes=2

#SBATCH --mem-per-cpu=3900M

#SBATCH --time=1-00:00:00

#SBATCH --array=1-100

#SBATCH --output=output/out_${SLURM_ARRAY_TASK_ID}.txt
#
#

ml load Python

python solver.py ${SLURM_ARRAY_TASK_ID} 0 
