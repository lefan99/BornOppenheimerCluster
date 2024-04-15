#!/usr/bin/zsh

#SBATCH --ntasks=96

#SBATCH --nodes=2

#SBATCH --mem-per-cpu=3900M

#SBATCH --time=00:10:00

#SBATCH --output=output/COM_calc_${SLURM_ARRAY_TASK_ID}_%j.out
#
#SBATCH --error=output/COM_calc_${SLURM_ARRAY_TASK_ID}_%j.err

#SBATCH --array=0-99

#SBATCH --output=output/out_${SLURM_ARRAY_TASK_ID}.txt
#
#

ml load Python

python solver.py ${SLURM_ARRAY_TASK_ID} 0 
