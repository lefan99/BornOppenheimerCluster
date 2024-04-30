#!/usr/bin/zsh

#SBATCH --ntasks=96

#SBATCH --nodes=2

#SBATCH --mem-per-cpu=3900M

#SBATCH --time=00:10:00

#SBATCH --array=0-99

#SBATCH --output=/home/kk472919/PhD/BO_parallel/output/COM_calc_%x_%j.out

#SBATCH --error=/home/kk472919/PhD/BO_parallel/output/COM_calc_%x_%j.err

#SBATCH --output=/home/kk472919/PhD/BO_parallel/output/out_${SLURM_ARRAY_TASK_ID}.txt


###Python system first and second arg give the X/Y COM coordinate by index in the BO_array, third arg gives the potential by index. 

ml load Python
python solver.py ${SLURM_ARRAY_TASK_ID} 0 0 
