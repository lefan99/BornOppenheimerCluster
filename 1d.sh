#!/usr/bin/zsh

#SBATCH --job-name=Exciton1D_BO

#SBATCH --comment="1D exciton wannier, radius fix"

#SBATCH --account=rwth1610

#SBATCH --ntasks=1

###SBATCH --cpus-per-task=96

#SBATCH --nodes=1

#SBATCH --mem-per-cpu=2600M

#SBATCH --time=48:00:00

#SBATCH --array=0-199

#SBATCH --error=/home/kk472919/PhD/BO_parallel/output/Error_%A_%a.err

#SBATCH --output=/home/kk472919/PhD/BO_parallel/output/OUTPUT_%A_%a.txt

###Python system first and second arg give the X/Y COM coordinate by index in the BO_array, third arg gives the potential by index. 

ml load Python
python -u relative1D_potloop.py ${SLURM_ARRAY_TASK_ID} 
