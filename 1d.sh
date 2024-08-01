#!/usr/bin/zsh

#SBATCH --job-name=Exciton1D_BO

#SBATCH --comment="Exciton ED in BornOppenheimer approximation, treating 1D"

#SBATCH --account=rwth1610

#SBATCH --ntasks=1

###SBATCH --cpus-per-task=96

#SBATCH --nodes=1

#SBATCH --mem-per-cpu=3900M

#SBATCH --time=10:00:00

#SBATCH --array=90-125

#SBATCH --error=/home/kk472919/PhD/BO_parallel/output/Error_%A_%a.err

#SBATCH --output=/home/kk472919/PhD/BO_parallel/output/OUTPUT_%A_%a.txt


###Python system first and second arg give the X/Y COM coordinate by index in the BO_array, third arg gives the potential by index. 

ml load Python
python -u relative1D.py ${SLURM_ARRAY_TASK_ID} 
