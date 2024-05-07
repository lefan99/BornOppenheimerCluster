#!/usr/bin/zsh

#SBATCH --job-name=ExcitonBornOppenheimer

#SBATCH --comment="Exciton ED in BornOppenheimer approximation, treating 1D and 0D confinement"

#SBATCH --account=rwth1610

#SBATCH --ntasks=1

###SBATCH --cpus-per-task=96

#SBATCH --nodes=1

#SBATCH --mem-per-cpu=3900M

#SBATCH --time=05:00:00

#SBATCH --array=0-99

#SBATCH --output=/home/kk472919/PhD/BO_parallel/output/COM_calc_%A_%a.out

#SBATCH --error=/home/kk472919/PhD/BO_parallel/output/COM_calc_%A_%a.err

#SBATCH --output=/home/kk472919/PhD/BO_parallel/output/out_%A_%a.txt


###Python system first and second arg give the X/Y COM coordinate by index in the BO_array, third arg gives the potential by index. 

ml load Python
python -u missing_solver.py ${SLURM_ARRAY_TASK_ID}
