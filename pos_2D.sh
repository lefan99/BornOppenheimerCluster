#!/usr/bin/zsh

#SBATCH --job-name=ExcitonBornOppenheimer

#SBATCH --comment="Exciton ED in BornOppenheimer approximation, treating 1D and 0D confinement"

#SBATCH --account=rwth1610

#SBATCH --ntasks=16

###SBATCH --cpus-per-task=96

#SBATCH --nodes=1

#SBATCH --mem-per-cpu=3900M

#SBATCH --time=1:00:00

#SBATCH --error=err_pos2d.txt
#SBATCH --output=out_pos2D.txt

###Python system first and second arg give the X/Y COM coordinate by index in the BO_array, third arg gives the potential by index. 

ml load Python
python -u combine.py
