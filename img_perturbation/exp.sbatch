#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=eps10
#SBATCH --output=eps10.out
#SBATCH --gres=gpu:1


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate memes
python attack.py