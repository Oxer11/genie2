#!/bin/bash

#SBATCH --partition=unkillable,long,main                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o slurm-%j.out  # Write the log on scratch

conda init
conda activate
conda activate fmas_env

cp -r data/$1/ /tmp/ 
python -u genie/inference.py --config runs/$1/configuration
