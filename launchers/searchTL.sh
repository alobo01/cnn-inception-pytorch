#!/bin/bash

###

#SBATCH --qos=train

###

#SBATCH --cpus-per-task=40

#SBATCH --gres gpu:1

#SBATCH --time=1-00:00:00

###

#SBATCH --job-name="searchTL"

#SBATCH --chdir=.

#SBATCH --output=searchTL_%j.out

#SBATCH --error=searchTL_%j.err

###

module purgue
module load  impi  intel  hdf5  mkl  python/3.12.1-gcc

time python train.py --config config/searchTL.yaml
