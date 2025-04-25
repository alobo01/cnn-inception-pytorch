#!/bin/bash

###

#SBATCH --qos=train

###

#SBATCH --cpus-per-task=40

#SBATCH --gres gpu:1

#SBATCH --time=1-00:00:00

###

#SBATCH --job-name="test"

#SBATCH --chdir=.

#SBATCH --output=test_%j.out

#SBATCH --error=test_%j.err

###

module purgue
module load  impi  intel  hdf5  mkl  python/3.12.1-gcc

# # Search–StandardCNN
# python test.py --config config/searchS.yaml   --weights searchS_StandardCNN_mame.pth --output results/standard

# # Search–InceptionNet
# python test.py --config config/searchNS.yaml   --weights search_InceptionNet_mame.pth     --output results/search_InceptionNet_mame

# # Search–InceptionNetV3
# python test.py --config config/searchNS2.yaml   --weights train_InceptionNetV3_mame.pth     --output results/search_InceptionNetV3_mame


# Search–TransferLearning 
python test.py --config config/searchTL.yaml  --weights searchTL12_TransferLearningCNN_mame.pth  --output results/searchTL_TransferLearningCNN_mame


