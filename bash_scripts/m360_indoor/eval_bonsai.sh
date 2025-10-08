#!/bin/bash
#SBATCH --output=logs/eval_bonsai.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G # Memory to allocate in MB per allocated CPU core
#SBATCH --time="0-00:50:00" # Max execution time
#SBATCH --account=telim


micromamba activate ts-dev

python train.py \
-s /gpfs/scratch/acad/telim/datasets/MipNeRF360/bonsai  \
-i images_2 \
-m models/$1/bonsai \
--quiet \
--eval \
--indoor

python render.py --iteration 30000 -s /gpfs/scratch/acad/telim/datasets/MipNeRF360/bonsai -m models/$1/bonsai --eval --skip_train --quiet

python metrics.py -m models/$1/bonsai