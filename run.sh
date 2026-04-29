#!/bin/bash
#SBATCH --job-name=endonerf
#SBATCH --account=project_2017044
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

echo "Starting job"
nvidia-smi

module purge
module load pytorch
module load imagemagick

echo "Python path:"
which python

echo "Running training..."

srun python run_endonerf.py \
    --datadir /projappl/project_2017044/EndoNerf/Data/data1 \
    --dataset_type llff \
    --use_depth \
    --use_fgmask \
    --no_batching \
    --factor 1 \
    --N_iter 50000 \
    --i_weights 5000 \
    --expname endonerf_run
