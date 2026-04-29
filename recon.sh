#!/bin/bash
#SBATCH --job-name=endo_recon
#SBATCH --account=project_2017044
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

module purge
module load pytorch

cd /projappl/project_2017044/EndoNerf

python3 endo_pc_reconstruction.py \
  --config_file configs/example.txt \
  --n_frames 156 \
  --depth_smoother \
  --depth_smoother_d 28
