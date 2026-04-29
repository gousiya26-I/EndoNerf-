#!/bin/bash
#SBATCH --account=project_2017044
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1

# 🔴 Save logs to SCRATCH (not home)
#SBATCH --output=/scratch/project_2017044/logs/train_%j.out
#SBATCH --error=/scratch/project_2017044/logs/train_%j.err

module purge
module load pytorch

# 🔴 ===== CRITICAL FIXES =====

# 1. Redirect temporary directory (prevents /tmp overflow crash)
export TMPDIR=/scratch/project_2017044/tmp
mkdir -p $TMPDIR

# 2. Redirect cache (prevents home quota overflow)
export TORCH_HOME=/scratch/project_2017044/torch_cache
export XDG_CACHE_HOME=/scratch/project_2017044/.cache

# 3. Ensure directories exist
mkdir -p /scratch/project_2017044/logs
mkdir -p /scratch/project_2017044/torch_cache
mkdir -p /scratch/project_2017044/.cache

# ============================

# Go to project directory
cd /projappl/project_2017044/EndoNerf

# Run training
python3 -u run_endonerf.py \
    --config configs/example.txt \
    --N_iter 50000 \
    --i_weights 5000 \
    --i_img 1000000 \
    --no_depth_refine \
    --ft_path /scratch/project_2017044/logs/example_training/008000.tar