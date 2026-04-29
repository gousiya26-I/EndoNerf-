#!/bin/bash
#SBATCH --job-name=endo_render
#SBATCH --account=project_2017044
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=render_out.txt
#SBATCH --error=render_err.txt

module purge
module load pytorch

cd /projappl/project_2017044/EndoNerf

python3 -u run_endonerf.py \
    --config configs/example.txt \
    --render_only