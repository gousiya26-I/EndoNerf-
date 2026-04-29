#!/bin/bash
#SBATCH --job-name=endo_eval
#SBATCH --account=project_2017044
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=eval_out.txt
#SBATCH --error=eval_err.txt

module purge
module load pytorch

cd /projappl/project_2017044/EndoNerf

python eval_rgb.py \
  --gt_dir /projappl/project_2017044/EndoNerf/Data/data1/images \
  --mask_dir /projappl/project_2017044/EndoNerf/Data/data1/gt_masks \
  --img_dir /scratch/project_2017044/logs/example_training/renderonly_path_fixidentity_050000/estim