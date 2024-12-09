#!/bin/bash
#SBATCH -p general
#SBATCH -t 2-00:00:00
#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH --mail-type=ALL
#SBATCH --mail-user=phegde7@asu.edu
#SBATCH -o logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate /scratch/phegde7/.conda/envs/vq

python3 evaluate.py --log-file=logs/default_vae_evaluate.log --dataset-dir=dataset/ --model-path=models/vae_default/epoch_1
