#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=prior
#SBATCH --partition=g80
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --account=account_name
#SBATCH --output=%x_%j.out

eval "$(conda shell.bash hook)"
conda activate torch 

module load cuda/11.8

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export OMPI_MCA_mtl_base_verbose=1
export FI_PROVIDER=efa
export NCCL_TREE_THRESHOLD=0

srun python train.py --config_path=configs/prior_big_g.yaml --num_workers=6 --num_nodes=32 --devices=8

