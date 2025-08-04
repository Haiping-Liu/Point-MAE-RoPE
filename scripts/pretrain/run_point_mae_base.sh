#!/bin/bash
#SBATCH --job-name=mae_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/mae_base_%j.out
#SBATCH --error=log/mae_base_%j.err

# Load required modules
module load GCC/11.2.0
module load CUDA/11.7.0

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate point-mae

# Change to project directory
cd /ceph/hpc/home/liuh/code/Point-MAE

# Create log directory if it doesn't exist
mkdir -p log

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0

# Print environment info
echo "=== Environment Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
python --version
nvcc --version
nvidia-smi

echo "=== Starting Point-MAE Base Model Pretraining ==="
echo "Config: trans_dim=432, num_heads=6, head_dim=72"

# Run Point-MAE base model pretraining
python main.py \
    --config cfgs/pretrain_base.yaml \
    --exp_name mae_base_$(date +%Y%m%d_%H%M%S) \
    --launcher none

echo "=== Training Completed ==="