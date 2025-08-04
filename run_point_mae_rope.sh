#!/bin/bash
# Get RoPE type from command line argument
ROPE_TYPE=${1:-standard}
#SBATCH --job-name=mae_rope_${ROPE_TYPE}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/point_mae_rope_%j.out
#SBATCH --error=log/point_mae_rope_%j.err

# Usage: sbatch run_point_mae_rope_unified.sh [rope_type]
# rope_type options: standard, cayley, givens, householder
# Default: standard

# Get RoPE type from command line argument
ROPE_TYPE=${1:-standard}

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
echo "RoPE Type: $ROPE_TYPE"
python --version
nvcc --version
nvidia-smi

echo "=== Starting Point-MAE Pretraining with $ROPE_TYPE RoPE ==="

# Determine config file based on rope type
case "$ROPE_TYPE" in
    "standard")
        CONFIG_FILE="cfgs/pretrain_rope.yaml"
        ;;
    "cayley")
        CONFIG_FILE="cfgs/pretrain_rope_cayley.yaml"
        ;;
    "givens")
        CONFIG_FILE="cfgs/pretrain_rope_givens.yaml"
        ;;
    "householder")
        CONFIG_FILE="cfgs/pretrain_rope_householder.yaml"
        ;;
    *)
        echo "Error: Unknown RoPE type: $ROPE_TYPE"
        echo "Valid options: standard, cayley, givens, householder"
        exit 1
        ;;
esac

# Run Point-MAE pretraining
python main.py \
    --config $CONFIG_FILE \
    --exp_name point_mae_rope_${ROPE_TYPE}_$(date +%Y%m%d_%H%M%S) \
    --launcher none

echo "=== Training Completed ==="