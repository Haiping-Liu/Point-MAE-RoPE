#!/bin/bash
#SBATCH --job-name=mae_cayley_fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=log/mae_cayley_fast_%j.out
#SBATCH --error=log/mae_cayley_fast_%j.err

# Load modules
module load CUDA/11.7.0

# Activate conda environment FIRST before changing directory
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate point-mae

# Change to project directory
cd /ceph/hpc/home/liuh/code/Point-MAE

# Environment info
echo "=== Environment Info ==="
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Ensure log directory exists
mkdir -p log

# Experiment name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="finetune_cayley_fast_${TIMESTAMP}"

# Path to pretrained checkpoint
PRETRAIN_CKPT="/ceph/hpc/data/b2025b05-014-users/haiping_env/point-mae-pretrain/experiments/pretrain_rope_cayley/cfgs/point_mae_rope_cayley_20250803_224302/ckpt-epoch-300.pth"

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found at $PRETRAIN_CKPT"
    exit 1
else
    echo "✓ Found pretrained checkpoint"
fi

echo "=== Starting Fine-tuning with Fast Dataset ==="
echo "Config: cfgs/finetune/finetune_modelnet_cayley_fast.yaml"
echo "Checkpoint: $PRETRAIN_CKPT"
echo "Experiment: $EXP_NAME"

# Run finetuning
python main.py \
    --config cfgs/finetune/finetune_modelnet_cayley_fast.yaml \
    --finetune_model \
    --ckpts $PRETRAIN_CKPT \
    --exp_name $EXP_NAME \
    --launcher none \
    --num_workers 4

echo "=== Fine-tuning completed ==="