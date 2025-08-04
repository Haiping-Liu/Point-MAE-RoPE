#!/bin/bash
#SBATCH --job-name=mae_cayley_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=log/mae_cayley_finetune_%j.out
#SBATCH --error=log/mae_cayley_finetune_%j.err

module load CUDA/11.7.0
module load Python/3.9.6-GCCcore-11.2.0

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate point-mae

# Ensure the experiment name includes the timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="finetune_modelnet_cayley_${TIMESTAMP}"

# Path to pretrained checkpoint
PRETRAIN_CKPT="/ceph/hpc/data/b2025b05-014-users/haiping_env/point-mae-pretrain/experiments/pretrain_rope_cayley/cfgs/point_mae_rope_cayley_20250803_224302/ckpt-epoch-300.pth"

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found at $PRETRAIN_CKPT"
    echo "Please run pretraining first or update the path"
    exit 1
fi

# Run finetuning
python main.py \
    --config cfgs/finetune/finetune_modelnet_cayley.yaml \
    --finetune_model \
    --ckpts $PRETRAIN_CKPT \
    --exp_name $EXP_NAME \
    --launcher pytorch

echo "Finetuning job started with experiment name: $EXP_NAME"