#!/bin/bash
#SBATCH --job-name=mae_cayley_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --output=log/mae_cayley_debug_%j.out
#SBATCH --error=log/mae_cayley_debug_%j.err

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
EXP_NAME="debug_finetune_cayley_${TIMESTAMP}"

# Path to pretrained checkpoint
PRETRAIN_CKPT="/ceph/hpc/data/b2025b05-014-users/haiping_env/point-mae-pretrain/experiments/pretrain_rope_cayley/cfgs/point_mae_rope_cayley_20250803_224302/ckpt-epoch-300.pth"

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found at $PRETRAIN_CKPT"
    exit 1
else
    echo "✓ Found pretrained checkpoint"
fi

# Create a debug config with fewer epochs
cat > cfgs/finetune/finetune_modelnet_cayley_debug.yaml << EOF
optimizer : {
  type: AdamW,
  kwargs: {
  lr : 0.0005,
  weight_decay : 0.05
}}

scheduler: {
  type: CosLR,
  kwargs: {
    epochs: 5,  # Only 5 epochs for debug
    initial_epochs : 1
}}

dataset : {
  train : { _base_: cfgs/dataset_configs/ModelNet40HDF5.yaml,
            others: {subset: 'train'}},
  val : { _base_: cfgs/dataset_configs/ModelNet40HDF5.yaml,
            others: {subset: 'test'}},
  test : { _base_: cfgs/dataset_configs/ModelNet40HDF5.yaml,
            others: {subset: 'test'}}}

model : {
  NAME: PointTransformer,
  trans_dim: 384,
  depth: 12,
  drop_path_rate: 0.1,
  cls_dim: 40,
  num_heads: 6,
  group_size: 32,
  num_group: 64,
  encoder_dims: 384,
}

npoints: 1024
total_bs : 16  # Smaller batch size for debug
step_per_update : 1
max_epoch : 5  # Only 5 epochs for debug
grad_norm_clip : 10
EOF

echo "=== Starting Debug Finetuning ==="
echo "Config: cfgs/finetune/finetune_modelnet_cayley_debug.yaml"
echo "Checkpoint: $PRETRAIN_CKPT"
echo "Experiment: $EXP_NAME"

# Run finetuning with debug settings
python main.py \
    --config cfgs/finetune/finetune_modelnet_cayley_debug.yaml \
    --finetune_model \
    --ckpts $PRETRAIN_CKPT \
    --exp_name $EXP_NAME \
    --launcher none \
    --num_workers 2

echo "=== Debug finetuning completed ==="