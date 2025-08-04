# Scripts Organization

This directory contains all the scripts for running Point-MAE experiments.

## Directory Structure

```
scripts/
├── pretrain/           # Pretraining scripts
│   ├── run_point_mae_pretrain.sh      # Standard Point-MAE pretraining
│   ├── run_point_mae_aligned.sh       # Aligned version pretraining
│   ├── run_point_mae_rope.sh          # RoPE version pretraining
│   ├── run_rope_standard.sh           # Standard RoPE pretraining
│   ├── run_rope_cayley.sh             # Cayley RoPE pretraining
│   ├── run_rope_givens.sh             # Givens RoPE pretraining
│   └── run_rope_householder.sh        # Householder RoPE pretraining
│
└── finetune/           # Finetuning scripts
    ├── run_point_mae_finetune.sh      # Standard Point-MAE finetuning
    └── run_finetune_cayley.sh         # Cayley RoPE finetuning
```

## Usage

### Pretraining
```bash
cd /ceph/hpc/home/liuh/code/Point-MAE
sbatch scripts/pretrain/run_rope_cayley.sh
```

### Finetuning
```bash
cd /ceph/hpc/home/liuh/code/Point-MAE
sbatch scripts/finetune/run_finetune_cayley.sh
```

## Configuration Files

The configuration files have also been reorganized:
- Pretrain configs: `cfgs/pretrain*.yaml`
- Finetune configs: `cfgs/finetune/*.yaml`
- Dataset configs: `cfgs/dataset_configs/*.yaml`