#!/bin/bash
set -e

cd /root/GlueBind

echo "=== 1. 创建专业目录结构 ==="
mkdir -p gluebind/{models,data,utils}
mkdir -p scripts
mkdir -p configs
mkdir -p notebooks

echo "=== 2. 创建核心模块文件 ==="

# gluebind/__init__.py
cat > gluebind/__init__.py << 'EOF'
"""
GlueBind: A Tripartite Discriminator for Molecular Glue Ternary Complex Prediction

Author: [Team Name]
Date: 2026-03-27
"""

__version__ = "0.1.0"
__author__ = "[Team Name]"

from .models import GlueBindModel
from .data import TernaryDataset

__all__ = ["GlueBindModel", "TernaryDataset"]
EOF

# gluebind/models/__init__.py
cat > gluebind/models/__init__.py << 'EOF'
from .gluebind_model import GlueBindModel
from .attention import TripartiteCrossAttention

__all__ = ["GlueBindModel", "TripartiteCrossAttention"]
EOF

# gluebind/data/__init__.py
cat > gluebind/data/__init__.py << 'EOF'
from .dataset import TernaryDataset

__all__ = ["TernaryDataset"]
EOF

# gluebind/utils/__init__.py
cat > gluebind/utils/__init__.py << 'EOF'
from .metrics import compute_auroc, compute_auprc

__all__ = ["compute_auroc", "compute_auprc"]
EOF

echo "=== 3. 移动和重构文件 ==="

# 保留旧的 src 作为备份
mv src src_backup

# 移动预处理脚本
mkdir -p gluebind/data/preprocessing
mv data/preprocess.py gluebind/data/preprocessing/

# 创建配置文件
cat > configs/model_config.yaml << 'EOF'
# GlueBind Model Configuration

model:
  protein_dim: 1280        # ESM-2 embedding dimension
  mol_dim: 2048            # Morgan fingerprint dimension
  hidden_dim: 256          # Model hidden dimension
  dropout: 0.3             # Dropout probability
  n_heads: 8               # Attention heads

training:
  batch_size: 128
  learning_rate: 0.0001
  num_epochs: 100
  early_stopping_patience: 10
  pos_weight: 4            # For weighted BCE

data:
  esm_model: "facebook/esm2_t33_650M_UR50D"
  morgan_radius: 2
  morgan_nbits: 2048
  train_val_test_split: [0.7, 0.15, 0.15]
EOF

cat > configs/train_config.yaml << 'EOF'
# Training Configuration

experiment_name: "gluebind_v1"
seed: 42

device: "cuda"  # or "cpu"

logging:
  use_wandb: true
  project_name: "gluebind"
  log_interval: 10

checkpoint:
  save_dir: "checkpoints"
  save_best_only: true
  save_interval: 5
EOF

echo "=== 4. 创建训练脚本 ==="

cat > scripts/train.py << 'EOF'
#!/usr/bin/env python3
"""
Training script for GlueBind
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gluebind.models import GlueBindModel
from gluebind.data import TernaryDataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train GlueBind model')
    parser.add_argument('--model-config', default='configs/model_config.yaml')
    parser.add_argument('--train-config', default='configs/train_config.yaml')
    parser.add_argument('--data-path', default='dataset/processed_index.csv')
    args = parser.parse_args()
    
    # Load configs
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    
    print(f"Starting training: {train_config['experiment_name']}")
    print(f"Model config: {model_config}")
    
    # TODO: Implement training loop
    # 1. Load dataset
    # 2. Create model
    # 3. Train with cross-validation
    # 4. Save checkpoints


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/train.py

cat > scripts/preprocess.sh << 'EOF'
#!/bin/bash
# Preprocessing pipeline for GlueBind

set -e

echo "=== GlueBind Preprocessing Pipeline ==="

# Activate environment
conda activate GlueBind 2>/dev/null || source activate GlueBind

# Run preprocessing
echo "Step 1: Parsing and aligning data..."
python -c "
from gluebind.data.preprocessing.preprocess import parse_and_align_data
df = parse_and_align_data('dataset/mock_ternary_data.csv')
df.to_csv('dataset/aligned_data.csv', index=False)
print(f'Aligned {len(df)} samples')
"

echo "Step 2: Computing ESM-2 embeddings..."
python -c "
import torch
from gluebind.data.preprocessing.preprocess import compute_esm_embeddings
df = parse_and_align_data('dataset/mock_ternary_data.csv')
all_seqs = df['target_seq'].tolist() + df['ligase_seq'].tolist()
esm_dict = compute_esm_embeddings(all_seqs)
torch.save(esm_dict, 'dataset/esm_embeddings.pt')
print(f'Saved {len(esm_dict)} protein embeddings')
"

echo "Step 3: Computing Morgan fingerprints..."
python -c "
import torch
from gluebind.data.preprocessing.preprocess import compute_morgan_fps
df = parse_and_align_data('dataset/mock_ternary_data.csv')
fps_dict = compute_morgan_fps(df['smiles'].tolist())
torch.save(fps_dict, 'dataset/morgan_fps.pt')
print(f'Saved {len(fps_dict)} molecule fingerprints')
"

echo "Step 4: Generating CReM decoys..."
python -c "
import json
from gluebind.data.preprocessing.preprocess import generate_crem_decoys
df = parse_and_align_data('dataset/mock_ternary_data.csv')
decoys = generate_crem_decoys(df['smiles'].tolist())
with open('dataset/crem_decoys.json', 'w') as f:
    json.dump(decoys, f)
print(f'Generated {len(decoys)} decoys')
"

echo "=== Preprocessing complete ==="
EOF

chmod +x scripts/preprocess.sh

echo "=== 5. 更新 .gitignore ==="

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb

# Model checkpoints and data
*.pth
*.pt
*.ckpt
*.h5
*.pkl
*.pickle
checkpoints/
models/

# Logs
*.log
logs/
wandb/

# Environment variables
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Large files
data/*.db
data/*.db.gz
dataset/*.pt
dataset/*.json

# Backup
src_backup/
EOF

echo "=== 6. 创建 README 文档 ==="

c