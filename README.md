# GlueBind

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GlueBind** is a deep learning discriminator for large-scale molecular glue ternary complex prediction via latent space feature fusion.

> 🏆 Developed for the 48-Hour AI for Science Hackathon (Shanghai West Bund, March 2026)

---

## 🎯 Overview

Molecular glue degraders (MGDs) represent a powerful class of targeted protein degradation (TPD) agents that induce ternary complex formation between a target protein and an E3 ubiquitin ligase. **GlueBind** predicts the ternary binding probability of a (Target, E3 Ligase, Small Molecule) triplet, enabling large-scale virtual screening for novel molecular glues.

### Key Features

- 🧬 **Tripartite Cross-Attention**: Novel architecture modeling three-body cooperative interactions
- 🔬 **ESM-2 + Morgan Fingerprints**: State-of-the-art protein and molecule representations
- 🎯 **Stratified Negative Sampling**: Three-tier difficulty (Easy/Medium/Hard) for robust training
- ⚡ **GPU-Accelerated**: Optimized for NVIDIA GPUs on Bohrium platform
- 📊 **End-to-End Pipeline**: From data preprocessing to virtual screening

---

## 🏗️ Architecture

```
Input: (Target Protein, E3 Ligase, Small Molecule)
       ↓
ESM-2 (1280-dim) + Morgan FP (2048-dim)
       ↓
Feature Projection (256-dim)
       ↓
Tripartite Cross-Attention
  ├─ Query: Small Molecule
  └─ Key/Value: Target + E3 Ligase
       ↓
Element-wise Interactions
  ├─ Target ⊙ Molecule
  └─ E3 ⊙ Molecule
       ↓
MLP Classifier → Probability [0, 1]
```

### Model Details

| Component | Specification |
|-----------|---------------|
| Protein Encoder | ESM-2 (650M parameters) |
| Molecule Encoder | Morgan Fingerprint (ECFP4, radius=2) |
| Hidden Dimension | 256 |
| Attention Heads | 8 |
| Dropout | 0.3 |
| Output | Sigmoid (binary classification) |

---

## 📁 Project Structure

```
GlueBind/
├── gluebind/              # Main package
│   ├── models/            # Model architectures
│   ├── data/              # Data processing
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Executable scripts
├── dataset/               # Dataset directory
├── notebooks/             # Jupyter notebooks
└── docs/                  # Documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA T4 or better)
- Conda or virtual environment

### Installation

```bash
# Clone repository
git clone https://github.com/yiyanliao/GlueBind.git
cd GlueBind

# Create conda environment
conda create -n GlueBind python=3.10
conda activate GlueBind

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing

```bash
bash scripts/preprocess.sh
```

### Training

```bash
python scripts/train.py \
    --model-config configs/model_config.yaml \
    --train-config configs/train_config.yaml
```

---

## 📊 Data Format

### Input CSV Schema

```csv
id,source_db,protein_a_uniprot_id,protein_a_name,protein_a_full_name,protein_a_seq,
protein_b_uniprot_id,protein_b_name,protein_b_full_name,protein_b_seq,
target_protein,moa_type,molecule_name,canonical_smiles,label
```

---

## 🔬 Key References

1. **DeepTernary** (Xue et al., 2025) - *Nature Communications*
2. **Rui et al.** (2023) - *RSC Chemical Biology*
3. **Liao et al.** (2025) - *JCIM*

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Last Updated: 2026-03-27*
