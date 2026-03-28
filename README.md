# GlueBind

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GlueBind** is a deep learning framework for molecular glue ternary binding prediction. Given a **(target protein, E3 ligase, small molecule)** triplet, the model estimates the probability of productive ternary complex formation and supports large-scale virtual screening.

> 🏆 Developed for the 48-Hour AI for Science Hackathon (Shanghai West Bund, March 2026)

---

## Overview

Molecular glue degraders (MGDs) are an important class of targeted protein degradation (TPD) agents that induce ternary complex formation between a target protein and an E3 ubiquitin ligase. **GlueBind** is formulated as a ternary binding probability predictor for (Target, E3 Ligase, Small Molecule) triplets and is intended for virtual screening of candidate molecular glues.

---

## Current Status

This repository contains an **early-stage research implementation developed during a hackathon and subsequently organized into a reproducible workflow**.

- The current repository supports preprocessing, dataset splitting, and model training.
- The scripts define the recommended execution path for the current pipeline.
- Users should refer to the script implementations for exact runtime behavior and file dependencies.

---

## Data Source and Input Format

The current pipeline assumes that upstream data integration has already been completed. In the present implementation, the training table is constructed from a curated merger of multiple ternary-complex-related resources assembled during data collection.

The integrated table currently combines records derived from four sources:

- **MGTbind**
- **MolGlueDB**
- **TPD-related database entries**
- **MG database entries**

Before model training, these sources are standardized into a unified schema, deduplicated at the table level, and filtered to remove problematic records such as duplicated rows and multi-protein / complex-style entries that are not suitable for the current triplet formulation.

The default input file expected by the preprocessing pipeline is:

```text
./dataset/zuixinban.csv
```

### Input CSV schema

```csv
id,source_db,protein_a_uniprot_id,protein_a_name,protein_a_full_name,protein_a_seq,protein_b_uniprot_id,protein_b_name,protein_b_full_name,protein_b_seq,target_protein,moa_type,molecule_name,canonical_smiles,label
```

### Field notes

- `protein_a_*`: typically the recruiter / E3 side
- `protein_b_*`: typically the target / substrate side
- `canonical_smiles`: molecule structure input
- `label`: binary supervision label used for training
- `source_db`: original database source retained for traceability after integration

---

## Environment Setup

### Prerequisites

- Python 3.10+
- Conda / Mamba recommended
- GPU is recommended for embedding generation and training, but some preprocessing steps can run on CPU

### Installation

```bash
# Clone repository
git clone https://github.com/yiyanliao/GlueBind.git
cd GlueBind

# Create environment from yaml
mamba env create -f environment.yaml
conda activate GlueBind

# Install additional Python dependencies
pip install -r requirements_new.txt
```

---

## Recommended Workflow

The recommended usage is script-based and follows a four-step pipeline:

### 1. Prepare the integrated input table
Place the curated training table at:

```text
./dataset/zuixinban.csv
```

### 2. Run preprocessing
```bash
bash scripts/preprocess.sh
```
This step prepares the processed table, protein embeddings, molecular fingerprints, and decoy-related assets used downstream.

### 3. Split the dataset
```bash
python scripts/split_dataset.py
```
This step generates the train / validation / test splits under sequence-similarity-aware and chemistry-aware constraints.

### 4. Train the model
```bash
python scripts/train.py \
    --model-config configs/model_config.yaml \
    --train-config configs/train_config.yaml
```

---

## Known Limitations

- `main.py` reflects an older entrypoint style and is **not** the recommended way to run the current full pipeline.
- The current dataset is limited and may contain residual noise from upstream integration.
- Some scripts assume access to NVIDIA GPUs and a compatible CUDA environment.
- Documentation is being aligned with the current codebase and workflow.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Last Updated: 2026-03-28*
