# GlueBind - Agent Memory

## Project Overview

**GlueBind** is a deep learning discriminator for large-scale molecular glue ternary complex prediction via latent space feature fusion.

### Core Mission
Predict the ternary binding probability of a (Target, E3 Ligase, Small Molecule) triplet for molecular glue degraders (MGDs).

### Key Differentiation
- **GlueBind**: Discriminative model (probability [0,1]) for large-scale virtual screening
- **DeepTernary**: Structure prediction (complementary approach)
- Workflow: GlueBind screening → DeepTernary validation

## Technical Architecture

### Input Features
1. **Protein Embeddings**: ESM-2 (650M) → 1280-dim, mean pooled
2. **Small Molecule**: Morgan fingerprint (ECFP4, radius=2) → 2048-bit

### Model Architecture
- Tripartite Cross-Attention network
  - Small molecule as Query
  - Proteins as Key/Value
- MLP with Dropout (p=0.3)
- Sigmoid output

### Training
- Loss: Weighted Binary Cross-Entropy (pos_weight=4)
- Optimizer: Adam (lr=1e-4)
- Batch size: 128
- Early stopping: AUROC, patience=10

## Data Pipeline

### Raw Data Format (dataset/mock_ternary_data.csv)
```
id,source_db,protein_a_uniprot_id,protein_a_name,protein_a_full_name,protein_a_seq,
protein_b_uniprot_id,protein_b_name,protein_b_full_name,protein_b_seq,
target_protein,moa_type,molecule_name,canonical_smiles,label
```

**Note**: Dataset contains **only positive samples** (label=1). Negative samples are generated during preprocessing.

### Preprocessing Pipeline (data/preprocess.py)

#### Step 1: Role Alignment
- Align Target and E3 Ligase based on `target_protein` field
- Extract sequences for each role

#### Step 2: Negative Sample Generation
| Type | Method | Implementation |
|------|--------|----------------|
| **Hard Negative** | CReM Decoy | `generate_crem_decoy()` - Single-point mutation with Tanimoto similarity 0.7-0.9 |

**CReM Database**: `replacore10.db` (from http://www.qsar4u.com/files/cremdb/)

#### Step 3: Feature Extraction
- **ESM-2 Embeddings**: `esm2_t33_650M_UR50D` → 1280-dim
  - Mean pooling over residues (excluding <cls> and <eos>)
  - Saved to: `protein_embeddings.pt`
- **Morgan Fingerprints**: Radius=2, nBits=2048
  - Saved to: `molecule_fps.pt`

#### Step 4: Output Index
- **File**: `processed_index.csv`
- **Columns**: `target_seq`, `ligase_seq`, `smiles`, `label`, `type`
  - `type`: "real" (positive) or "crem_decoy" (negative)

### Data Sources

#### Positive Samples (Label=1)
| Database | Size | Description |
|----------|------|-------------|
| MolGlueDB | ~1,840 | Curated MGD entries |
| MGTbind | - | Ternary interactome + affinity |
| MGDB | - | SAR records |
| TPDdb | - | Broader TPD landscape |

**Current mock data**: 10 positive samples

### Negative Sampling Strategy (Label=0)
| Type | Ratio | Method | Status |
|------|-------|--------|--------|
| Easy | 20% | Random permutation of triplets | Planned |
| Medium | 40% | Tanimoto decoy (0.7 ≤ Tc ≤ 0.9) | **Implemented (CReM)** |
| Hard | 40% | Homologous protein swap (cos ≥ 0.85) | Planned |

**Note**: Current preprocessing implements Hard negatives via CReM. Easy/Medium to be added.

## File Structure

```
GlueBind/
├── AGENT.md                    # This file
├── README.md                   # Project documentation
├── dataset/
│   └── mock_ternary_data.csv   # Raw data (positive only)
├── data/
│   ├── preprocess.py           # Preprocessing pipeline
│   ├── replacore10.db          # CReM database (required)
│   ├── processed_index.csv     # Output: processed data index
│   ├── protein_embeddings.pt   # Output: ESM-2 embeddings
│   └── molecule_fps.pt         # Output: Morgan fingerprints
├── src/                        # Model implementation
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Dependencies

```python
# Core
pandas, torch, tqdm

# ESM-2
fair-esm

# Chemistry
rdkit, crem  # CReM for decoy generation
```

## Development Environment

### GPU Node
- **Host**: fsms1440533.bohrium.tech
- **GPU**: NVIDIA T4 (15GB VRAM)
- **CUDA**: 12.2
- **Driver**: 535.274.02
- **Python**: 3.10

### Repository
- **GitHub**: https://github.com/yiyanliao/GlueBind
- **Local**: /root/GlueBind/

## 48-Hour Timeline

| Hours | Task | Agent | Status |
|-------|------|-------|--------|
| 0-6 | Data collection & preprocessing | OpenClaw | **In Progress** |
| 6-12 | ESM-2 inference + Morgan FP | BohrClaw | Pending |
| 12-24 | Model training (2 folds) | BohrClaw + OpenClaw | Pending |
| 24-36 | 5-fold CV + ablation | BohrClaw | Pending |
| 36-42 | Virtual screening (ZINC-22) | BohrClaw | Pending |
| 42-46 | Molecular docking (top-50) | BohrClaw | Pending |
| 46-48 | Paper writing | SciMaster | Pending |

## Key References

1. **DeepTernary** (Xue et al., 2025) - Nature Communications
   - SE(3)-equivariant GNN for ternary structure
   - TernaryDB: 22,303 complexes
   - DockQ = 0.65 on PROTAC benchmark

2. **Rui et al.** (2023) - RSC Chemical Biology
   - MGD classification: Group 1 (domain-domain) vs Group 2 (motif-domain)

3. **Liao et al.** (2025) - JCIM
   - Benchmarking AlphaFold3/Boltz-2 on MGD

4. **CReM** (https://github.com/DrrDom/crem)
   - Fragment-based molecule mutation for decoy generation

## Critical Insights

### BSA-Degradation Correlation
- BSA range 1100-1500 Å² correlates with high degradation potential
- Can be used for virtual screening validation

### MGD Classification
- **Group 1**: Domain-domain (harder to predict)
- **Group 2**: Motif-domain (easier to predict)

### CReM Decoy Generation
- Single-point mutation using fragment database
- Tanimoto similarity control: 0.7-0.9
- Generates chemically plausible but inactive analogues

## Expected Metrics

- **AUROC** ≥ 0.85 (primary)
- **AUPRC** ≥ 0.70
- **EF** @ top-1% vs random

## Current Status

- [x] Literature survey completed
- [x] GPU node provisioned
- [x] Mock dataset created (positive only)
- [x] Preprocessing pipeline implemented
  - [x] Role alignment
  - [x] CReM decoy generation (hard negatives)
  - [x] ESM-2 embedding extraction
  - [x] Morgan fingerprint computation
- [ ] Easy/Medium negative sampling
- [ ] Model implementation
- [ ] Training & evaluation
- [ ] Virtual screening
- [ ] Docking validation

## Notes

- ESM-2 650M parameters → 1280-dim embeddings
- Latent dimension: 256
- Morgan fingerprint: 2048-bit, radius=2
- Use RDKit for SMILES canonicalization
- UniProt IDs for protein standardization
- CReM database required for decoy generation

## Recent Changes

### 2026-03-27
- **Dataset**: Removed negative samples from mock data (now positive only)
- **Preprocessing**: Implemented CReM-based decoy generation
- **Pipeline**: Complete preprocessing flow with ESM-2 + Morgan FP

---
Last Updated: 2026-03-27
