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
