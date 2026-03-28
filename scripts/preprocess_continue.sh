#!/bin/bash
# Continue Preprocessing pipeline for GlueBind (Step 4 only)
set -e

echo "=== GlueBind Preprocessing: Phase 2 ==="
echo "Checking required files..."
if [ ! -f "dataset/aligned_data.csv" ] || [ ! -f "dataset/crem_decoys.json" ]; then
    echo "Error: Missing required files in dataset/ directory."
    exit 1
fi
echo "Files found. Proceeding..."

echo "Step 4: Computing Morgan fingerprints (Originals + Decoys)..."
python -c "
import torch, json, pandas as pd
from gluebind.data.preprocessing.preprocess import compute_morgan_fps

print('Loading data...')
df = pd.read_csv('dataset/aligned_data.csv')
with open('dataset/crem_decoys.json', 'r') as f:
    decoys_dict = json.load(f)

print('Calculating fingerprints...')
fps_dict = compute_morgan_fps(df['smiles'].tolist(), decoys_dict=decoys_dict)

print('Saving to dataset/morgan_fps.pt...')
torch.save(fps_dict, 'dataset/morgan_fps.pt')
"

echo "=== Preprocessing completely finished! All assets are ready. ==="
