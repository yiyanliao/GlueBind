#!/bin/bash
# Preprocessing pipeline for GlueBind (Pre-compute Mode)
set -e

echo "=== GlueBind Preprocessing Pipeline ==="

# 确保输出目录存在
mkdir -p dataset

echo "Step 1: Parsing, aligning and SWAP augmenting data..."
python -c "
from gluebind.data.preprocessing.preprocess import parse_and_align_data
df = parse_and_align_data('dataset/zuixinban.csv', augment_swap=True)
df.to_csv('dataset/aligned_data.csv', index=False)
"

echo "Step 2: Computing ESM-2 embeddings for all proteins..."
python -c "
import torch, pandas as pd
from gluebind.data.preprocessing.preprocess import compute_esm_embeddings
df = pd.read_csv('dataset/aligned_data.csv')
all_seqs = df['protein_1_seq'].tolist() + df['protein_2_seq'].tolist()
esm_dict = compute_esm_embeddings(all_seqs)
torch.save(esm_dict, 'dataset/esm_embeddings.pt')
"

echo "Step 3: Generating CReM decoys..."
python -c "
import json, pandas as pd
from gluebind.data.preprocessing.preprocess import generate_crem_decoys
df = pd.read_csv('dataset/aligned_data.csv')
decoys = generate_crem_decoys(df['smiles'].tolist(), db_path='dataset/chembl33_sa2_f5.db')
with open('dataset/crem_decoys.json', 'w') as f:
    json.dump(decoys, f)
"

echo "Step 4: Computing Morgan fingerprints (Originals + Decoys)..."
python -c "
import torch, json, pandas as pd
from gluebind.data.preprocessing.preprocess import compute_morgan_fps
df = pd.read_csv('dataset/aligned_data.csv')
with open('dataset/crem_decoys.json', 'r') as f:
    decoys_dict = json.load(f)
fps_dict = compute_morgan_fps(df['smiles'].tolist(), decoys_dict=decoys_dict)
torch.save(fps_dict, 'dataset/morgan_fps.pt')
"

echo "=== Preprocessing complete! All assets are in dataset/ ==="
