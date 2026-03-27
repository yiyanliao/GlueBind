import torch
from data.preprocess import (
    parse_and_align_data, 
    compute_morgan_fps, 
    compute_esm_embeddings, 
    generate_crem_decoys
)

# 1. 对齐数据
df = parse_and_align_data("dataset/mock_ternary_data.csv")

# 2. 提取特征并保存到本地 (避免每次训练都算)
all_seqs = df['target_seq'].tolist() + df['ligase_seq'].tolist()
esm_dict = compute_esm_embeddings(all_seqs, model_name="facebook/esm2_t33_650M_UR50D") 
torch.save(esm_dict, "dataset/esm_embeddings.pt")

fps_dict = compute_morgan_fps(df['smiles'].tolist())
torch.save(fps_dict, "dataset/morgan_fps.pt")

# 3. 生成 CReM 困难负样本并保存
decoys = generate_crem_decoys(df['smiles'].tolist(), db_path="data/chembl33_sa2_f5.db")
import json
with open("dataset/crem_decoys.json", "w") as f:
    json.dump(decoys, f)

