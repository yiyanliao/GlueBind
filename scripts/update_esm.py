import os
import torch
import pandas as pd
import logging
from transformers import AutoTokenizer, EsmModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_esm_only(csv_path: str = "dataset/zuixinban.csv", out_path: str = "dataset/esm_embeddings.pt"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始数据文件: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['protein_a_seq', 'protein_b_seq']).copy()
    
    unique_seqs = set(df['protein_a_seq'].tolist() + df['protein_b_seq'].tolist())
    logging.info(f"成功提取 {len(unique_seqs)} 个独特的蛋白质序列，准备计算高维 ESM...")

    # 模型加载
    model_name = "facebook/esm2_t33_650M_UR50D"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    emb_dict = {}
    
    with torch.no_grad():
        for i, seq in enumerate(unique_seqs):
            if i % 100 == 0:
                logging.info(f"进度: {i} / {len(unique_seqs)}")
                
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # 【核心保留】去除首尾特殊字符，保留完整的 (Seq_Len, 1280) 空间序列维度
            hidden_states = outputs.last_hidden_state[0, 1:-1, :] 
            emb_dict[seq] = hidden_states.cpu()

    # 安全覆写
    torch.save(emb_dict, out_path)
    logging.info(f"✅ ESM 高维特征更新完毕！已安全保存至 {out_path}")

if __name__ == "__main__":
    update_esm_only()
