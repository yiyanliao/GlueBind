import os
import logging
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, EsmModel

# 严格要求：错误绝不静默传递
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_and_align_data(csv_path: str) -> pd.DataFrame:
    """
    清洗原始 CSV，严格根据 target_protein 列对齐 Target 和 Ligase。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始数据文件: {csv_path}")

    df = pd.read_csv(csv_path)
    aligned_data = []

    for index, row in df.iterrows():
        # 显式判断谁是 Target，谁是 Ligase (E3)
        if row['target_protein'] == 'A':
            target_seq = row['protein_a_seq']
            ligase_seq = row['protein_b_seq']
        elif row['target_protein'] == 'B':
            target_seq = row['protein_b_seq']
            ligase_seq = row['protein_a_seq']
        else:
            raise ValueError(f"行 {row['id']} 的 target_protein 无法匹配 A 或 B，当前值为: {row['target_protein']}")

        aligned_data.append({
            'interaction_id': row['id'],
            'target_seq': target_seq,
            'ligase_seq': ligase_seq,
            'smiles': row['canonical_smiles'],
            'label': int(row['label'])
        })
    
    logging.info(f"成功对齐 {len(aligned_data)} 条正样本数据。")
    return pd.DataFrame(aligned_data)


def compute_morgan_fps(smiles_list: list, radius: int = 2, n_bits: int = 2048) -> dict:
    """
    计算 SMILES 列表的 Morgan 指纹，返回 {smiles: tensor} 的字典。
    """
    fps_dict = {}
    for sm in set(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            raise ValueError(f"RDKit 无法解析 SMILES: {sm}")
        
        # 显式生成指纹并转为 PyTorch Tensor
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fp_tensor = torch.tensor(list(fp), dtype=torch.float32)
        fps_dict[sm] = fp_tensor
        
    logging.info(f"成功计算 {len(fps_dict)} 个独特分子的 Morgan Fingerprint。")
    return fps_dict


def compute_esm_embeddings(seq_list: list, model_name: str = "facebook/esm2_t33_650M_UR50D", max_len: int = 1024) -> dict:
    """
    使用 HuggingFace Transformers 提取蛋白质序列的 ESM2 Embedding (取 EOS token 之前的全局表征或平均池化)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"加载 ESM 模型 {model_name} 到 {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    emb_dict = {}
    unique_seqs = set(seq_list)
    
    with torch.no_grad():
        for seq in unique_seqs:
            if not isinstance(seq, str) or len(seq) == 0:
                raise ValueError("遇到无效的空白蛋白质序列。")
                
            # 截断过长序列以防 OOM
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # 采用 sequence 的平均池化作为全局表征 (去除首尾特殊字符)
            hidden_states = outputs.last_hidden_state[0, 1:-1, :] 
            seq_emb = hidden_states.mean(dim=0).cpu()
            
            emb_dict[seq] = seq_emb

    logging.info(f"成功计算 {len(emb_dict)} 个独特蛋白质的 ESM Embedding。")
    return emb_dict


def generate_crem_decoys(smiles_list: list, db_path: str = "replacements02_sc2.db", decoys_per_mol: int = 3) -> dict:
    """
    使用 CReM 生成具有化学合理性的困难负样本 (Decoys)。
    返回 {原始smiles: [decoy_smiles_1, decoy_smiles_2, ...]}
    """
    try:
        from crem.crem import mutate_mol
    except ImportError:
        raise ImportError("请先安装 CReM: pip install crem")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到 CReM 数据库文件: {db_path}。请先下载。")

    decoy_dict = {}
    for sm in set(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        
        # 显式调用 CReM 突变 (限制原子替换范围以保持较高相似度)
        mutations = list(mutate_mol(mol, db_name=db_path, min_size=1, max_size=3, max_replacements=decoys_per_mol))
        
        # mutate_mol 默认直接返回 SMILES 字符串列表，直接截取即可
        decoy_smiles = mutations[:decoys_per_mol]
        if not decoy_smiles:
            logging.warning(f"CReM 无法为 SMILES {sm} 生成突变，请检查分子结构。")

        decoy_dict[sm] = decoy_smiles
        
    logging.info(f"使用 CReM 为 {len(decoy_dict)} 个活性分子生成了负样本。")
    return decoy_dict
