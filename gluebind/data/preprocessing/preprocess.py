import os
import logging
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, EsmModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 严格要求：错误绝不静默传递
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_and_align_data(csv_path: str, augment_swap: bool = True) -> pd.DataFrame:
    """
    清洗新版 CSV，自动为所有数据添加 label=1，并进行 A/B 蛋白的互换扩增。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始数据文件: {csv_path}")

    df = pd.read_csv(csv_path)
    aligned_data = []

    for index, row in df.iterrows():
        seq_a = row['protein_a_seq']
        seq_b = row['protein_b_seq']
        smiles = row['canonical_smiles']
        
        # 严格检查空值
        if pd.isna(seq_a) or pd.isna(seq_b) or pd.isna(smiles):
            continue

        # 基础正样本 (强制加上 label=1，并使用中性的 protein_1 和 protein_2)
        aligned_data.append({
            'interaction_id': f"{row['id']}_orig",
            'protein_1_seq': seq_a,
            'protein_2_seq': seq_b,
            'smiles': smiles,
            'label': 1.0
        })

        # 数据增强：互换蛋白位置 (Swap)
        if augment_swap:
            aligned_data.append({
                'interaction_id': f"{row['id']}_swap",
                'protein_1_seq': seq_b,
                'protein_2_seq': seq_a,
                'smiles': smiles,
                'label': 1.0
            })
            
    logging.info(f"成功解析并扩增数据，共生成 {len(aligned_data)} 条正样本数据。")
    return pd.DataFrame(aligned_data)


def compute_morgan_fps(smiles_list: list, decoys_dict: dict = None, radius: int = 2, n_bits: int = 2048) -> dict:
    """
    计算所有分子的 Morgan 指纹 (包含原始分子 + CReM 生成的诱骗分子)。
    """
    fps_dict = {}
    all_smiles = set(smiles_list)
    
    # 将生成的 Decoy 也加进计算池
    if decoys_dict is not None:
        for decoys in decoys_dict.values():
            all_smiles.update(decoys)

    for sm in all_smiles:
        # CReM 偶尔会生成 RDKit 难以解析的边缘分子，或由于立体化学问题返回 None
        # 这里用 warning 替代 ValueError 防止整个预处理中断
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            logging.warning(f"RDKit 无法解析 SMILES，跳过: {sm}")
            continue
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fp_tensor = torch.tensor(list(fp), dtype=torch.float32)
        fps_dict[sm] = fp_tensor
        
    logging.info(f"成功计算 {len(fps_dict)} 个独特分子(含Decoys)的 Morgan Fingerprint。")
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
            
            # 【核心修改】去除首尾特殊字符，但不做平均池化，保留完整的空间序列维度
            # 最终张量形状为 (Seq_Len, 1280)
            hidden_states = outputs.last_hidden_state[0, 1:-1, :] 
            seq_emb = hidden_states.cpu()
            
            emb_dict[seq] = seq_emb

    logging.info(f"成功计算 {len(emb_dict)} 个独特蛋白质的 ESM Embedding (保留序列维度)。")
    return emb_dict


import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 定义顶层 worker 函数供多进程池调用
def _crem_worker(args):
    sm, row_num, db_path, decoys_per_mol = args
    try:
        from rdkit import Chem
        from crem.crem import mutate_mol
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            return sm, row_num, None, "INVALID_MOL"

        mutations = list(mutate_mol(mol, db_name=db_path, min_size=1, max_size=3, max_replacements=decoys_per_mol))
        decoy_smiles = mutations[:decoys_per_mol]
        if not decoy_smiles:
            return sm, row_num, [], "NO_MUTATION"

        return sm, row_num, decoy_smiles, None
    except Exception as e:
        return sm, row_num, None, str(e)


def generate_crem_decoys(smiles_list: list, db_path: str = "replacements02_sc2.db", decoys_per_mol: int = 3) -> dict:
    """
    使用 CReM 生成具有化学合理性的困难负样本 (Decoys)。
    使用多进程进行加速，并保持严格的顺序输出。
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到 CReM 数据库文件: {db_path}。请先下载。")

    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError("请先安装 tqdm: pip install tqdm")

    decoy_dict = {}
    ordered_unique_smiles = list(dict.fromkeys(smiles_list))

    # 构建多进程任务参数
    worker_args = []
    for sm in ordered_unique_smiles:
        row_num = smiles_list.index(sm) + 2
        worker_args.append((sm, row_num, db_path, decoys_per_mol))

    # 根据服务器配置分配核心数（最多使用 32 核，防止 OOM）
    max_workers = min(32, multiprocessing.cpu_count())
    logging.info(f"启动多进程 CReM 突变池，分配 {max_workers} 个 CPU 核心...")

    # 启动进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 executor.map 保证结果返回的顺序与提交顺序绝对一致
        results = executor.map(_crem_worker, worker_args)

        for result in tqdm(results, total=len(worker_args), desc="CReM Decoys (Multi-core)"):
            sm, row_num, decoys, err = result

            if err == "INVALID_MOL":
                continue
            elif err == "NO_MUTATION":
                logging.warning(f"aligned_data.csv 第 {row_num} 行附近(原CSV约第{(row_num//2)+1}行): CReM 无法为 SMILES {sm} 生成突变，请检查分子结构。")
                decoy_dict[sm] = []
            elif err is not None:
                logging.error(f"aligned_data.csv 第 {row_num} 行附近(原CSV约第{(row_num//2)+1}行): CReM 内部发生错误，跳过 SMILES {sm}。错误详细信息: {err}")
                decoy_dict[sm] = []
            else:
                decoy_dict[sm] = decoys

    logging.info(f"使用 CReM 为 {len(decoy_dict)} 个活性分子生成了负样本。")
    return decoy_dict
