import os
import json
import random
import logging
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_mmseqs_clustering(fasta_path: str, out_prefix: str, tmp_dir: str):
    """
    调用 MMseqs2 进行严格的 50% 序列相似度聚类。
    """
    cmd = [
        "mmseqs", "easy-cluster", fasta_path, out_prefix, tmp_dir,
        "--min-seq-id", "0.5", "-c", "0.8", "--cov-mode", "1",
        "--single-step-clustering", "1", "--threads", "1"
    ]
    logging.info(f"正在执行 MMseqs2 聚类: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"MMseqs2 运行失败!\n{result.stderr}")
        raise RuntimeError("MMseqs2 聚类失败，绝不静默传递错误。")
    
    tsv_path = f"{out_prefix}_cluster.tsv"
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"未找到预期的 MMseqs2 输出文件: {tsv_path}")
    
    return tsv_path

def compute_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def get_max_tanimoto(query_smiles, target_fps_dict):
    """计算单个分子与一组分子之间的最大 Tanimoto 相似度"""
    query_fp = compute_fp(query_smiles)
    if query_fp is None: return 1.0 # 如果解析失败，保守视为高相似度（触发丢弃）
    
    max_sim = 0.0
    for target_fp in target_fps_dict.values():
        sim = DataStructs.TanimotoSimilarity(query_fp, target_fp)
        if sim > max_sim:
            max_sim = sim
    return max_sim

def generate_static_samples(df_positives, all_molecules, all_target_seqs, decoys_dict):
    """为验证集和测试集生成 1:4 的固定静态正负样本"""
    static_data = []
    
    for _, row in df_positives.iterrows():
        t_seq = row['protein_a_seq']
        l_seq = row['protein_b_seq']
        smiles = row['canonical_smiles']
        
        # 1. 写入正样本
        static_data.append({'protein_1_seq': t_seq, 'protein_2_seq': l_seq, 'smiles': smiles, 'label': 1.0})
        
        # 2. 生成 4 个静态负样本
        for _ in range(4):
            neg_type = random.random()
            neg_smiles = smiles
            neg_t_seq = t_seq
            
            if neg_type < 0.25:
                # Easy Negative: 随机分子
                neg_smiles = random.choice(all_molecules)
            elif neg_type < 0.75 and smiles in decoys_dict and len(decoys_dict[smiles]) > 0:
                # Hard Negative: CReM 诱骗分子
                neg_smiles = random.choice(decoys_dict[smiles])
            else:
                # Hard Negative: 随机替换 Target 蛋白
                neg_t_seq = random.choice(all_target_seqs)
                
            static_data.append({'protein_1_seq': neg_t_seq, 'protein_2_seq': l_seq, 'smiles': neg_smiles, 'label': 0.0})
            
    return pd.DataFrame(static_data)

def main():
    csv_path = "dataset/zuixinban.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始数据文件: {csv_path}")

    # 加载 Deco 数据以供生成负样本
    decoys_path = "dataset/crem_decoys.json"
    decoys_dict = {}
    if os.path.exists(decoys_path):
        with open(decoys_path, "r") as f:
            decoys_dict = json.load(f)
    else:
        logging.warning("未找到 crem_decoys.json，将仅使用随机替换生成负样本。")

    df = pd.read_csv(csv_path)
    # 严格清理空值
    df = df.dropna(subset=['protein_a_seq', 'protein_b_seq', 'canonical_smiles']).copy()
    
    all_molecules = df['canonical_smiles'].unique().tolist()
    all_target_seqs = list(set(df['protein_a_seq'].tolist() + df['protein_b_seq'].tolist()))

    # 1. 提取所有唯一蛋白并生成 FASTA
    logging.info("提取唯一蛋白质序列...")
    unique_proteins = list(set(df['protein_a_seq'].tolist() + df['protein_b_seq'].tolist()))
    fasta_path = "dataset/tmp_proteins.fasta"
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(unique_proteins):
            f.write(f">PROT_{i}\n{seq}\n")
            
    # 建立 ID 到序列的映射
    id_to_seq = {f"PROT_{i}": seq for i, seq in enumerate(unique_proteins)}
    seq_to_id = {seq: f"PROT_{i}" for i, seq in enumerate(unique_proteins)}

    # 2. 运行 MMseqs2
    tsv_path = run_mmseqs_clustering(fasta_path, "dataset/tmp_clu", "dataset/tmp_dir")
    
    # 3. 解析聚类结果
    seq_to_cluster = {}
    with open(tsv_path, "r") as f:
        for line in f:
            rep, member = line.strip().split('\t')
            seq_to_cluster[id_to_seq[member]] = rep
            
    # --- 终极修复：基于 Cluster 大小的防黑洞策略 ---
    seq_counts = pd.concat([df['protein_a_seq'], df['protein_b_seq']]).value_counts().to_dict()

    def get_target_seq(row):
        count_a = seq_counts.get(row['protein_a_seq'], 0)
        count_b = seq_counts.get(row['protein_b_seq'], 0)
        return row['protein_a_seq'] if count_a <= count_b else row['protein_b_seq']
        
    df['real_target_seq'] = df.apply(get_target_seq, axis=1)
    df['target_cluster'] = df['real_target_seq'].map(seq_to_cluster)
    
    # 统计每个 Cluster 包含的数据行数
    cluster_sizes = df['target_cluster'].value_counts()
    
    # 将占据总数据量 80% 的头部巨无霸 Clusters 锁定在 Train 中，严禁进入 Test
    total_rows = len(df)
    cumulative_sum = 0
    train_only_clusters = set()
    
    for cluster, size in cluster_sizes.items():
        if cumulative_sum < total_rows * 0.8:
            train_only_clusters.add(cluster)
            cumulative_sum += size
        else:
            break
            
    # 从剩下的长尾 Cluster 中挑选 100 个作为 Test/Val 的独立宇宙
    candidate_clusters = [c for c in cluster_sizes.index if c not in train_only_clusters]
    import random
    random.seed(42) # 保证可重复
    selected_test_clusters = set(random.sample(candidate_clusters, min(100, len(candidate_clusters))))
    logging.info(f"成功锁定 {len(train_only_clusters)} 个巨无霸靶点至 Train。抽选了 {len(selected_test_clusters)} 个独立 Cluster 用于 Test/Val。")

    # 4. 抽取 Test Samples
    logging.info("筛选 Test Set...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_positives = []
    test_fps_dict = {}
    test_sampled_clusters = set()
    
    # 第一遍遍历：为选中的每个 Test Cluster 精确抽取 1 条数据
    for idx, row in df_shuffled.iterrows():
        c_target = row['target_cluster']
        if c_target in selected_test_clusters and c_target not in test_sampled_clusters:
            test_positives.append(row)
            test_sampled_clusters.add(c_target)
            fp = compute_fp(row['canonical_smiles'])
            if fp is not None:
                test_fps_dict[row['canonical_smiles']] = fp

    df_test_pos = pd.DataFrame(test_positives)
    
    # 5. 划分 Train 和 Val (执行 Scaffold Hopping 防泄漏)
    logging.info("执行验证集 Scaffold Hopping 筛选及防泄漏丢弃...")
    train_positives = []
    val_positives = []
    discarded_count = 0
    
    # 第二遍遍历：处理剩余数据
    remaining_df = df_shuffled.drop(df_test_pos.index)
    
    for _, row in remaining_df.iterrows():
        c_target = row['target_cluster']
        
        # 如果属于 Test 的势力范围
        if c_target in selected_test_clusters:
            # 检查配体相似度
            max_t = get_max_tanimoto(row['canonical_smiles'], test_fps_dict)
            if max_t < 0.85:
                val_positives.append(row)
            else:
                discarded_count += 1
        else:
            # 属于巨无霸或其他未被选中作为 Test 的 Cluster，安全进入 Train
            train_positives.append(row)

    df_val_pos = pd.DataFrame(val_positives)
    df_train_pos = pd.DataFrame(train_positives)
    logging.info(f"正样本分配完毕: Train={len(df_train_pos)}, Val={len(df_val_pos)}, Test={len(df_test_pos)}, 剔除泄漏风险={discarded_count}")

    # 6. 生成输出数据 (Train 仅保留正样本供 Dataset 动态增强；Val/Test 生成静态 1:4 负样本)
    logging.info("正在固化 Val / Test 的静态负样本...")
    
    train_out = []
    for _, row in df_train_pos.iterrows():
        # Train 集强制 label=1，并使用中性的 protein_1 和 protein_2 (兼容后续的 Swap 增强)
        train_out.append({'protein_1_seq': row['protein_a_seq'], 'protein_2_seq': row['protein_b_seq'], 'smiles': row['canonical_smiles'], 'label': 1.0})
    
    df_train_final = pd.DataFrame(train_out)
    df_val_final = generate_static_samples(df_val_pos, all_molecules, all_target_seqs, decoys_dict)
    df_test_final = generate_static_samples(df_test_pos, all_molecules, all_target_seqs, decoys_dict)
    
    df_train_final.to_csv("dataset/train.csv", index=False)
    df_val_final.to_csv("dataset/val.csv", index=False)
    df_test_final.to_csv("dataset/test.csv", index=False)
    
    # 清理临时文件
    os.remove(fasta_path)
    os.remove("dataset/tmp_clu_cluster.tsv")
    logging.info("数据集划分完成！生成 dataset/train.csv, dataset/val.csv, dataset/test.csv")

if __name__ == "__main__":
    main()
