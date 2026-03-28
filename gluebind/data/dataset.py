import torch
from torch.utils.data import Dataset
import random
import logging

class TernaryDataset(Dataset):
    def __init__(self, df, esm_dict, fps_dict, decoys_dict=None, is_train=True):
        self.esm_dict = esm_dict
        self.fps_dict = fps_dict
        self.is_train = is_train
        
        # 终极防御：在装载数据时，直接丢弃那些在原始 CSV 中存在、但 RDKit/ESM 特征提取失败的“脏数据”
        valid_records = []
        missing_count = 0
        for row in df.to_dict('records'):
            if row['smiles'] in self.fps_dict and row['protein_1_seq'] in self.esm_dict and row['protein_2_seq'] in self.esm_dict:
                valid_records.append(row)
            else:
                missing_count += 1
                
        if missing_count > 0:
            logging.warning(f"【数据清洗】由于缺乏预计算特征 (FP 或 ESM)，已安全剔除 {missing_count} 条脏数据。")
            
        self.data = valid_records
        
        # 严格防漏：过滤掉 RDKit 无法解析（没有计算出 FP）的无效 Decoys
        self.decoys_dict = {}
        if decoys_dict:
            for active_sm, decoy_list in decoys_dict.items():
                # 列表推导式：只保留真正在 fps_dict 里有指纹的分子
                valid_decoys = [sm for sm in decoy_list if sm in self.fps_dict]
                if valid_decoys:
                    self.decoys_dict[active_sm] = valid_decoys
        
       # 提取池用于生成负样本的蛋白替换 (现在统一叫 protein_1_seq)
        self.all_target_seqs = list(set(df['protein_1_seq'].tolist() + df['protein_2_seq'].tolist()))
        self.all_molecules = list(self.fps_dict.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        target_seq = row['protein_1_seq']
        ligase_seq = row['protein_2_seq']
        smiles = row['smiles']

        # 如果是测试/验证集，直接读取CSV中固化好的 label；否则进行动态负样本增强
        if not self.is_train:
            label = float(row.get('label', 1.0))
        else:
            # 1:4 动态正负样本策略
            if random.random() < 0.2:
                label = 1.0  # 正样本
            else:
                label = 0.0  # 负样本
                neg_type = random.random()
                
                if neg_type < 0.25: 
                    # Easy Negative: 完全随机分子
                    smiles = random.choice(self.all_molecules)
                elif neg_type < 0.75 and smiles in self.decoys_dict and len(self.decoys_dict[smiles]) > 0:
                    # Hard Negative: Tanimoto Decoys (挑选高相似度无活性分子)
                    smiles = random.choice(self.decoys_dict[smiles])
                else:
                    # Hard Negative: Homologous Target Swaps (替换为其他蛋白)
                    target_seq = random.choice(self.all_target_seqs)

        # 严格提取特征防错
        if target_seq not in self.esm_dict:
            raise KeyError("Target sequence 丢失于 ESM embedding 字典中。")
        if ligase_seq not in self.esm_dict:
            raise KeyError("Ligase sequence 丢失于 ESM embedding 字典中。")
            
        t_esm = self.esm_dict[target_seq]
        l_esm = self.esm_dict[ligase_seq]

        # 如果抽到了 Decoy 分子，我们需要确保有它的 FP
        if smiles in self.fps_dict:
            m_fp = self.fps_dict[smiles]
        else:
            # 此处有坑：我们在 __init__ 里指出需要确认不确定的事
            raise ValueError(f"在 fps_dict 中找不到分子 {smiles} 的指纹。如果这是新生成的 Decoy，你需要先计算它的 Fingerprint！")

        return t_esm, l_esm, m_fp, torch.tensor([label], dtype=torch.float32)
