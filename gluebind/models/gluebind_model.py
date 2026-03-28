import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlueBindModel(nn.Module):
    """
    Tripartite Cross-Attention Network 
    专为分子胶诱导的三元复合物 (Target, E3, Molecule) 设计的注意力融合架构。
    """
    def __init__(self, esm_dim=1280, fp_dim=2048, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        # 1. 独立特征加深映射层 (引入 LayerNorm 稳定深层梯度)
        self.target_proj = nn.Sequential(nn.Linear(esm_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.ligase_proj = nn.Sequential(nn.Linear(esm_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.mol_proj = nn.Sequential(nn.Linear(fp_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

        # 【新增】以小分子为 Query 的口袋注意力池化层 (Pocket Attention Pooling)
        self.target_pool_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ligase_pool_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 2. 全局多模态自注意力编码器 (让 Target, E3, Molecule 自由两两交互)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, 
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 更深的前馈网络判别器 (输入维度变为 hidden_dim * 6，因为包含 3个自注意力特征 + 3个物理交互特征)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        logging.info(f"成功初始化 Deep Tripartite Cross-Attention 模型 (Hidden={hidden_dim}, Heads={num_heads}, Layers={num_layers})")

    def forward(self, target_esm, target_mask, ligase_esm, ligase_mask, mol_fp):
        # target_esm 现在包含了空间维度: (Batch, Seq_Len_T, 1280)
        
        t_seq_feat = self.target_proj(target_esm) # (Batch, Seq_Len_T, Hidden_Dim)
        l_seq_feat = self.ligase_proj(ligase_esm) # (Batch, Seq_Len_L, Hidden_Dim)
        m_feat = self.mol_proj(mol_fp)            # (Batch, Hidden_Dim)

        # 构造小分子 Query 去主动寻找蛋白质上的口袋特征
        Q_m = m_feat.unsqueeze(1) # (Batch, 1, Hidden_Dim)
        
        # 动态口袋池化: 屏蔽掉 Padding 的无效区域
        t_pooled, _ = self.target_pool_attn(query=Q_m, key=t_seq_feat, value=t_seq_feat, key_padding_mask=target_mask)
        l_pooled, _ = self.ligase_pool_attn(query=Q_m, key=l_seq_feat, value=l_seq_feat, key_padding_mask=ligase_mask)
        
        # 降维回扁平特征 -> (Batch, Hidden_Dim)
        t_feat_pooled = t_pooled.squeeze(1)
        l_feat_pooled = l_pooled.squeeze(1)

        # 构造 Sequence: 将三个汇聚后的特征视为 3 个平等的 Token，形状 -> (Batch, 3, Hidden_Dim)
        tokens = torch.stack([t_feat_pooled, l_feat_pooled, m_feat], dim=1)

        # 通过 Transformer Encoder 进行全局多模态注意力交互 (T, L, M 完全双向互看)
        encoded_tokens = self.transformer(tokens)
        
        # 提取融合后的各方特征
        t_out = encoded_tokens[:, 0, :]
        l_out = encoded_tokens[:, 1, :]
        m_out = encoded_tokens[:, 2, :]

        # 显式计算二元物理交互 (Element-wise Product)
        tm_interact = t_out * m_out  # Target ⊙ Molecule
        tl_interact = t_out * l_out  # Target ⊙ E3 (极其关键的 PPI 界面)
        lm_interact = l_out * m_out  # E3 ⊙ Molecule

        # 将编码后的独立特征与协同物理交互特征全量拼接 -> (Batch, Hidden_Dim * 6)
        concat_feat = torch.cat([t_out, l_out, m_out, tm_interact, tl_interact, lm_interact], dim=1)

        # 预测打分
        logits = self.mlp(concat_feat)
        return logits
