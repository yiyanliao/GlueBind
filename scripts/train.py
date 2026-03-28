import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

# 导入我们写的模块
from gluebind.models.gluebind_model import GlueBindModel
from gluebind.data.dataset import TernaryDataset
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FocalLoss(nn.Module):
    """解决 1:4 极度不平衡与 Hard Decoys 困难样本的自适应损失函数"""
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # 控制正负样本比例
        self.gamma = gamma # 控制困难样本权重

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def ternary_collate_fn(batch):
    """动态 Padding 处理不等长的序列特征，并生成掩码"""
    t_esms = [item[0] for item in batch]
    l_esms = [item[1] for item in batch]
    m_fps = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    # 动态用 0.0 填充至当前 Batch 内的最长序列
    t_esm_padded = pad_sequence(t_esms, batch_first=True, padding_value=0.0)
    l_esm_padded = pad_sequence(l_esms, batch_first=True, padding_value=0.0)
    
    # 生成 Padding Mask: 数值为 0 的位置设为 True (让 Attention 忽略这些位置)
    t_mask = (t_esm_padded[:, :, 0] == 0.0)
    l_mask = (l_esm_padded[:, :, 0] == 0.0)
    
    m_fps_tensor = torch.stack(m_fps)
    labels_tensor = torch.stack(labels)
    
    return t_esm_padded, t_mask, l_esm_padded, l_mask, m_fps_tensor, labels_tensor

def train_model():
    # 1. 检查数据文件是否存在
    required_files = [
        "dataset/train.csv",
        "dataset/val.csv",
        "dataset/test.csv",
        "dataset/esm_embeddings.pt",
        "dataset/morgan_fps.pt",
        "dataset/crem_decoys.json"
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺失必要文件: {file_path}。请确保已运行拆分脚本并生成了特征文件。")

    # 2. 加载分割好的数据集
    logging.info("加载特征文件...")
    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/val.csv")
    test_df = pd.read_csv("dataset/test.csv")
    esm_dict = torch.load("dataset/esm_embeddings.pt")
    fps_dict = torch.load("dataset/morgan_fps.pt")
    with open("dataset/crem_decoys.json", "r") as f:
        decoys_dict = json.load(f)

    # 3. 实例化数据集 (显式指明 is_train 状态)
    train_dataset = TernaryDataset(train_df, esm_dict, fps_dict, decoys_dict, is_train=True)
    val_dataset = TernaryDataset(val_df, esm_dict, fps_dict, decoys_dict, is_train=False)
    test_dataset = TernaryDataset(test_df, esm_dict, fps_dict, decoys_dict, is_train=False)

    # 性能解除封印：增大 Batch Size，开启 4 个 worker 多进程加载，并启用 pin_memory 加速 CPU->GPU 传输
    # 注意：如果 128 爆显存 (OOM)，可以改回 64。如果显存还空很多，可以拉到 256。
    BATCH_SIZE = 128
    NUM_WORKERS = 1
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=ternary_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ternary_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ternary_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    logging.info(f"数据加载完毕。训练集/验证集/测试集 Batch 数: {len(train_loader)} / {len(val_loader)} / {len(test_loader)}")

    # 4. 初始化硬件和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    model = GlueBindModel(esm_dim=1280, fp_dim=2048, hidden_dim=256, num_heads=8, num_layers=3).to(device)
    
    # 启用 Focal Loss 替代 BCELoss，专治极度不平衡与困难样本
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 5. 训练循环与早停机制 (Early Stopping)
    num_epochs = 50
    patience = 10
    best_auprc = 0.0
    epochs_no_improve = 0
    best_model_path = "dataset/gluebind_model.pth"
    
    logging.info("开始多模态深度模型训练 (引入 Focal Loss 与早停)...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (t_esm, t_mask, l_esm, l_mask, m_fp, labels) in enumerate(train_loader):
            # 将数据推到 GPU
            t_esm, t_mask = t_esm.to(device), t_mask.to(device)
            l_esm, l_mask = l_esm.to(device), l_mask.to(device)
            m_fp, labels = m_fp.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播 (带入掩码)
            preds = model(t_esm, t_mask, l_esm, l_mask, m_fp)
            loss = criterion(preds, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # 验证循环
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for t_esm, t_mask, l_esm, l_mask, m_fp, labels in val_loader:
                t_esm, t_mask = t_esm.to(device), t_mask.to(device)
                l_esm, l_mask = l_esm.to(device), l_mask.to(device)
                m_fp, labels = m_fp.to(device), labels.to(device)
                
                preds = model(t_esm, t_mask, l_esm, l_mask, m_fp)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                
                # 收集真实值和预测概率用于计算 AUC 和 AUPRC
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
       
        # 严谨评估指标计算 (防范极端情况)
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            val_auprc = average_precision_score(all_labels, all_preds)
        except ValueError:
            val_auc, val_auprc = 0.0, 0.0
            
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")

        # 早停逻辑判定 (监控 AUPRC)
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"   -> 发现更佳模型！AUPRC 提升至 {val_auprc:.4f}，权重已保存。")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.warning(f"连续 {patience} 个 Epoch 验证集指标未提升，触发早停 (Early Stopping)。")
                break

    # 6. 测试集终极评估 (加载早停保存的最佳权重)
    logging.info(f"正在加载最佳验证集权重进行 Test 集测试...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for t_esm, t_mask, l_esm, l_mask, m_fp, labels in test_loader:
            t_esm, t_mask = t_esm.to(device), t_mask.to(device)
            l_esm, l_mask = l_esm.to(device), l_mask.to(device)
            m_fp, labels = m_fp.to(device), labels.to(device)
            
            preds = model(t_esm, t_mask, l_esm, l_mask, m_fp)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    try:
        test_auc = roc_auc_score(test_labels, test_preds)
        test_auprc = average_precision_score(test_labels, test_preds)
    except ValueError:
        test_auc, test_auprc = 0.0, 0.0
        
    logging.info("="*50)
    logging.info(f"终极测试集 (Diverse Clusters) 成绩单：")
    logging.info(f"Test AUC:   {test_auc:.4f}")
    logging.info(f"Test AUPRC: {test_auprc:.4f}")
    logging.info("="*50)

if __name__ == "__main__":
    train_model()
