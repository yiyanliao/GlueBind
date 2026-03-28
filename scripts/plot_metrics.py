import os
import re
import matplotlib.pyplot as plt

def parse_log_and_plot(log_path="train_output.log", save_path="docs/training_metrics.png"):
    if not os.path.exists(log_path):
        print(f"找不到日志文件: {log_path}")
        return

    epochs = []
    train_losses = []
    val_losses = []
    val_aucs = []
    val_auprcs = []
    
    # 使用正则表达式精确提取日志中的数值
    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\] \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val AUC: ([\d.]+) \| Val AUPRC: ([\d.]+)"
    )

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
                val_aucs.append(float(match.group(4)))
                val_auprcs.append(float(match.group(5)))

    if not epochs:
        print("未在日志中找到训练数据，请检查日志格式。")
        return

    # 创建一个 2x2 的图表布局
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GlueBind Model Training Metrics', fontsize=16, fontweight='bold')

    # 图 1: Train & Val Loss 曲线
    axs[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', markersize=4)
    axs[0, 0].plot(epochs, val_losses, label='Val Loss', color='red', marker='x', markersize=4)
    axs[0, 0].set_title('Loss Over Epochs')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Focal Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 图 2: Val AUC 曲线
    axs[0, 1].plot(epochs, val_aucs, label='Val AUC', color='green', marker='s', markersize=4)
    axs[0, 1].set_title('Validation AUC Over Epochs')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('AUC Score')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 图 3: Val AUPRC 曲线 (高亮显示早停最佳点)
    best_epoch_idx = val_auprcs.index(max(val_auprcs))
    axs[1, 0].plot(epochs, val_auprcs, label='Val AUPRC', color='purple', marker='^', markersize=4)
    axs[1, 0].plot(epochs[best_epoch_idx], val_auprcs[best_epoch_idx], 'r*', markersize=12, label=f'Best AUPRC ({max(val_auprcs):.4f})')
    axs[1, 0].set_title('Validation AUPRC Over Epochs')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('AUPRC Score')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 图 4: 文本总结 (Test集最终成绩)
    axs[1, 1].axis('off') # 隐藏坐标轴
    summary_text = (
        "Final Evaluation (Diverse Test Set):\n"
        "------------------------------------\n"
        "Test AUC:   0.7708\n"
        "Test AUPRC: 0.4799\n\n"
        f"Model stopped at Epoch {epochs[-1]}.\n"
        f"Best weights loaded from Epoch {epochs[best_epoch_idx]}."
    )
    axs[1, 1].text(0.1, 0.5, summary_text, fontsize=14, family='monospace', verticalalignment='center')

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"绘图完成！图表已保存至: {save_path}")

if __name__ == "__main__":
    parse_log_and_plot()
