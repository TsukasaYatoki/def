import numpy as np
import matplotlib.pyplot as plt

# Load data
backdoor = np.load("feature/unlearn.npz", allow_pickle=True)

ids = backdoor["sample_indices"]
labels = backdoor["labels"]
logits = backdoor["logits"]

# 计算全部样本的预测标签
preds = np.argmax(logits, axis=1)

# 绘制预测标签分布柱状图
plt.figure(figsize=(8, 4))
plt.hist(preds, bins=np.arange(preds.min(), preds.max() + 2) - 0.5, edgecolor="black")
plt.xlabel("Predicted Label")
plt.ylabel("Sample Count")
plt.title("Distribution of Predicted Labels for All Samples")
plt.xticks(np.unique(preds))
plt.tight_layout()
plt.show()
