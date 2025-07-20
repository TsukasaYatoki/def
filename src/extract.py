from dataset import BackdoorDataset, build_transform
from model import ResNet18

from torch.utils.data import DataLoader
import numpy as np
import torch


def get_features(model, loader, device="cuda"):
    """评估模型并提取特征向量"""
    model.to(device)
    model.eval()

    all_indices = []
    all_labels = []
    all_features = []
    all_logits = []

    with torch.no_grad():
        for indices, inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 获取预测和特征
            logits, features = model(inputs, return_features=True)

            # 收集数据
            all_indices.append(indices.numpy())
            all_labels.append(targets.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    # 合并所有批次的数据
    indices = np.concatenate(all_indices, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    features = np.concatenate(all_features, axis=0)
    logits = np.concatenate(all_logits, axis=0)

    return indices, labels, features, logits


def main():
    _, transform = build_transform()
    dataset = BackdoorDataset(True, "blended", transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    poison_indices = dataset._get_poison_indices()
    np.save("feature/poison_id.npy", poison_indices)

    model = ResNet18()
    model.load_state_dict(torch.load("model/unlearn.pth"))

    indices, labels, features, logits = get_features(model, loader)

    results = {
        "indices": indices,
        "labels": labels,
        "features": features,
        "logits": logits,
    }

    np.savez("feature/unlearn.npz", **results)


if __name__ == "__main__":
    main()
