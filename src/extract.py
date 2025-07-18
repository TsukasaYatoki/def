from dataset import BackdoorDataset, build_transform, build_dataloader
from model import ResNet18
import numpy as np
import torch


def get_features(model, loader, device="cuda"):
    """评估模型并提取特征向量"""
    model.to(device)
    model.eval()

    all_features = []
    all_indices = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for indices, inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 获取预测和特征
            logits, features = model(inputs, return_features=True)

            # 收集数据
            all_features.append(features.cpu().numpy())
            all_indices.append(indices.numpy())
            all_labels.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    # 合并所有批次的数据
    features = np.concatenate(all_features, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)

    return features, indices, labels, logits


def main():
    _, transform = build_transform()
    dataset = BackdoorDataset(True, "blended", transform)
    loader = build_dataloader(dataset)

    poison_indices = dataset._get_poison_indices()

    model = ResNet18()
    model.load_state_dict(torch.load("model/big.pth"))

    features, indices, labels, logits = get_features(model, loader)

    results = {
        "sample_indices": indices,
        "labels": labels,
        "poison_indices": list(poison_indices),
        "features": features,
        "logits": logits,
    }

    np.savez("feature/big.npz", **results)


if __name__ == "__main__":
    main()
