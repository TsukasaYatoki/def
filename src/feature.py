from dataset import BackdoorDataset, build_transform, build_dataloader
from model import ResNet18
import numpy as np
import torch


def evaluate_with_features(model, loader, device="cuda"):
    """评估模型并提取特征向量"""
    model.to(device)
    model.eval()

    all_features = []
    all_indices = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for indices, inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 获取预测和特征
            logits, features = model(inputs, return_features=True)
            predictions = logits.argmax(dim=1)

            # 收集数据
            all_features.append(features.cpu().numpy())
            all_indices.append(indices.numpy())
            all_labels.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    # 合并所有批次的数据
    features = np.concatenate(all_features, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    # 计算准确率
    accuracy = (predictions == labels).mean() * 100

    return features, indices, labels, predictions, accuracy


def main():
    _, transform = build_transform()
    dataset = BackdoorDataset(True, transform)
    loader = build_dataloader(dataset)

    poison_indices = dataset._get_poison_indices()
    print(f"中毒样本数量: {len(poison_indices)}")

    backdoor_model = ResNet18()
    backdoor_model.load_state_dict(torch.load("backdoor_model.pth"))

    bd_features, bd_indices, bd_labels, bd_predictions, bd_accuracy = (
        evaluate_with_features(backdoor_model, loader)
    )
    print(f"Backdoor准确率: {bd_accuracy:.2f}%")

    unlearn_model = ResNet18()
    unlearn_model.load_state_dict(torch.load("unlearn_model.pth"))

    ul_features, ul_indices, ul_labels, ul_predictions, ul_accuracy = (
        evaluate_with_features(unlearn_model, loader)
    )
    print(f"Unlearn准确率: {ul_accuracy:.2f}%")

    # 保存结果
    results = {
        "poison_indices": list(poison_indices),
        "backdoor_features": bd_features,
        "unlearn_features": ul_features,
        "sample_indices": bd_indices,
        "labels": bd_labels,
        "backdoor_predictions": bd_predictions,
        "unlearn_predictions": ul_predictions,
    }

    np.savez("results.npz", **results)
    print("结果已保存到 results.npz")
    print(f"特征向量维度: {bd_features.shape}")


if __name__ == "__main__":
    main()
