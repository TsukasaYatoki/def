from dataset import CIFAR10Dataset, BackdoorDataset, build_transform
from model import ResNet18

from torch.utils.data import DataLoader
import torch


def eval(model, loader, device="cuda"):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for idx, inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def main():
    _, transform = build_transform()

    clean_dataset = CIFAR10Dataset(False, transform=transform)
    poison_dataset = BackdoorDataset(False, "blended", transform=transform)
    clean_loader = DataLoader(clean_dataset, 128, shuffle=False, num_workers=4)
    poison_loader = DataLoader(poison_dataset, 128, shuffle=False, num_workers=4)

    model = ResNet18()
    model.load_state_dict(torch.load("model/defense.pth"))

    acc = eval(model, clean_loader)
    asr = eval(model, poison_loader)
    print(f"Model - Accuracy: {acc:.2f}%, ASR: {asr:.2f}%")


if __name__ == "__main__":
    main()
