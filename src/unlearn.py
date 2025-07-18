from dataset import CIFAR10Dataset, BackdoorDataset, build_transform, build_dataloader
from eval import eval
from model import ResNet18
from tqdm import tqdm
import torch
import random


def unlearn(model, loader, device="cuda", num_epochs=20, lr=0.01):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for _, (idx, inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            (-loss).backward()
            optimizer.step()

            predicted = outputs.max(1)[1]
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            progress_bar.set_postfix(loss=loss.item())

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Accuracy {acc:.2f}%")

    return model


def main():
    subset_ratio = 0.01

    train_transform, test_transform = build_transform()
    dataset = CIFAR10Dataset(True, transform=train_transform)
    subset = torch.utils.data.Subset(
        dataset, random.sample(range(len(dataset)), int(len(dataset) * subset_ratio))
    )
    data_loader = build_dataloader(subset)

    model = ResNet18()
    model.load_state_dict(torch.load("model/backdoor.pth"))
    model = unlearn(model, data_loader)

    clean_dataset = CIFAR10Dataset(False, test_transform)
    poison_dataset = BackdoorDataset(False, "blended", test_transform)
    clean_loader = build_dataloader(clean_dataset)
    poison_loader = build_dataloader(poison_dataset)

    acc = eval(model, clean_loader)
    asr = eval(model, poison_loader)
    print(f"Model - Accuracy: {acc:.2f}%, ASR: {asr:.2f}%")

    torch.save(model.state_dict(), "model/unlearn.pth")


if __name__ == "__main__":
    main()
