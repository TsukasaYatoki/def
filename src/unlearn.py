from dataset import UnlearnDataset, build_transform
from model import ResNet18

from torch.utils.data import DataLoader
import torch


def unlearn(model, loader, device="cuda", num_epochs=10, lr=0.001):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        for _, (idx, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            predicted = outputs.max(1)[1]
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Accuracy {acc:.2f}%, Loss {loss.item():.4f}")

    return model


def main():
    transform, _ = build_transform()

    dataset = UnlearnDataset(transform, 0.05)
    loader = DataLoader(dataset, 32, True, num_workers=4)

    model = ResNet18()
    model.load_state_dict(torch.load("model/backdoor.pth"))
    model = unlearn(model, loader)

    torch.save(model.state_dict(), "model/unlearn.pth")


if __name__ == "__main__":
    main()
