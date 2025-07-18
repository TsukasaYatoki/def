from dataset import BackdoorDataset, build_transform, build_dataloader
from model import ResNet18
from tqdm import tqdm
import torch


def train(model, loader, device="cuda", num_epochs=100, lr=0.001):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=5e-4)

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
            loss.backward()
            optimizer.step()

            predicted = outputs.max(1)[1]
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            progress_bar.set_postfix(loss=loss.item())

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Accuracy {acc:.2f}%")

    return model


def main():
    transform, _ = build_transform()
    dataset = BackdoorDataset(True, "blended", transform)
    dataloader = build_dataloader(dataset)

    model = ResNet18()
    model = train(model, dataloader)

    torch.save(model.state_dict(), "model/backdoor.pth")


if __name__ == "__main__":
    main()
