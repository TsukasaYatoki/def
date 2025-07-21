from dataset import UnlearnDataset, build_transform
from model import ResNet18

from torch.utils.data import DataLoader
import torch


def unlearn(
    model, train_loader, test_loader, frozen, num_epochs=20, lr=0.001, device="cuda"
):
    model.to(device)

    if frozen:
        for param in model.linear.parameters():
            param.requires_grad = False
        optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        optimizer_params = model.parameters()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(optimizer_params, lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total = 0
        for _, (idx, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            total += targets.size(0)

        train_loss = train_loss / total

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            total = 0
            for _, (idx, inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                total += targets.size(0)
            test_loss = test_loss / total

        print(
            f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}"
        )

    return model


def main():
    train_transform, test_transform = build_transform()

    train_dataset = UnlearnDataset(train_transform, confuse=True)
    test_dataset = UnlearnDataset(test_transform, confuse=False)
    train_loader = DataLoader(train_dataset, 32, True, num_workers=4)
    test_loader = DataLoader(test_dataset, 32, False, num_workers=4)

    model = ResNet18()
    model.load_state_dict(torch.load("model/backdoor.pth"))
    model = unlearn(model, train_loader, test_loader, True)

    torch.save(model.state_dict(), "model/unlearn.pth")


if __name__ == "__main__":
    main()
