from dataset import DefenseDataset, UnlearnDataset, build_transform
from model import ResNet18

from torch.utils.data import DataLoader
import torch
import itertools


def defense(
    model,
    suspect_loader,
    clean_loader,
    num_epochs=20,
    lr=1e-4,
    unlearn_weight=1.0,
    clean_weight=5.0,
    device="cuda",
):
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        model.train()
        total_unlearn_loss = 0.0
        total_clean_loss = 0.0
        total_suspect_samples = 0
        total_clean_samples = 0

        # 使用itertools.cycle来循环较小的数据集，确保所有数据都被使用
        # 由于clean_dataset更大，我们循环suspect_loader
        suspect_cycle = itertools.cycle(suspect_loader)
        for clean_batch in clean_loader:
            suspect_batch = next(suspect_cycle)
            optimizer.zero_grad()

            _, s_inputs, s_targets = suspect_batch
            s_inputs, s_targets = s_inputs.to(device), s_targets.to(device)
            s_outputs = model(s_inputs)
            suspect_loss = criterion(s_outputs, s_targets)

            _, c_inputs, c_targets = clean_batch
            c_inputs, c_targets = c_inputs.to(device), c_targets.to(device)
            c_outputs = model(c_inputs)
            clean_loss = criterion(c_outputs, c_targets)

            # 组合损失
            combine_loss = (
                clean_loss * clean_weight - suspect_loss * unlearn_weight
            )  # 负梯度用于unlearning

            combine_loss.backward()
            optimizer.step()

            # 分别统计两种损失
            suspect_batch_size = s_targets.size(0)
            clean_batch_size = c_targets.size(0)
            total_unlearn_loss += suspect_loss.item() * suspect_batch_size
            total_clean_loss += clean_loss.item() * clean_batch_size
            total_suspect_samples += suspect_batch_size
            total_clean_samples += clean_batch_size

        avg_unlearn_loss = total_unlearn_loss / total_suspect_samples
        avg_clean_loss = total_clean_loss / total_clean_samples
        print(
            f"Epoch {epoch+1}: Unlearn Loss {avg_unlearn_loss:.4f}, Clean Loss {avg_clean_loss:.4f}"
        )

    return model


transform, _ = build_transform()

suspect_dataset = DefenseDataset("suspect_ids.npy", transform, False)
clean_dataset = UnlearnDataset(transform, confuse=False)
suspect_loader = DataLoader(suspect_dataset, 32, True, num_workers=4)
clean_loader = DataLoader(clean_dataset, 32, True, num_workers=4)

print(len(suspect_dataset), len(clean_dataset))

model = ResNet18()
model.load_state_dict(torch.load("model/backdoor.pth"))
model = defense(model, suspect_loader, clean_loader)

torch.save(model.state_dict(), "model/defense.pth")
