from dataset import BackdoorDataset, DefenseDataset, build_transform
from model import ResNet18
from unlearn import unlearn

from torch.utils.data import DataLoader
import torch


train_transform, test_transform = build_transform()

train_dataset = DefenseDataset("suspect_ids.npy", train_transform, False)
test_dataset = BackdoorDataset(False, "blended", transform=test_transform)
train_loader = DataLoader(train_dataset, 32, True, num_workers=4)
test_loader = DataLoader(test_dataset, 256, False, num_workers=8)

model = ResNet18()
model.load_state_dict(torch.load("model/backdoor.pth"))
model = unlearn(model, train_loader, train_loader, False, 5, 1e-4)

torch.save(model.state_dict(), "model/defense.pth")
