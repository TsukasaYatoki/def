from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import random

MEAN_CIFAR10 = [0.4914, 0.4822, 0.4465]
STD_CIFAR10 = [0.2023, 0.1994, 0.2010]


class TriggerHandler:
    """处理后门触发器的类"""

    def __init__(self, target_label, mode, trigger_size=4):
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.mode = mode

        with open("trigger/blended.npy", "rb") as f:
            self.pattern = np.load(f)

    def add_trigger(self, image):
        """添加触发器 - 在transform之前"""
        image = np.array(image).copy()

        if self.mode == "badnet":
            image[-self.trigger_size :, -self.trigger_size :, :] = 255
        elif self.mode == "blended":
            alpha = 0.2
            image = (1 - alpha) * image + alpha * self.pattern
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported attack mode: {self.mode}")

        return Image.fromarray(image), self.target_label


class BackdoorDataset(Dataset):
    """后门数据集"""

    def __init__(
        self,
        train,
        mode,
        transform=None,
        data_dir="./dataset",
        poison_rate=0.01,
        trigger_size=4,
        target_label=0,
    ):
        self.base_dataset = CIFAR10(root=data_dir, train=train)
        self.transform = transform
        self.poison_rate = poison_rate
        self.train = train
        self.trigger_handler = TriggerHandler(target_label, mode, trigger_size)
        self.poison_indices = self._build_poison_indices()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        if idx in self.poison_indices:
            image, label = self.trigger_handler.add_trigger(image)

        if self.transform:
            image = self.transform(image)

        return idx, image, label

    def _build_poison_indices(self):
        total_num = len(self.base_dataset)
        total_samples = range(total_num)

        if not self.train:
            return set(total_samples)

        random.seed(39)
        return set(random.sample(total_samples, int(total_num * self.poison_rate)))

    def _get_poison_indices(self):
        return self.poison_indices


class CIFAR10Dataset(CIFAR10):
    """干净数据集，返回 idx, image, label"""

    def __init__(self, train, transform=None, root="./dataset"):
        super().__init__(root, train, transform)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return idx, image, label


def build_transform():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
        ]
    )
    return train_transform, test_transform


def build_dataloader(dataset, batch_size=256, num_workers=4):
    shuffle = dataset.train if hasattr(dataset, "train") else True
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)
