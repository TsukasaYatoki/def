from torch.utils.data import Dataset
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
        self, train, mode, transform, poison_rate=0.01, trigger_size=4, target_label=1
    ):
        random.seed(39)

        self.base_dataset = CIFAR10(root="./dataset", train=train)
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
            return list(total_samples)

        poison_num = int(total_num * self.poison_rate)
        return random.sample(total_samples, poison_num)

    def _get_poison_indices(self):
        return self.poison_indices


class CIFAR10Dataset(CIFAR10):
    """干净数据集，返回 idx, image, label"""

    def __init__(self, train, transform=None, root="./dataset"):
        super().__init__(root, train, transform)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return idx, image, label


class UnlearnDataset(Dataset):
    """用于unlearn的数据集，基于干净数据子集但标签被打乱重新分配"""

    def __init__(self, transform, subset_ratio=0.01):
        random.seed(39)

        self.base_dataset = CIFAR10(root="./dataset")
        self.transform = transform
        self.subset_ratio = subset_ratio
        self.subset_indices = self._build_subset_indices()

        # 获取子集的数据和标签
        self.data = []
        self.original_labels = []
        for idx in self.subset_indices:
            image, label = self.base_dataset[idx]
            self.data.append(image)
            self.original_labels.append(label)

        # 打乱标签重新分配
        self.shuffled_labels = self.original_labels.copy()
        random.shuffle(self.shuffled_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.shuffled_labels[idx]

        if self.transform:
            image = self.transform(image)

        return self.subset_indices[idx], image, label

    def _build_subset_indices(self):
        """构建子集索引"""
        total_num = len(self.base_dataset)
        total_samples = range(total_num)
        subset_num = int(total_num * self.subset_ratio)

        return random.sample(total_samples, subset_num)


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
