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
        self.alpha = 0.2  # 混合比例
        self.blend = np.load("trigger/blended.npy")
        self.sig = np.load("trigger/signal.npy").reshape((32, 32, 1))

    def add_trigger(self, image):
        """添加触发器 - 在transform之前"""
        image = np.array(image).copy()

        if self.mode == "badnet":
            image[-self.trigger_size :, -self.trigger_size :, :] = 255
        elif self.mode == "blended":
            image = (1 - self.alpha) * image + self.alpha * self.blend
        elif self.mode == "signal":
            image = (1 - self.alpha) * image + self.alpha * self.sig
        else:
            raise ValueError(f"Unsupported attack mode: {self.mode}")

        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image), self.target_label


class BackdoorDataset(Dataset):
    """后门数据集"""

    def __init__(
        self, train, mode, transform=None, target_label=1, poison_rate=0.01, trig_size=4
    ):
        random.seed(39)

        self.base_dataset = CIFAR10(root="./dataset", train=train)
        self.transform = transform
        self.poison_rate = poison_rate
        self.train = train
        self.trigger_handler = TriggerHandler(target_label, mode, trig_size)
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
    """用于unlearn的数据集，基于干净数据子集"""

    def __init__(self, transform=None, subset_ratio=0.01, confuse=True):
        random.seed(39)

        self.base_dataset = CIFAR10(root="./dataset")
        self.transform = transform
        self.subset_ratio = subset_ratio
        self.shuffle = confuse
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

        if self.shuffle:
            label = self.shuffled_labels[idx]
        else:
            label = self.original_labels[idx]

        if self.transform:
            image = self.transform(image)

        return self.subset_indices[idx], image, label

    def _build_subset_indices(self):
        """构建子集索引"""
        total_num = len(self.base_dataset)
        total_samples = range(total_num)
        subset_num = int(total_num * self.subset_ratio)

        return random.sample(total_samples, subset_num)


class DefenseDataset(Dataset):
    """用于防御的子集数据集"""

    def __init__(self, indeces_path, transform=None, random_label=True):
        random.seed(39)

        self.base_dataset = BackdoorDataset(True, "blended", None)
        self.transform = transform
        self.subset_indices = np.load(indeces_path).tolist()
        self.random = random_label

        self.images = []
        self.labels = []
        for idx in self.subset_indices:
            _, image, label = self.base_dataset[idx]
            self.images.append(image)
            self.labels.append(label)

        self.random_labels = [
            random.randint(0, 9) for _ in range(len(self.subset_indices))
        ]

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.random:
            label = self.random_labels[idx]
        else:
            label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return self.subset_indices[idx], image, label


class TestASRDataset(Dataset):
    """用于测试ASR的子集数据集"""

    def __init__(self, mode, transform=None, target_label=1, trigger_size=4):
        self.base_dataset = CIFAR10(root="./dataset", train=False)
        self.trigger_handler = TriggerHandler(target_label, mode, trigger_size)
        self.transform = transform

        self.indices = []
        self.images = []
        for idx in range(len(self.base_dataset)):
            image, label = self.base_dataset[idx]
            if label != target_label:
                self.indices.append(idx)
                self.images.append(image)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        indice = self.indices[idx]
        image = self.images[idx]

        image, label = self.trigger_handler.add_trigger(image)

        if self.transform:
            image = self.transform(image)

        return indice, image, label


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
