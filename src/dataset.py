from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy
import random

class TriggerHandler:
    """处理后门触发器的类"""
    def __init__(self, trigger_size=4, target_label=0):
        self.trigger_size = trigger_size
        self.target_label = target_label
    
    def add_trigger(self, image):
        """添加触发器 - 在transform之前"""
        image = numpy.array(image).copy()
        image[-self.trigger_size:, -self.trigger_size:, :] = 255
        return Image.fromarray(image), self.target_label

class BackdoorDataset(Dataset):
    """后门数据集"""
    def __init__(self, data_dir='./dataset', train=True, transform=None, poison_rate=0.1):
        self.base_dataset = CIFAR10(root=data_dir, train=train)
        self.transform = transform
        self.poison_rate = poison_rate if train else 1.0
        self.train = train
        self.trigger_handler = TriggerHandler()
        self.poison_indices = self._build_poison_indices(39)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if idx in self.poison_indices:
            image, label= self.trigger_handler.add_trigger(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _build_poison_indices(self, seed):
        random.seed(seed)
        total_samples = len(self.base_dataset)
        poison_num = int(total_samples * self.poison_rate)
        return set(random.sample(range(total_samples), poison_num))

def get_dataloader(clean=True, train=True, poison_rate=0.1, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    if clean:
        dataset = CIFAR10(root='./dataset', train=train, transform=transform)
    else:
        dataset = BackdoorDataset(train=train, transform=transform, poison_rate=poison_rate)
    
    shuffle = True if train else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader