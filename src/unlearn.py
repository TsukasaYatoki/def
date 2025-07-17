from dataset import get_dataloader
from eval import eval
from model import ResNet18
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import random

def unlearn(model, train_loader, num_epochs=50, learning_rate=0.0001, device='cuda'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for _, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            (-loss).backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
    
    return model

def main():
    assert(torch.cuda.is_available()), "CUDA is not available."
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    dataset = CIFAR10(root='./dataset', train=True, transform=transform)
    subset_ratio = 0.01  # 使用1%的干净训练数据
    subset_size = int(len(dataset) * subset_ratio)
    subset_indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, subset_indices)
    data_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=4)
    
    # 加载后门模型
    model = ResNet18()
    model.load_state_dict(torch.load('backdoor_model.pth'))
    model = unlearn(model, data_loader)
    
    # 评估unlearn后的性能
    clean_test_loader = get_dataloader(train=False)
    backdoor_test_loader = get_dataloader(clean=False, train=False)
    acc = eval(model, clean_test_loader)
    asr = eval(model, backdoor_test_loader)
    print(f'\nAccuracy: {acc:.2f}%, ASR: {asr:.2f}%')
    
    # 保存unlearn模型
    torch.save(model.state_dict(), 'unlearn_model.pth')
    print('Model saved to unlearn_model.pth')

if __name__ == '__main__':
    main()