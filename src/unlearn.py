import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from dataset import CIFAR10Dataset, BackdoorTestDataset
from model import ResNet18
from eval import eval
import random

def unlearn(model, train_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    """反向学习：最大化损失函数"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # 最大化损失函数（梯度上升）
            (-0.5 * loss).backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Unlearn Epoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f}')
    
    return model

def main():
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subset_ratio = 0.01  # 使用1%的干净训练数据
    
    print(f'Using device: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 创建干净训练数据集的子集
    train_dataset = CIFAR10Dataset(train=True, transform=transform)
    subset_size = int(len(train_dataset) * subset_ratio)
    subset_indices = random.sample(range(len(train_dataset)), subset_size)
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    
    # 创建测试数据集
    test_dataset = CIFAR10Dataset(train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 创建后门测试数据集
    backdoor_test_dataset = BackdoorTestDataset(test_dataset, target_label=0, trigger_size=4)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print(f'Clean subset size: {len(train_subset)}')
    
    # 加载后门模型
    model = ResNet18()
    model.load_state_dict(torch.load('backdoor_model.pth'))
    print('Loaded backdoor model')
    
    # 执行反向学习
    print('\n=== Starting Unlearning ===')
    model = unlearn(model, train_loader, num_epochs=10, learning_rate=0.001, device=device)
    
    # 评估unlearn后的性能
    acc = eval(model, test_loader, device)
    asr = eval(model, backdoor_test_loader, device)
    print(f'\nFinal - Accuracy: {acc:.2f}%, ASR: {asr:.2f}%')
    
    # 保存unlearn模型
    torch.save(model.state_dict(), 'unlearn_model.pth')
    print('Unlearn model saved to unlearn_model.pth')

if __name__ == '__main__':
    main()