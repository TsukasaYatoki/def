import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CIFAR10Dataset, BackdoorDataset
from model import ResNet18
import json
import numpy as np

def evaluate_with_outputs(model, data_loader, device='cuda'):
    """评估模型并保存每个样本的ID和输出"""
    model.to(device)
    model.eval()
    
    results = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (sample_ids, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = model(inputs, return_features=True)
            _, predicted = outputs.max(1)
            
            # 记录每个样本的结果
            for i in range(len(sample_ids)):
                results.append({
                    'sample_id': sample_ids[i].item(),
                    'true_label': targets[i].item(),
                    'predicted_label': predicted[i].item(),
                    'features': features[i].cpu().numpy().tolist(),
                    'logits': outputs[i].cpu().numpy().tolist(),
                    'is_correct': (predicted[i] == targets[i]).item()
                })
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy, results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 创建基础训练数据集
    base_dataset = CIFAR10Dataset(train=True, transform=transform)
    
    # 重建中毒训练集（使用相同的中毒参数）
    poison_dataset = BackdoorDataset(
        base_dataset,
        load_from_file=True,
        filename="poison_info.json"
    )
    
    poison_loader = DataLoader(poison_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print(f'Poison dataset size: {len(poison_dataset)}')
    print(f'Poison samples: {len(poison_dataset.poison_indices)}')
    
    # 加载unlearn模型
    model = ResNet18()
    model.load_state_dict(torch.load('unlearn_model.pth'))
    print('Loaded unlearn model')
    
    # 评估unlearn模型在中毒数据集上的表现
    print('\n=== Evaluating Unlearn Model on Poison Dataset ===')
    accuracy, results = evaluate_with_outputs(model, poison_loader, device)
    
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    # 分析中毒样本和干净样本的表现
    poison_results = [r for r in results if r['sample_id'] in poison_dataset.poison_indices]
    clean_results = [r for r in results if r['sample_id'] not in poison_dataset.poison_indices]
    
    poison_accuracy = 100. * sum([r['is_correct'] for r in poison_results]) / len(poison_results)
    clean_accuracy = 100. * sum([r['is_correct'] for r in clean_results]) / len(clean_results)
    
    print(f'Poison samples accuracy: {poison_accuracy:.2f}% ({len(poison_results)} samples)')
    print(f'Clean samples accuracy: {clean_accuracy:.2f}% ({len(clean_results)} samples)')
    
    # 保存详细结果
    output_data = {
        'overall_accuracy': accuracy,
        'poison_accuracy': poison_accuracy,
        'clean_accuracy': clean_accuracy,
        'poison_indices': list(poison_dataset.poison_indices),
        'target_label': poison_dataset.target_label,
        'results': results
    }
    
    with open('unlearn_evaluation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print('\nResults saved to unlearn_evaluation_results.json')

if __name__ == '__main__':
    main()