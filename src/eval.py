import torch
from dataset import get_dataloader
from model import ResNet18

def eval(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx, inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total

def main():
    assert(torch.cuda.is_available()), "CUDA is not available."
    
    clean_test_loader = get_dataloader(train=False)
    backdoor_test_loader = get_dataloader(clean=False, train=False)
    
    print('\n=== Evaluating Clean Model ===')
    clean_model = ResNet18()
    clean_model.load_state_dict(torch.load('clean_model.pth'))
    clean_acc = eval(clean_model, clean_test_loader)
    clean_asr = eval(clean_model, backdoor_test_loader)
    print(f'Clean Model - Accuracy: {clean_acc:.2f}%, ASR: {clean_asr:.2f}%')
    
    print('\n=== Evaluating Backdoor Model ===')
    backdoor_model = ResNet18()
    backdoor_model.load_state_dict(torch.load('backdoor_model.pth'))
    backdoor_acc = eval(backdoor_model, clean_test_loader)
    backdoor_asr = eval(backdoor_model, backdoor_test_loader)
    print(f'Backdoor Model - Accuracy: {backdoor_acc:.2f}%, ASR: {backdoor_asr:.2f}%')

if __name__ == '__main__':
    main()