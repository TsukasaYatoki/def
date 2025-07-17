from dataset import get_dataloader
from model import ResNet18
from tqdm import tqdm
import torch

def train(model, train_loader, num_epochs=50, learning_rate=0.001,
          device='cuda', model_save_path='model.pth'):
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
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    
    return model

def main():
    assert(torch.cuda.is_available()), "CUDA is not available."
    
    print('\n=== Training Clean Model ===')
    clean_train_loader = get_dataloader()
    clean_model = ResNet18()
    train(clean_model, clean_train_loader, model_save_path='clean_model.pth')
    
    print('\n=== Training Backdoor Model ===')
    backdoor_train_loader = get_dataloader(clean=False, poison_rate=0.01)
    backdoor_model = ResNet18()
    train(backdoor_model, backdoor_train_loader, model_save_path='backdoor_model.pth')

if __name__ == '__main__':
    main()