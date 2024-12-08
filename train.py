import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader.gtea_dataset import GTEADataset
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
            
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train_model(config_path="dataloader/dataset.yml"):
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir='runs/gtea_action_recognition')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize dataset and model
    dataset = GTEADataset(config['dataset_dir'])
    
    # Split dataset into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize base model for action recognition
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, len(config['action_mapping']))
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress_bar = tqdm(train_loader)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / labels.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update running metrics
            running_loss += loss.item()
            running_acc += accuracy
            
            # Log batch metrics
            writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Batch/Accuracy', accuracy, epoch * len(train_loader) + batch_idx)
            
            progress_bar.set_description(f'Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        
        # Evaluate
        val_accuracy, val_loss = evaluate_model(model, test_loader, criterion, device)
        
        # Log epoch metrics
        writer.add_scalars('Epoch/Loss', {
            'train': epoch_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('Epoch/Accuracy', {
            'train': epoch_acc,
            'val': val_accuracy
        }, epoch)
        
        # Log model gradients and weights
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        print(f'Epoch {epoch}:')
        print(f'  Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, 'best_model.pth')
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch+1}.pth')

    writer.close()

if __name__ == '__main__':
    train_model()