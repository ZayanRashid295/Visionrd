
import torch
import argparse
from dataloader.gtea_dataset import GTEADataset
from torch.utils.data import DataLoader
import yaml

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize the same model architecture
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
        nn.Linear(256, 11)  # 11 classes including background
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to saved model')
    parser.add_argument('--test_data', required=True, help='Path to test data')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.model_path, device)
    model.eval()
    
    # Load test data
    test_dataset = GTEADataset(args.test_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f'Accuracy on test set: {100. * correct / total:.2f}%')

if __name__ == '__main__':
    main()