
import torch
from models.custom_model import CustomModel
from dataloader.gtea_dataset import GTEADataset
from utils.data_visualization import visualize_annotations
import yaml
import os
import cv2
import numpy as np

def evaluate_model(model_path, config_path="dataloader/dataset.yml"):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and data
    model = CustomModel('resnet50', num_classes=len(config['action_mapping']))
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    dataset = GTEADataset(config['dataset_dir'])
    
    # Create results directory
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate each sample
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, keypoints, bboxes, segmentation, label = dataset[idx]
            
            # Prepare input
            image = image.unsqueeze(0).to(device)
            keypoints = keypoints.unsqueeze(0).to(device)
            bboxes = bboxes.unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(image, keypoints, bboxes)
            predictions = torch.softmax(outputs, dim=1)
            
            # Visualize results
            vis_img = visualize_annotations(
                image[0].cpu().numpy().transpose(1,2,0),
                [{'keypoints': keypoints[0].cpu().numpy(),
                  'bbox': bboxes[0].cpu().numpy()}],
                action_probs=predictions[0].cpu().numpy(),
                class_names=list(config['action_mapping'].keys())
            )
            
            # Save visualization
            cv2.imwrite(os.path.join(results_dir, f'sample_{idx}.png'), vis_img)

if __name__ == '__main__':
    evaluate_model('model.pth')