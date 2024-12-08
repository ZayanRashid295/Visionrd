import os
import torch
import numpy as np
import yaml
import cv2
import torch.utils.data as data
import pycocotools.coco as coco
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GTEADataset(data.Dataset):
    def __init__(self, dataset_dir):
        with open(os.path.join('dataloader', 'dataset.yml'), 'r') as file:
            self.configs = yaml.safe_load(file)
        
        # Initialize correct data paths
        self.data_dir = '/home/user/AI-Hackathon24/data'
        self.data_root = os.path.join(self.data_dir, 'features/gtea_png/png')
        
        # Verify directory exists
        if not os.path.exists(self.data_root):
            raise RuntimeError(f"Data directory not found: {self.data_root}")
            
        # Load COCO annotations with full path
        ann_path = os.path.join(self.data_dir, 'merged.json')
        if not os.path.exists(ann_path):
            raise RuntimeError(f"Annotations file not found: {ann_path}")
            
        self.coco = coco.COCO(ann_path)
        self.image_ids = self.coco.getImgIds()
        
        # Validate image paths and create valid IDs list
        self.valid_ids = []
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(ids=[img_id])[0]
            file_name = img_info['file_name']
            # Remove any '../' or './' from path
            file_name = os.path.normpath(file_name)
            # Try both with and without the features/gtea_png/png prefix
            possible_paths = [
                os.path.join(self.data_root, file_name),
                os.path.join(self.data_dir, file_name),
                os.path.join(self.data_root, os.path.basename(file_name))
            ]
            
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    self.valid_ids.append((img_id, img_path))
                    break
        
        if not self.valid_ids:
            raise RuntimeError("No valid images found!")
            
        print(f"Found {len(self.valid_ids)} valid images")

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, index):
        img_id, img_path = self.valid_ids[index]
        
        # Load and verify image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            # Return a blank image as fallback
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        
        # Get action label with default fallback
        action_label = 10  # default to background class
        if anns and 'action' in anns[0]:
            action_label = self.configs['action_mapping'].get(anns[0]['action'], 10)
        
        # Apply transforms with error handling
        try:
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
        except Exception as e:
            print(f"Transform error for {img_path}: {str(e)}")
            # Return blank transformed image as fallback
            img = torch.zeros((3, 224, 224))
        
        return img, action_label

# Example usage and verification
if __name__ == '__main__':
    # Create dataset and dataloader
    dataset = GTEADataset(dataset_dir='/home/user/AI-Hackathon24/dataloader')
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Verify data loading
    for i, (inputs, labels) in enumerate(loader):
        print(f"\nBatch {i + 1}")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")
        
        if i == 1:  # Only print the first two batches for verification
            break