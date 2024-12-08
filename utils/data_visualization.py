import cv2
import numpy as np
import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

FINGER_CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12)],
    'ring': [(13, 14), (14, 15), (15, 16)],
    'pinky': [(17, 18), (18, 19), (19, 20)]
}

def draw_hand_skeleton(image, keypoints, connections, color=(0, 255, 0)):
    """Draw hand skeleton with different colors for each finger."""
    finger_colors = {
        'thumb': (255, 0, 0),    # Red
        'index': (0, 255, 0),    # Green
        'middle': (0, 0, 255),   # Blue
        'ring': (255, 255, 0),   # Yellow
        'pinky': (255, 0, 255)   # Magenta
    }
    
    keypoints = np.array(keypoints).flatten()  # Convert keypoints to numpy array and flatten
    
    for finger, finger_connections in FINGER_CONNECTIONS.items():
        color = finger_colors[finger]
        for connection in finger_connections:
            start_idx = connection[0] * 3
            end_idx = connection[1] * 3
            
            start_point = (int(keypoints[start_idx]), int(keypoints[start_idx + 1]))
            end_point = (int(keypoints[end_idx]), int(keypoints[end_idx + 1]))
            
            if keypoints[start_idx + 2] > 0 and keypoints[end_idx + 2] > 0:
                cv2.line(image, start_point, end_point, color, 2)
                cv2.circle(image, start_point, 3, color, -1)
                cv2.circle(image, end_point, 3, color, -1)

def visualize_action_heatmap(image, action_probs, class_names):
    """Create heatmap visualization of action probabilities."""
    plt.figure(figsize=(10, 3))
    sns.heatmap(action_probs.reshape(1, -1), annot=True, fmt='.2f', 
                xticklabels=class_names, yticklabels=False, cmap='YlOrRd')
    plt.title('Action Probabilities')
    plt.tight_layout()
    return plt.gcf()

def visualize_annotations(image, anns, action_probs=None, class_names=None, visualize_bboxes=True, 
                        visualize_keypoints=True, visualize_segmentations=True):
    """Enhanced visualization with action probabilities and hand skeleton."""
    vis_img = image.copy()
    
    # Draw bounding boxes
    if visualize_bboxes:
        for ann in anns:
            bbox = ann.get('bbox', None)
            if bbox is not None and np.any(bbox):
                x, y, w, h = map(int, bbox.tolist())  # Convert bbox to list before mapping to int
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw hand skeletons with enhanced visualization
    if visualize_keypoints:
        for ann in anns:
            keypoints = ann.get('keypoints', None)
            if keypoints is not None and len(keypoints) > 0:
                draw_hand_skeleton(vis_img, keypoints, FINGER_CONNECTIONS)
    
    # Draw segmentation
    if visualize_segmentations:
        for ann in anns:
            segmentation = ann.get('segmentation', None)
            if segmentation:
                for seg in segmentation:
                    points = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(vis_img, [points], isClosed=True, color=(0, 255, 255), thickness=2)
    
    # Add action probability heatmap if available
    if action_probs is not None and class_names is not None:
        heatmap = visualize_action_heatmap(vis_img, action_probs, class_names)
        
        # Convert heatmap to image
        heatmap.canvas.draw()
        heatmap_img = np.frombuffer(heatmap.canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_img = heatmap_img.reshape(heatmap.canvas.get_width_height()[::-1] + (3,))
        
        # Stack visualization vertically
        vis_img = np.vstack([vis_img, heatmap_img])
        plt.close()
        
    return vis_img

def process_folder(input_folder, annotations_file, output_folder, class_names=None):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)

    # Load class names from dataset.yml if not provided
    if class_names is None:
        with open(os.path.join(os.path.dirname(annotations_file), 'dataset.yml'), 'r') as f:
            config = yaml.safe_load(f)
            class_names = list(config['action_mapping'].keys())

    # Iterate over all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Could not load image {filename}")
                continue
            image_id = int(filename.split('.')[0].split('_')[-1])
            annotations = [annos for annos in annotations_data['annotations'] if annos['image_id']==image_id]
            action = annotations[0].get('action', 'dummy')
            cv2.putText(image, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Generate dummy action probabilities for visualization
            action_probs = np.zeros(len(class_names))
            action_probs[annotations[0].get('category_id', 0)] = 1.0

            # Enhanced visualization
            annotated_image = visualize_annotations(image, annotations, 
                                                  action_probs=action_probs,
                                                  class_names=class_names)

            # Save the annotated image in the output folder
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, annotated_image)
            print(f"Processed {filename} and saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description='Process a folder of images and save annotated frames.')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing PNG frames',default="data")
    parser.add_argument('--annotations_file', type=str, help='Path to the JSON file containing annotations',default="data")
    parser.add_argument('--output_folder', type=str, help='Path to the folder where processed frames will be saved',default="output_folder")
    args = parser.parse_args()

    # Process the folder and save annotated frames
    process_folder(args.input_folder, args.annotations_file, args.output_folder)

if __name__ == '__main__':
    main()
