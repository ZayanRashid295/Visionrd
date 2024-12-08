import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
import argparse
import numpy as np
from models.custom_model import CustomModel

def load_model(model_path, num_classes):
    model = CustomModel('resnet50', num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def recognize_actions(video_path, model, output_path, action_labels):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        input_frame = cv2.resize(frame, (224, 224))
        input_frame = input_frame.astype(np.float32) / 255.0
        input_frame = np.transpose(input_frame, (2, 0, 1))
        input_frame = torch.tensor(input_frame).unsqueeze(0)

        # Predict action
        with torch.no_grad():
            output = model(input_frame)
            action_idx = torch.argmax(output, dim=1).item()
            action_label = action_labels[action_idx]

        # Overlay action label on frame
        cv2.putText(frame, action_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize actions in video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of action classes")
    args = parser.parse_args()

    action_labels = ['take', 'pour', 'shake', 'open', 'close', 'scoop', 'stir', 'put', 'fold', 'spread']
    model = load_model(args.model_path, args.num_classes)
    recognize_actions(args.video_path, model, args.output_path, action_labels)
