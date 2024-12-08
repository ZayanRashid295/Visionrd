import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import yaml
import json

def get_arguments() -> argparse.Namespace:
    """
    Parse all the arguments from the command line interface.
    Return a list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert pred and gt list to images.")
    parser.add_argument(
        "frames_dir",
        type=str,
        help="Path to the directory containing folders of video frames",
        default="data/features/gtea_png/png"
    )
    parser.add_argument(
        "labels_path",
        type=str,
        help="Path to the file containing dataset labels",
        default="data/features/gtea_png/labels.txt"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output images",
        default="output"
    )
    return parser.parse_args()

def load_action_dict(yaml_path):
    """
    Load the action labels into a dictionary (id to class and vice versa) from the dataset.yml file.
    """
    with open(yaml_path, "r", encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    
    action_mapping = configs["action_mapping"]
    id2class_map = {v: k for k, v in action_mapping.items()}
    class2id_map = action_mapping

    return id2class_map, class2id_map

def parse_frame_directories(frames_dir):
    """
    Get a list of frame directory names from the given parent directory.
    Each directory represents a set of frames for one video.
    """
    frame_directories = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
    return frame_directories

def main() -> None:
    args = get_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the action dictionaries from the dataset.yml file
    id2class_map, class2id_map = load_action_dict("/home/user/AI-Hackathon24/dataloader/dataset.yml")

    # Parse frame directories in the provided parent directory
    frame_dirs = parse_frame_directories(args.frames_dir)
    print(f"Found {len(frame_dirs)} frame directories in {args.frames_dir}.")

    # Load labels from the single labels.txt file
    labels_file = args.labels_path
    if not os.path.exists(labels_file):
        print(f"Labels file {labels_file} not found.")
        return

    # Read mappings into a dictionary
    frame_label_dict = {}
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
        for label in labels_data:
            video_name = label.get("name")
            action_label = label.get("name")
            frame_range = label.get("id")
            start_frame, end_frame = 0, 100  # Assuming a default range, update as needed
            if video_name not in frame_label_dict:
                frame_label_dict[video_name] = []
            frame_label_dict[video_name].append((start_frame, end_frame, action_label))

    # Loop over the frame directories and process them
    for frame_dir in tqdm(frame_dirs, desc="Processing frame directories"):
        frame_dir_path = os.path.join(args.frames_dir, frame_dir)
        video_name = frame_dir

        # Get labels for this video
        mappings = frame_label_dict.get(video_name, [])
        if not mappings:
            print(f"No labels found for video {video_name}. Skipping...")
            continue

        # Load all frame files in the directory and sort them
        frame_files = [f for f in os.listdir(frame_dir_path) if f.endswith(('.jpg', '.png'))]
        frame_files.sort()  # Sorting ensures the frames are processed in the correct order

        # Process each frame or frame range, extract actions, and output images
        for start_frame, end_frame, action_label in mappings:
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < len(frame_files):
                    frame_file = frame_files[frame_idx]
                    frame_path = os.path.join(frame_dir_path, frame_file)

                    # Load the frame
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"Failed to read {frame_path}. Skipping...")
                        continue

                    # Overlay action label onto the frame (optional: customize the text overlay)
                    overlay_text = f"Action: {action_label}"
                    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save the frame with the overlaid action label
                    output_image_path = os.path.join(args.output_dir, f"{video_name}_{frame_idx}_{action_label}.png")
                    cv2.imwrite(output_image_path, frame)

                    print(f"Generated image for {video_name}, frame {frame_idx} with action {action_label}.")

if __name__ == "__main__":
    main()