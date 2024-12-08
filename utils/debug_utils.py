
import torch

def print_tensor_details(inputs, keypoints, bboxes, segmentation, labels):
    """Print detailed information about tensor shapes and types"""
    print("\n--- Tensor Shape Diagnostics ---")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Inputs dtype: {inputs.dtype}")
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Keypoints dtype: {keypoints.dtype}")
    print(f"Bboxes shape: {bboxes.shape}")
    print(f"Bboxes dtype: {bboxes.dtype}")
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Segmentation dtype: {segmentation.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print("-----------------------------\n")

def validate_tensor_shapes(inputs, keypoints, bboxes, segmentation, labels):
    """Validate tensor shapes and types"""
    assert inputs.dim() == 4, f"Expected 4D input tensor, got shape {inputs.shape}"
    assert keypoints.dim() == 3, f"Expected 3D keypoints tensor, got shape {keypoints.shape}"
    assert bboxes.dim() == 3, f"Expected 3D bboxes tensor, got shape {bboxes.shape}"
    assert segmentation.dim() == 4, f"Expected 4D segmentation tensor, got shape {segmentation.shape}"
    assert labels.dim() == 1, f"Expected 1D labels tensor, got shape {labels.shape}"