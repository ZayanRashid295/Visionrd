import random

def generate_sample_labels(file_path, num_samples=100, num_classes=10):
    """
    Generate a sample video.txt file with random labels.

    Args:
        file_path (str): Path to save the generated video.txt file.
        num_samples (int): Number of samples to generate labels for.
        num_classes (int): Number of distinct classes.
    """
    labels = [f"class_{random.randint(0, num_classes - 1)}" for _ in range(num_samples)]
    
    with open(file_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"Sample labels file saved at {file_path}")

if __name__ == "__main__":
    generate_sample_labels("video.txt", num_samples=100)