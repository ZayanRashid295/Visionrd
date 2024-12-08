import numpy as np

def generate_sample_input(file_path, num_samples=100, feature_dim=2048):
    """
    Generate a sample input.npy file with random data.

    Args:
        file_path (str): Path to save the generated input.npy file.
        num_samples (int): Number of samples to generate.
        feature_dim (int): Dimensionality of each feature vector.
    """
    # Generate random data
    data = np.random.rand(num_samples, feature_dim)
    
    # Save to .npy file
    np.save(file_path, data)
    print(f"Sample input file saved at {file_path}")

if __name__ == "__main__":
    generate_sample_input("input.npy", num_samples=100)  # Ensure num_samples=100