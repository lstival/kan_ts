import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.chronos_dataset import ChronosDataset
from src.utils.wavelet_transform import WaveletTransform

def test_wavelet_transformation():
    dataset_names = ["exchange_rate"]
    image_size = 224
    print(f"Loading dataset with Wavelet Transform (Square {image_size}x{image_size}): {dataset_names}...")
    
    # Define the transform to produce square images
    wavelet_transform = WaveletTransform(wavelet='mexh', image_size=image_size)
    
    # Initialize dataset with the transform
    ds = ChronosDataset(dataset_names, split="train", normalize=True, transform=wavelet_transform)
    
    # Get a sample (already transformed to 2D square image)
    target, _ = ds[0]
    
    print(f"Transformed sample shape: {target.shape} (Should be {image_size}x{image_size})")
    assert target.shape == (image_size, image_size)
    
    # Visualization
    plt.figure(figsize=(8, 8))
    
    plt.imshow(target.numpy(), aspect='equal', cmap='jet')
    plt.title(f"Square Wavelet Scalogram ({image_size}x{image_size})")
    plt.ylabel("Scale")
    plt.xlabel("Time (Resampled)")
    plt.colorbar(label="Magnitude")
    
    plt.tight_layout()
    plot_path = "wavelet_transformation.png"
    plt.savefig(plot_path)
    print(f"Wavelet plot saved to {plot_path}")

if __name__ == "__main__":
    test_wavelet_transformation()
