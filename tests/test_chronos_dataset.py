import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.chronos_dataset import ChronosDataset
from src.utils.normalization import TimeSeriesScaler

def test_chronos_loading_and_normalization():
    # Use a small dataset for testing
    dataset_names = ["exchange_rate"]
    print(f"Loading dataset: {dataset_names}...")
    ds = ChronosDataset(dataset_names, split="train", normalize=True)
    
    assert len(ds) > 0
    
    # Get a sample
    target, scaler = ds[0]
    print(f"Sample 0 shape: {target.shape}")
    print(f"Normalized sample (first 5 values): {target[:5].tolist()}")
    
    assert isinstance(target, torch.Tensor)
    assert isinstance(scaler, TimeSeriesScaler)
    
    # Check normalization (min should be 0, max should be 1)
    assert torch.allclose(target.min(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(target.max(), torch.tensor(1.0), atol=1e-5)
    
    # Check reverse normalization
    reversed_target = scaler.inverse_transform(target)
    print(f"Denormalized sample (first 5 values): {reversed_target[:5].tolist()}")
    
    # Original data from dataset (without normalization)
    ds_no_norm = ChronosDataset(dataset_names, split="train", normalize=False)
    original_target = ds_no_norm[0]
    print(f"Original sample (first 5 values): {original_target[:5].tolist()}")
    
    assert torch.allclose(reversed_target, original_target, atol=1e-5)
    
    # Visualization
    print("Generating plots...")
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(original_target.numpy(), label="Original", color="blue", alpha=0.7)
    plt.title("Original Time Series")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(target.numpy(), label="Normalized", color="green", alpha=0.7)
    plt.title("Normalized Time Series (Scale: [0, 1])")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(reversed_target.numpy(), label="Denormalized", color="red", linestyle="--", alpha=0.7)
    plt.title("Denormalized Time Series (should match original)")
    plt.legend()
    
    plt.tight_layout()
    plot_path = "time_series_comparison.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    print("Test passed: Loading, normalization, and reverse normalization are correct.")

if __name__ == "__main__":
    test_chronos_loading_and_normalization()
