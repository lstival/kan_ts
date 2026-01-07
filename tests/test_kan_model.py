import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.models.kan_contrastive import KANEncoder, InfoNCELoss

def test_kan_model_shapes_and_nans():
    print("Testing KAN Encoder 1D...")
    
    input_dim = 384
    batch_size = 8
    projection_dim = 64
    
    model = KANEncoder(input_dim=input_dim, projection_dim=projection_dim)
    
    # Create dummy input (batch, input_dim)
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, projection_dim), f"Expected shape {(batch_size, projection_dim)}, got {output.shape}"
    
    # Check for NaNs
    has_nan = torch.isnan(output).any().item()
    print(f"Output contains NaNs: {has_nan}")
    assert not has_nan, "Model output contains NaN values!"
    
    print("Shape and NaN checks passed.")

def test_infonce_loss():
    print("\nTesting InfoNCE Loss...")
    
    batch_size = 4
    dim = 64
    criterion = InfoNCELoss(temperature=0.07)
    
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    
    loss = criterion(z1, z2)
    
    print(f"Loss value: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive"
    
    print("InfoNCE Loss check passed.")

if __name__ == "__main__":
    test_kan_model_shapes_and_nans()
    test_infonce_loss()
