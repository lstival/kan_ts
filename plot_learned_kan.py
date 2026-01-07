import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src.models.lightning_kan import KANContrastiveLightning

def find_latest_checkpoint(base_dir="outputs/kan-time-series"):
    """Finds the latest checkpoint in the specified directory."""
    checkpoints = list(Path(base_dir).rglob("*.ckpt"))
    if not checkpoints:
        # Try lightning_logs
        checkpoints = list(Path("outputs/lightning_logs").rglob("*.ckpt"))
    
    if not checkpoints:
        return None
        
    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return str(checkpoints[0])

def plot_kan_activations(model, layer_idx=0, num_samples=100, max_plots=5):
    """
    Plots the learned univariate activation functions for a specific KAN layer.
    """
    kan = model.encoder.kan
    if layer_idx >= len(kan.layers):
        print(f"Layer index {layer_idx} out of range.")
        return

    layer = kan.layers[layer_idx]
    in_features = layer.in_features
    out_features = layer.out_features
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Create input range [-1, 1] as defined by grid_range commonly
    x = torch.linspace(-1, 1, num_samples).unsqueeze(1).repeat(1, in_features).to(device)
    
    with torch.no_grad():
        # Get base activation output: base_weight is (out, in)
        # base_activation(x) is (num_samples, in)
        base_act = layer.base_activation(x) 
        
        # Get spline bases: (num_samples, in, grid_size + spline_order)
        spline_bases = layer.b_splines(x)
        
        # We want to see phi_{i,j}(x) = w_b * act(x) + sum(c_k * B_k(x))
        # Since efficient_kan uses F.linear for speed, we need to reconstruct for plotting
        
        # Pick a few random input/output pairs to plot
        fig, axes = plt.subplots(1, min(max_plots, in_features), figsize=(15, 3))
        if max_plots == 1 or in_features == 1:
            axes = [axes]
            
        for plot_idx in range(len(axes)):
            i = plot_idx # Input feature index
            j = 0        # Output feature index (just look at the first output for simplicity)
            
            # Base part for this pair
            y_base = layer.base_weight[j, i] * base_act[:, i]
            
            # Spline part for this pair
            # scaled_spline_weight is (out, in, coeff)
            coeffs = layer.scaled_spline_weight[j, i] # (coeff,)
            y_spline = (spline_bases[:, i, :] * coeffs).sum(dim=-1)
            
            y_total = y_base + y_spline
            
            axes[plot_idx].plot(x[:, i].cpu().numpy(), y_total.cpu().numpy(), label='Total', color='black', linewidth=2)
            axes[plot_idx].plot(x[:, i].cpu().numpy(), y_base.cpu().numpy(), '--', label='Base', alpha=0.5)
            axes[plot_idx].plot(x[:, i].cpu().numpy(), y_spline.cpu().numpy(), ':', label='Spline', alpha=0.5)
            axes[plot_idx].set_title(f"In {i} -> Out {j}")
            axes[plot_idx].grid(True)
            if plot_idx == 0:
                axes[plot_idx].legend()

    plt.suptitle(f"KAN Layer {layer_idx} Learned Activations")
    plt.tight_layout()
    output_path = f"outputs/kan_plots/layer_{layer_idx}_equations.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoints found. Please train the model first.")
        return
        
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load model 
    # Note: KANContrastiveLightning needs hyperparams to initialize. 
    # load_from_checkpoint usually handles this if they were saved correctly.
    try:
        model = KANContrastiveLightning.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Manual workaround if load_from_checkpoint fails due to missing hyperparams or path issues
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # You might need to adjust this depending on your config
        model = KANContrastiveLightning(input_dim=384) 
        model.load_state_dict(checkpoint['state_dict'])
        
    model.eval()
    print("Model loaded successfully.")
    
    # Plot first layer equations
    plot_kan_activations(model, layer_idx=0, max_plots=5)
    
    # If there are more layers, plot them too
    num_layers = len(model.encoder.kan.layers)
    if num_layers > 1:
        plot_kan_activations(model, layer_idx=num_layers-1, max_plots=5)

if __name__ == "__main__":
    main()
