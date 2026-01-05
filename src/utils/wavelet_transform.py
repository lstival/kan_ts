import numpy as np
import pywt
import torch
from typing import Union, Optional
from scipy.signal import resample

def compute_wavelet_transform(
    x: Union[torch.Tensor, np.ndarray], 
    wavelet: str = 'mexh', 
    image_size: int = 64
) -> torch.Tensor:
    """
    Computes the Continuous Wavelet Transform (CWT) of a time series
    and produces a square image of shape (image_size, image_size).
    
    Args:
        x: Input time series (1D)
        wavelet: Wavelet name (default: 'mexh' - Mexican Hat)
        image_size: The target size for the square image (N x N)
        
    Returns:
        2D tensor of shape (image_size, image_size)
    """
    if isinstance(x, torch.Tensor):
        x_np = x.numpy()
    else:
        x_np = x

    # 1. Resample the input signal to the target image size to ensure the time dimension matches
    if len(x_np) != image_size:
        x_np = resample(x_np, image_size)

    # 2. Use a number of scales equal to the target image size to ensure the scale dimension matches
    scales = np.arange(1, image_size + 1)

    # 3. Compute CWT - this naturally produces a (image_size, image_size) array
    coefficients, _ = pywt.cwt(x_np, scales, wavelet)
    
    # 4. Convert to absolute values (magnitude) and then to tensor
    scalogram = np.abs(coefficients)
    
    # 5. Normalize scalogram to [0, 1] for better training stability
    s_min, s_max = scalogram.min(), scalogram.max()
    if s_max > s_min:
        scalogram = (scalogram - s_min) / (s_max - s_min)
    else:
        scalogram = np.zeros_like(scalogram)
        
    return torch.from_numpy(scalogram).float()

class WaveletTransform:
    """Transform class to produce square images from time series."""
    def __init__(self, wavelet: str = 'mexh', image_size: int = 64):
        self.wavelet = wavelet
        self.image_size = image_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return compute_wavelet_transform(x, self.wavelet, self.image_size)
