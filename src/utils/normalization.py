import torch
import numpy as np
from typing import Tuple, Union

class TimeSeriesScaler:
    """
    Min-Max scaler for time series data.
    Scales data to [0, 1] range.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.min_val = None
        self.max_val = None

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            self.min_val = np.nanmin(x)
            self.max_val = np.nanmax(x)
        else:
            # Handle torch tensors
            if torch.any(torch.isnan(x)):
                valid_x = x[~torch.isnan(x)]
                if valid_x.numel() == 0:
                    self.min_val = torch.tensor(0.0)
                    self.max_val = torch.tensor(1.0)
                else:
                    self.min_val = valid_x.min()
                    self.max_val = valid_x.max()
            else:
                self.min_val = x.min()
                self.max_val = x.max()
        
        diff = (self.max_val - self.min_val)
        # Avoid division by zero/epsilon for constant signals
        # If signal is nearly constant, we don't scale it (denom=1.0)
        # and we set min_val such that the output is just the original values
        if isinstance(diff, torch.Tensor):
            is_constant = diff < self.epsilon
            denom = torch.where(is_constant, torch.ones_like(diff), diff)
            if is_constant.any():
                # For constant parts, we don't want to subtract min_val if we aren't scaling
                # Actually, subtracting min_val is fine, it just makes it 0.
                pass
        else:
            denom = diff if diff >= self.epsilon else 1.0
            
        scaled = (x - self.min_val) / denom
        
        # Robustness: clamp scaled values to prevent extreme values from outliers in target
        if isinstance(scaled, torch.Tensor):
            scaled = torch.clamp(scaled, min=-10.0, max=10.0)
        else:
            scaled = np.clip(scaled, -10.0, 10.0)
            
        return scaled

    def transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        diff = (self.max_val - self.min_val)
        if isinstance(diff, torch.Tensor):
            denom = torch.where(diff < self.epsilon, torch.ones_like(diff), diff)
        else:
            denom = diff if diff >= self.epsilon else 1.0
            
        scaled = (x - self.min_val) / denom
        
        # Robustness: clamp scaled values
        if isinstance(scaled, torch.Tensor):
            scaled = torch.clamp(scaled, min=-10.0, max=10.0)
        else:
            scaled = np.clip(scaled, -10.0, 10.0)
            
        return scaled

    def inverse_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        diff = (self.max_val - self.min_val)
        if isinstance(diff, torch.Tensor):
            denom = torch.where(diff < self.epsilon, torch.ones_like(diff), diff)
        else:
            denom = diff if diff >= self.epsilon else 1.0
            
        return x * denom + self.min_val
