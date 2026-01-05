import torch
import numpy as np
from typing import Tuple, Union

class TimeSeriesScaler:
    """
    Mean scaler for time series data, following Chronos implementation.
    Scales by the mean of absolute values.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.scale = None

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            # Use numpy for calculation
            abs_x = np.abs(x)
            if np.all(np.isnan(abs_x)):
                self.scale = 1.0
            else:
                self.scale = float(np.nanmean(abs_x))
        else:
            # Use torch for calculation
            abs_x = torch.abs(x)
            if torch.all(torch.isnan(abs_x)):
                self.scale = 1.0
            else:
                # Use nanmean if available, else fallback to nansum/count
                if hasattr(torch, 'nanmean'):
                    self.scale = float(torch.nanmean(abs_x))
                else:
                    mask = ~torch.isnan(abs_x)
                    count = torch.sum(mask)
                    if count == 0:
                        self.scale = 1.0
                    else:
                        self.scale = float(torch.nansum(abs_x) / count)
        
        if self.scale < self.epsilon:
            self.scale = 1.0
            
        scaled = x / self.scale
        if isinstance(scaled, torch.Tensor):
            return torch.clamp(scaled, -10.0, 10.0)
        return np.clip(scaled, -10.0, 10.0)

    def transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.scale is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        scaled = x / self.scale
        if isinstance(scaled, torch.Tensor):
            return torch.clamp(scaled, -10.0, 10.0)
        return np.clip(scaled, -10.0, 10.0)

    def inverse_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.scale is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        return x * self.scale
        
        denom = (self.max_val - self.min_val) + self.epsilon
        return x * denom + self.min_val
