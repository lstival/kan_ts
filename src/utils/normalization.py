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
        
        denom = (self.max_val - self.min_val) + self.epsilon
        scaled = (x - self.min_val) / denom
        return scaled

    def transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        denom = (self.max_val - self.min_val) + self.epsilon
        scaled = (x - self.min_val) / denom
        return scaled

    def inverse_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        denom = (self.max_val - self.min_val) + self.epsilon
        return x * denom + self.min_val
