import torch
import numpy as np
from typing import Tuple, Union

class TimeSeriesScaler:
    """Simple min-max scaler for time series data."""
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
        self.min_val = None
        self.max_val = None

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            self.min_val = x.min()
            self.max_val = x.max()
        else:
            self.min_val = x.min()
            self.max_val = x.max()
        
        denom = (self.max_val - self.min_val) + self.epsilon
        return (x - self.min_val) / denom

    def transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        denom = (self.max_val - self.min_val) + self.epsilon
        return (x - self.min_val) / denom

    def inverse_transform(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        denom = (self.max_val - self.min_val) + self.epsilon
        return x * denom + self.min_val
