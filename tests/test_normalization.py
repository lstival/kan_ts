import torch
import numpy as np
import pytest
from src.utils.normalization import TimeSeriesScaler

def test_scaler_numpy_fit_transform():
    scaler = TimeSeriesScaler()
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaled = scaler.fit_transform(data)
    
    assert np.allclose(scaled.min(), 0.0)
    assert np.allclose(scaled.max(), 1.0)
    assert np.allclose(scaled[2], 0.5)

def test_scaler_torch_fit_transform():
    scaler = TimeSeriesScaler()
    data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    scaled = scaler.fit_transform(data)
    
    assert torch.allclose(scaled.min(), torch.tensor(0.0))
    assert torch.allclose(scaled.max(), torch.tensor(1.0))
    assert torch.allclose(scaled[2], torch.tensor(0.5))

def test_scaler_inverse_transform():
    scaler = TimeSeriesScaler()
    data = torch.tensor([10.0, 25.0, 50.0])
    scaled = scaler.fit_transform(data)
    reverted = scaler.inverse_transform(scaled)
    
    assert torch.allclose(data, reverted)

def test_scaler_with_nans():
    scaler = TimeSeriesScaler()
    data = torch.tensor([10.0, float('nan'), 30.0])
    scaled = scaler.fit_transform(data)
    
    # min should be 10, max should be 30
    # (10-10)/(30-10) = 0
    # (30-10)/(30-10) = 1
    assert torch.allclose(scaled[0], torch.tensor(0.0))
    assert torch.isnan(scaled[1])
    assert torch.allclose(scaled[2], torch.tensor(1.0))

def test_scaler_not_fitted_raises():
    scaler = TimeSeriesScaler()
    with pytest.raises(ValueError, match="Scaler has not been fitted yet"):
        scaler.transform(torch.tensor([1.0]))

def test_scaler_constant_values():
    scaler = TimeSeriesScaler()
    data = torch.tensor([10.0, 10.0, 10.0])
    scaled = scaler.fit_transform(data)
    # denominator should be epsilon, scaled should be 0
    assert torch.all(scaled < 1e-4)
