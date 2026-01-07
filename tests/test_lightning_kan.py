import torch
import pytest
from src.models.lightning_kan import KANContrastiveLightning

def test_kan_contrastive_lightning_1d():
    input_dim = 120
    hidden_dim = 64
    projection_dim = 32
    batch_size = 4
    
    model = KANContrastiveLightning(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim
    )
    
    # Mock batch: (view1, view2)
    x1 = torch.randn(batch_size, input_dim)
    x2 = torch.randn(batch_size, input_dim)
    batch = (x1, x2)
    
    loss = model.training_step(batch, 0)
    assert loss > 0
    assert not torch.isnan(loss)

def test_kan_contrastive_lightning_2d():
    image_size = 32
    hidden_dim = 64
    projection_dim = 32
    batch_size = 2
    
    model = KANContrastiveLightning(
        image_size=image_size,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim
    )
    
    # Mock batch: (view1, view2)
    x1 = torch.randn(batch_size, 1, image_size, image_size)
    x2 = torch.randn(batch_size, 1, image_size, image_size)
    batch = (x1, x2)
    
    loss = model.training_step(batch, 0)
    assert loss > 0
    assert not torch.isnan(loss)

def test_kan_contrastive_forward():
    input_dim = 100
    model = KANContrastiveLightning(input_dim=input_dim)
    x = torch.randn(8, input_dim)
    z = model(x)
    assert z.shape == (8, 64) # 64 is default projection_dim
