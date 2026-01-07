import torch
import pytest
from src.models.kan_contrastive import KANEncoder
from src.models.forecast_model import ForecastHead, ForecastLightning

def test_forecast_head_forward():
    input_dim = 128
    hidden_dim = 64
    projection_dim = 32
    prediction_length = 10
    batch_size = 4
    
    encoder = KANEncoder(input_dim=input_dim, hidden_dim=hidden_dim, projection_dim=projection_dim)
    head = ForecastHead(encoder=encoder, projection_dim=projection_dim, prediction_length=prediction_length)
    
    x = torch.randn(batch_size, input_dim)
    out = head(x)
    
    assert out.shape == (batch_size, prediction_length)
    assert not torch.isnan(out).any()

def test_forecast_lightning_step():
    input_dim = 64
    projection_dim = 16
    prediction_length = 8
    batch_size = 2
    
    encoder = KANEncoder(input_dim=input_dim, projection_dim=projection_dim)
    model = ForecastLightning(
        encoder=encoder, 
        projection_dim=projection_dim, 
        prediction_length=prediction_length
    )
    
    # Mock batch: (contexts, targets, scalers)
    from src.utils.normalization import TimeSeriesScaler
    contexts = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, prediction_length)
    scalers = [TimeSeriesScaler() for _ in range(batch_size)]
    for s in scalers: # Fit scalers so inverse_transform works
        s.fit_transform(torch.randn(10))
        
    batch = (contexts, targets, scalers)
    
    # Test training step
    loss = model.training_step(batch, 0)
    assert loss >= 0
    assert not torch.isnan(loss)
    
    # Test validation step
    val_loss = model.validation_step(batch, 0)
    # validation_step returns None or log dict in newer PL, 
    # but here it seems to return loss based on the code provided in read_file
    assert val_loss is not None

def test_forecast_lightning_freeze():
    encoder1 = KANEncoder(input_dim=10, projection_dim=5)
    model = ForecastLightning(encoder=encoder1, projection_dim=5, prediction_length=2, freeze_encoder=True)
    
    for param in model.model.encoder.parameters():
        assert not param.requires_grad
    
    encoder2 = KANEncoder(input_dim=10, projection_dim=5)
    model_unfrozen = ForecastLightning(encoder=encoder2, projection_dim=5, prediction_length=2, freeze_encoder=False)
    for param in model_unfrozen.model.encoder.parameters():
        assert param.requires_grad
