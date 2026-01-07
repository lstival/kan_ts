import torch
import pytest
from src.data.datamodule import ChronosDataModule

@pytest.mark.parametrize("forecast", [True, False])
def test_datamodule_setup(forecast):
    batch_size = 4
    dm = ChronosDataModule(
        dataset_names=["exchange_rate"],
        batch_size=batch_size,
        forecast=forecast,
        context_length=64,
        prediction_length=16
    )
    
    dm.setup("fit")
    
    assert dm.train_ds is not None
    assert dm.val_ds is not None
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    if forecast:
        # Expected: context, target, scalers
        assert len(batch) == 3
        contexts, targets, scalers = batch
        assert contexts.shape == (batch_size, 64)
        assert targets.shape == (batch_size, 16)
        assert len(scalers) == batch_size
    else:
        # Default ChronosDataset returns (target, scaler)
        # But if not forecast and not contrastive, it might just be the window.
        # Actually in non-forecast mode, it likely returns (sample, scaler) 
        # but the datamodule doesn't have a custom collate for that.
        assert batch[0].shape[0] == batch_size

def test_datamodule_contrastive_setup():
    dm = ChronosDataModule(
        dataset_names=["exchange_rate"],
        batch_size=4,
        contrastive=True,
        forecast=False,
        context_length=64
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    
    # In contrastive mode, it should return two views
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    view1, view2 = batch
    assert view1.shape == (4, 64)
    assert view2.shape == (4, 64)
