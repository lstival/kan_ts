import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.data.chronos_dataset import ChronosDataset
from typing import List, Optional

class ChronosDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_names: List[str], 
        batch_size: int = 32, 
        num_workers: int = 0,
        normalize: bool = True,
        transform: Optional[callable] = None,
        contrastive: bool = False,
        forecast: bool = False,
        context_length: int = 384,
        prediction_length: int = 96
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.transform = transform
        self.contrastive = contrastive
        self.forecast = forecast
        self.context_length = context_length
        self.prediction_length = prediction_length

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Create separate datasets for train and val to handle different windowing logic
            self.train_ds = ChronosDataset(
                self.dataset_names, 
                split="train", 
                normalize=self.normalize,
                context_length=self.context_length,
                prediction_length=self.prediction_length
            )
            self.train_ds.contrastive = self.contrastive
            self.train_ds.forecast = self.forecast
            self.train_ds.transform = self.transform

            self.val_ds = ChronosDataset(
                self.dataset_names, 
                split="train", # We still use the 'train' split of the HF dataset but change our internal mode
                normalize=self.normalize,
                context_length=self.context_length,
                prediction_length=self.prediction_length
            )
            self.val_ds.split = "val" # Set to val to use last window
            self.val_ds.contrastive = self.contrastive
            self.val_ds.transform = self.transform
            self.val_ds.forecast = self.forecast
            
            # Split the indices
            full_len = len(self.train_ds)
            train_size = int(0.9 * full_len)
            val_size = full_len - train_size
            
            indices = torch.randperm(full_len, generator=torch.Generator().manual_seed(42))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            from torch.utils.data import Subset
            self.train_ds = Subset(self.train_ds, train_indices)
            self.val_ds = Subset(self.val_ds, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn if self.forecast else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn if self.forecast else None
        )

    def _collate_fn(self, batch):
        """Custom collate to handle scalers in forecast mode."""
        contexts = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        scalers = [item[2] for item in batch]
        return contexts, targets, scalers
