import torch
import numpy as np
from torch.utils.data import Dataset
import datasets
from typing import List, Optional, Union

class ChronosDataset(Dataset):
    """
    Chronos dataset loader following the original implementation logic.
    Loads datasets from autogluon/chronos_datasets on Hugging Face.
    """
    def __init__(
        self, 
        dataset_names: List[str], 
        split: str = "train", 
        repo_id: str = "autogluon/chronos_datasets",
        normalize: bool = True,
        context_length: int = 384,
        prediction_length: int = 96,
        probabilities: Optional[List[float]] = None,
        transform: Optional[callable] = None
    ):
        self.datasets = []
        self.split = split
        for name in dataset_names:
            try:
                ds = datasets.load_dataset(repo_id, name, split=split)
                if "target" in ds.column_names:
                    ds = ds.select_columns(["target"])
                    # Unify the target column type to float32 to avoid alignment issues
                    ds = ds.cast_column("target", datasets.Sequence(datasets.Value("float32")))
                    self.datasets.append(ds)
            except Exception as e:
                print(f"Error loading dataset {name}: {e}")
                print(f"TIP: If you see 'Feature type List not found', try updating the datasets library: pip install --upgrade datasets")
        
        if not self.datasets:
            raise ValueError("No datasets loaded successfully.")

        self.combined = datasets.concatenate_datasets(self.datasets)
        self.combined.set_format("numpy")
        
        self.normalize = normalize
        self.contrastive = False 
        self.forecast = False
        self.transform = transform
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Probabilities for mixing (if we were using an iterable dataset)
        # For this Map-style dataset, we just use the combined dataset.
        self.probabilities = probabilities

    def __len__(self):
        return len(self.combined)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        entry = self.combined[idx]
        full_series = entry["target"]
        
        # Convert to torch and handle NaNs/Infs
        full_series = torch.from_numpy(full_series).float()
        full_series = torch.nan_to_num(full_series, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.forecast:
            return self._get_forecast_sample(full_series)

        # Ensure fixed context_length for contrastive or default behavior
        if len(full_series) < self.context_length:
            # Pad with zeros if too short
            pad_len = self.context_length - len(full_series)
            full_series = torch.cat([torch.zeros(pad_len), full_series])
            start_idx = 0
        else:
            if self.split == "train":
                # Randomly sample a window for training
                max_start = len(full_series) - self.context_length
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
            else:
                # Take the last window for validation/test
                start_idx = len(full_series) - self.context_length
        
        series_window = full_series[start_idx : start_idx + self.context_length]

        if self.contrastive:
            v1, v2 = self._get_contrastive_views(series_window)
            if self.transform:
                v1 = self.transform(v1)
                v2 = self.transform(v2)
            return v1, v2

        # Default behavior
        target = series_window
        scaler = None
        if self.normalize:
            from src.utils.normalization import TimeSeriesScaler
            scaler = TimeSeriesScaler()
            target = scaler.fit_transform(target)
        
        if self.transform:
            target = self.transform(target)
            
        if self.normalize:
            return target, scaler
        return target

    def _get_forecast_sample(self, x: torch.Tensor):
        """
        Splits series into context and target for forecasting.
        Follows Chronos logic: context_length and prediction_length.
        """
        total_len = self.context_length + self.prediction_length
        
        if len(x) < total_len:
            # Pad with zeros if too short (Chronos uses left-padding)
            pad_len = total_len - len(x)
            x = torch.cat([torch.zeros(pad_len), x])
            start_idx = 0
        else:
            if self.split == "train":
                # Randomly sample a window for training
                max_start = len(x) - total_len
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
            else:
                # Take the last window for validation/test
                start_idx = len(x) - total_len
        
        window = x[start_idx : start_idx + total_len]
        context = window[:self.context_length]
        target = window[self.context_length:]

        from src.utils.normalization import TimeSeriesScaler
        scaler = TimeSeriesScaler()
        
        if self.normalize:
            # Mean scaling as per Chronos
            context = scaler.fit_transform(context)
            target = scaler.transform(target)
        
        return context, target, scaler

    def _get_contrastive_views(self, x: torch.Tensor):
        """Generates two augmented views of the same time series."""
        # View 1: Original with normalization
        v1 = x.clone()
        # View 2: Add slight noise
        v2 = x + torch.randn_like(x) * 0.01

        views = []
        for v in [v1, v2]:
            if self.normalize:
                from src.utils.normalization import TimeSeriesScaler
                v = TimeSeriesScaler().fit_transform(v)
            views.append(v)
        
        return views[0], views[1]
