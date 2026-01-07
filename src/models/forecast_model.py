import torch
import torch.nn as nn
import pytorch_lightning as pl

class ForecastHead(nn.Module):
    """
    Forecast head that uses a pre-trained KAN encoder to predict future values.
    """
    def __init__(
        self, 
        encoder: nn.Module, 
        projection_dim: int, 
        prediction_length: int
    ):
        super().__init__()
        self.encoder = encoder
        # Add BatchNorm to stabilize features from the encoder
        self.bn = nn.BatchNorm1d(projection_dim)
        # Simple linear head for forecasting
        self.head = nn.Linear(projection_dim, prediction_length)
        
        # Robust initialization
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time series or images (batch, ...)
        Returns:
            Predictions (batch, prediction_length)
        """
        features = self.encoder(x)
        # Normalize features before the head
        features = self.bn(features)
        predictions = self.head(features)
        return predictions

class ForecastLightning(pl.LightningModule):
    def __init__(
        self, 
        encoder: nn.Module,
        projection_dim: int,
        prediction_length: int,
        lr: float = 1e-3,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.model = ForecastHead(encoder, projection_dim, prediction_length)
        self.criterion = nn.HuberLoss() # More robust than MSE for high initial losses
        self.lr = lr
        self.freeze_encoder = freeze_encoder
        self.save_hyperparameters(ignore=['encoder'])
        
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch # x: images, y: targets, _: scalers
        y_hat = self.model(x)
        
        # Check for NaNs and Clamp extreme values
        if torch.isnan(y_hat).any():
            print(f"DEBUG: NaN detected in y_hat at batch {batch_idx}")
            y_hat = torch.nan_to_num(y_hat, nan=0.0)
            
        y_hat = torch.clamp(y_hat, min=-11.0, max=11.0)
            
        loss = self.criterion(y_hat, y)
        
        if torch.isnan(loss):
            print(f"DEBUG: NaN detected in loss at batch {batch_idx}")
            return None # Skip this step
            
        self.log("train_loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, scalers = batch
        y_hat = self.model(x)
        
        # Clamp to prevent inf in original scale evaluation
        y_hat = torch.clamp(y_hat, min=-11.0, max=11.0)
        
        loss = self.criterion(y_hat, y)
        
        # Evaluation on original scale
        mse_orig, mae_orig = self._evaluate_original_scale(y, y_hat, scalers)
        
        self.log("val_loss", loss, prog_bar=True, batch_size=x.shape[0])
        self.log("val_mse_orig", mse_orig, prog_bar=True, batch_size=x.shape[0])
        self.log("val_mae_orig", mae_orig, prog_bar=True, batch_size=x.shape[0])
        return loss

    def _evaluate_original_scale(self, y, y_hat, scalers):
        """Calculates MSE and MAE on the inverse transformed (original) scale."""
        y_orig_list = []
        y_hat_orig_list = []
        
        for i in range(len(scalers)):
            # Inverse transform target and prediction
            y_orig = scalers[i].inverse_transform(y[i].cpu())
            y_hat_orig = scalers[i].inverse_transform(y_hat[i].detach().cpu())
            
            y_orig_list.append(y_orig)
            y_hat_orig_list.append(y_hat_orig)
            
        y_orig_all = torch.stack(y_orig_list)
        y_hat_orig_all = torch.stack(y_hat_orig_list)
        
        mse = torch.mean((y_orig_all - y_hat_orig_all) ** 2)
        mae = torch.mean(torch.abs(y_orig_all - y_hat_orig_all))
        
        return mse, mae

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
