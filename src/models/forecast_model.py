import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.kan_contrastive import KANEncoder2D

class ForecastHead(nn.Module):
    """
    Forecast head that uses a pre-trained KANEncoder2D to predict future values.
    """
    def __init__(
        self, 
        encoder: KANEncoder2D, 
        projection_dim: int, 
        prediction_length: int
    ):
        super().__init__()
        self.encoder = encoder
        # Simple linear head for forecasting
        self.head = nn.Linear(projection_dim, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch, 1, image_size, image_size)
        Returns:
            Predictions (batch, prediction_length)
        """
        features = self.encoder(x)
        predictions = self.head(features)
        return predictions

class ForecastLightning(pl.LightningModule):
    def __init__(
        self, 
        encoder: KANEncoder2D,
        projection_dim: int,
        prediction_length: int,
        lr: float = 1e-3
    ):
        super().__init__()
        self.model = ForecastHead(encoder, projection_dim, prediction_length)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch # x: images, y: targets, _: scalers
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, scalers = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        
        # Evaluation on original scale
        mse_orig, mae_orig = self._evaluate_original_scale(y, y_hat, scalers)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse_orig", mse_orig, prog_bar=True)
        self.log("val_mae_orig", mae_orig, prog_bar=True)
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
