import torch
import pytorch_lightning as pl
from src.models.kan_contrastive import KANEncoder2D, InfoNCELoss

class KANContrastiveLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for KAN-based contrastive learning.
    """
    def __init__(
        self, 
        image_size: int = 64, 
        hidden_dim: int = 128, 
        projection_dim: int = 64,
        lr: float = 1e-3,
        temperature: float = 0.07
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = KANEncoder2D(image_size, hidden_dim, projection_dim)
        self.criterion = InfoNCELoss(temperature)
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # In contrastive learning, batch usually contains two views of the same data
        # For simplicity, we assume the dataloader provides (view1, view2)
        x1, x2 = batch
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        loss = self.criterion(z1, z2)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = self.criterion(z1, z2)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
