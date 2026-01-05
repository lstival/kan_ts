import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleTimeSeriesModel(pl.LightningModule):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(-1) # (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out) # (batch, seq_len, 1)
        return out.squeeze(-1)

    def training_step(self, batch, batch_idx):
        # batch is (target, scaler) if normalize=True
        if isinstance(batch, list):
            x, _ = batch
        else:
            x = batch
            
        # Simple task: predict next value (shifted)
        y_hat = self(x[:, :-1])
        y = x[:, 1:]
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
