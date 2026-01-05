import pytorch_lightning as pl
import torch
import sys
import yaml
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.datamodule import ChronosDataModule
from src.models.kan_contrastive import KANEncoder2D
from src.models.lightning_kan import KANContrastiveLightning
from src.models.forecast_model import ForecastLightning
from src.utils.wavelet_transform import WaveletTransform

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Load Configuration
    config_path = root_dir / "config" / "config.yaml"
    cfg = load_config(config_path)
    
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    # 1. Define Wavelet Transform
    transform = WaveletTransform(image_size=data_cfg['image_size'])

    # 2. DataModule in Forecast mode
    dm = ChronosDataModule(
        dataset_names=data_cfg['dataset_names'], 
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        normalize=data_cfg['normalize'],
        transform=transform,
        forecast=True, # Enable forecasting mode
        context_length=data_cfg['context_length'],
        prediction_length=data_cfg['prediction_length']
    )

    # 3. Initialize Encoder
    encoder = KANEncoder2D(
        image_size=data_cfg['image_size'],
        hidden_dim=model_cfg['hidden_dim'],
        projection_dim=model_cfg['projection_dim']
    )

    # 4. Load Pre-trained Encoder weights if available
    ckpt_path = model_cfg.get('encoder_checkpoint')
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading pre-trained encoder from {ckpt_path}")
        # Load the contrastive lightning module first
        contrastive_model = KANContrastiveLightning.load_from_checkpoint(
            ckpt_path,
            image_size=data_cfg['image_size'],
            hidden_dim=model_cfg['hidden_dim'],
            projection_dim=model_cfg['projection_dim']
        )
        # Extract the encoder
        encoder = contrastive_model.encoder
    else:
        print("No pre-trained encoder found. Training from scratch (or check config).")

    # 5. Initialize Forecast Model
    model = ForecastLightning(
        encoder=encoder,
        projection_dim=model_cfg['projection_dim'],
        prediction_length=data_cfg['prediction_length'],
        lr=model_cfg['lr']
    )

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator="auto",
        precision=train_cfg['precision'],
        gradient_clip_val=train_cfg['gradient_clip_val'],
        gradient_clip_algorithm=train_cfg['gradient_clip_algorithm'],
        default_root_dir=Path(train_cfg.get('default_root_dir', 'outputs')) / "forecast",
        log_every_n_steps=1
    )

    # Train
    print(f"Starting forecast training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
