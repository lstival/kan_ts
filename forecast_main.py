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
from src.models.lightning_kan import KANContrastiveLightning
from src.models.forecast_model import ForecastLightning
from pytorch_lightning.loggers import CometLogger

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
    comet_cfg = cfg.get('comet', {})

    # 1. DataModule in Forecast mode (Just time series, no images)
    dm = ChronosDataModule(
        dataset_names=data_cfg['dataset_names'], 
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        normalize=data_cfg['normalize'],
        transform=None, # Raw time series
        forecast=True, 
        context_length=data_cfg['context_length'],
        prediction_length=data_cfg['prediction_length']
    )

    # 2. Initialize Model (Using pre-trained or fresh encoder)
    ckpt_path = model_cfg.get('encoder_checkpoint')
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading pre-trained encoder from {ckpt_path}")
        contrastive_model = KANContrastiveLightning.load_from_checkpoint(
            ckpt_path,
            input_dim=data_cfg['context_length'],
            hidden_dim=model_cfg['hidden_dim'],
            projection_dim=model_cfg['projection_dim']
        )
        encoder = contrastive_model.encoder
    else:
        print("No pre-trained encoder found. Initializing fresh 1D KAN encoder.")
        contrastive_model = KANContrastiveLightning(
            input_dim=data_cfg['context_length'],
            hidden_dim=model_cfg['hidden_dim'],
            projection_dim=model_cfg['projection_dim']
        )
        encoder = contrastive_model.encoder

    # 3. Initialize Forecast Model
    model = ForecastLightning(
        encoder=encoder,
        projection_dim=model_cfg['projection_dim'],
        prediction_length=data_cfg['prediction_length'],
        lr=model_cfg['lr'],
        freeze_encoder=True
    )

    # 4. Logger: Comet.ml
    comet_logger = None
    if comet_cfg.get('api_key') and comet_cfg['api_key'] != "YOUR_COMET_API_KEY_HERE":
        comet_logger = CometLogger(
            api_key=comet_cfg['api_key'],
            project=comet_cfg.get('project', "kan-time-series-forecast"),
            offline_directory=train_cfg.get('default_root_dir', 'outputs')
        )
    else:
        print("Comet API key not found or placeholder used. Skipping Comet logging.")

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator="auto",
        precision=train_cfg['precision'],
        gradient_clip_val=train_cfg['gradient_clip_val'],
        gradient_clip_algorithm=train_cfg['gradient_clip_algorithm'],
        default_root_dir=Path(train_cfg.get('default_root_dir', 'outputs')) / "forecast",
        logger=comet_logger,
        log_every_n_steps=1
    )

    # Train
    print(f"Starting forecast training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
