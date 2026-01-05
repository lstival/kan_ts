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
from src.utils.wavelet_transform import WaveletTransform

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Set float32 matmul precision for better performance on NVIDIA GPUs
    torch.set_float32_matmul_precision('medium')

    # Load Configuration
    config_path = root_dir / "config" / "config.yaml"
    cfg = load_config(config_path)
    
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    # 1. Define Wavelet Transform for 2D Image representation
    transform = WaveletTransform(image_size=data_cfg['image_size'])

    # 2. DataModule with Transform and Contrastive mode
    dm = ChronosDataModule(
        dataset_names=data_cfg['dataset_names'], 
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        normalize=data_cfg['normalize'],
        transform=transform,
        contrastive=True
    )

    # 3. KAN Contrastive Model
    model = KANContrastiveLightning(
        image_size=data_cfg['image_size'],
        hidden_dim=model_cfg['hidden_dim'],
        projection_dim=model_cfg['projection_dim'],
        lr=model_cfg['lr'],
        temperature=model_cfg['temperature']
    )

    # 4. Trainer with Mixed Precision (AMP) and Gradient Clipping
    trainer = pl.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator="auto",
        precision=train_cfg['precision'],
        gradient_clip_val=train_cfg['gradient_clip_val'],
        gradient_clip_algorithm=train_cfg['gradient_clip_algorithm'],
        default_root_dir=train_cfg.get('default_root_dir', 'outputs'),
        log_every_n_steps=1 # Ensure logs are visible even for small datasets
    )

    # Train
    print(f"Starting training with config from {config_path}")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
