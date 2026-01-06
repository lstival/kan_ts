import pytorch_lightning as pl
import torch
import sys
import yaml
import warnings
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.datamodule import ChronosDataModule
from src.models.lightning_kan import KANContrastiveLightning
from src.utils.wavelet_transform import WaveletTransform
from pytorch_lightning.loggers import CometLogger

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Suppress specific pykan warnings about std() on small batches
    warnings.filterwarnings("ignore", message=r".*std\(\): degrees of freedom is <= 0.*")
    
    # Set float32 matmul precision for better performance on NVIDIA GPUs
    torch.set_float32_matmul_precision('medium')

    # Load Configuration
    config_path = root_dir / "config" / "config.yaml"
    cfg = load_config(config_path)
    
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    comet_cfg = cfg.get('comet', {})

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

    # 4. Logger: Comet.ml
    comet_logger = None
    if comet_cfg.get('api_key') and comet_cfg['api_key'] != "YOUR_COMET_API_KEY_HERE":
        comet_logger = CometLogger(
            api_key=comet_cfg['api_key'],
            workspace=comet_cfg.get('workspace'),
            project_name="kan-time-series",
            save_dir=train_cfg.get('default_root_dir', 'outputs')
        )
    else:
        print("Comet API key not found or placeholder used. Skipping Comet logging.")

    # 5. Trainer with Mixed Precision (AMP) and Gradient Clipping
    trainer = pl.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator="auto",
        precision=train_cfg['precision'],
        gradient_clip_val=train_cfg['gradient_clip_val'],
        gradient_clip_algorithm=train_cfg['gradient_clip_algorithm'],
        default_root_dir=train_cfg.get('default_root_dir', 'outputs'),
        logger=comet_logger,
        log_every_n_steps=1 # Ensure logs are visible even for small datasets
    )

    # Train
    print(f"Starting training with config from {config_path}")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
