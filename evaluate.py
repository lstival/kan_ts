import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.datamodule import ChronosDataModule
from src.models.lightning_kan import KANContrastiveLightning
from src.models.forecast_model import ForecastLightning

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load Configuration
    config_path = root_dir / "config" / "config.yaml"
    cfg = load_config(config_path)
    
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    
    # 1. DataModule in Forecast mode
    dm = ChronosDataModule(
        dataset_names=data_cfg['dataset_names'], 
        batch_size=min(32, data_cfg['batch_size']), # Use smaller batch for eval
        num_workers=0,
        normalize=data_cfg['normalize'],
        transform=None, 
        forecast=True, 
        context_length=data_cfg['context_length'],
        prediction_length=data_cfg['prediction_length']
    )
    dm.setup(stage="fit")
    val_loader = dm.val_dataloader()

    # 2. Find Checkpoint
    checkpoint_path = model_cfg.get('forecast_checkpoint')
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"Error: Forecast checkpoint not found at {checkpoint_path}")
        # Try to find any forecast checkpoint
        possible_paths = list(Path("outputs").rglob("*.ckpt"))
        forecast_paths = [p for p in possible_paths if "forecast" in str(p)]
        if forecast_paths:
            checkpoint_path = str(forecast_paths[-1])
            print(f"Using found checkpoint: {checkpoint_path}")
        else:
            print("No forecast checkpoints found in outputs/ directory.")
            return

    # 3. Initialize Encoder (1D)
    # We need to load the encoder first because ForecastLightning ignores it in hparams
    encoder_ckpt = model_cfg.get('encoder_checkpoint')
    if encoder_ckpt and Path(encoder_ckpt).exists():
        print(f"Loading encoder from {encoder_ckpt}")
        contrastive_model = KANContrastiveLightning.load_from_checkpoint(
            encoder_ckpt,
            input_dim=data_cfg['context_length'],
            hidden_dim=model_cfg['hidden_dim'],
            projection_dim=model_cfg['projection_dim']
        )
        encoder = contrastive_model.encoder
    else:
        print("Using fresh 1D KAN encoder for loading forecast model.")
        contrastive_model = KANContrastiveLightning(
            input_dim=data_cfg['context_length'],
            hidden_dim=model_cfg['hidden_dim'],
            projection_dim=model_cfg['projection_dim']
        )
        encoder = contrastive_model.encoder

    # 4. Load Forecast Model
    print(f"Loading forecast model from {checkpoint_path}")
    model = ForecastLightning.load_from_checkpoint(
        checkpoint_path,
        encoder=encoder,
        projection_dim=model_cfg['projection_dim'],
        prediction_length=data_cfg['prediction_length']
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. Inference and Evaluation
    all_y_true = []
    all_y_pred = []
    plot_samples = []

    print("Running evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            x, y, scalers = batch
            x, y = x.to(device), y.to(device)
            
            y_hat = model(x)
            
            # Denormalize
            for j in range(len(scalers)):
                y_true_orig = scalers[j].inverse_transform(y[j].cpu())
                y_pred_orig = scalers[j].inverse_transform(y_hat[j].cpu())
                x_orig = scalers[j].inverse_transform(x[j].cpu())
                
                all_y_true.append(y_true_orig)
                all_y_pred.append(y_pred_orig)
                
                # Save first few samples for plotting
                if len(plot_samples) < 5:
                    plot_samples.append({
                        'input': x_orig.numpy(),
                        'pred': y_pred_orig.numpy(),
                        'true': y_true_orig.numpy()
                    })

    # 6. Calculate Metrics
    y_true_all = torch.stack(all_y_true)
    y_pred_all = torch.stack(all_y_pred)
    
    mse = torch.mean((y_true_all - y_pred_all) ** 2).item()
    mae = torch.mean(torch.abs(y_true_all - y_pred_all)).item()
    rmse = np.sqrt(mse)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS (Denormalized)")
    print("="*30)
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*30)

    # 7. Plot Samples
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 20))
    for i, sample in enumerate(plot_samples):
        input_len = len(sample['input'])
        pred_len = len(sample['pred'])
        
        # Time axes
        time_input = np.arange(input_len)
        time_pred = np.arange(input_len, input_len + pred_len)
        
        axes[i].plot(time_input, sample['input'], label='Input (Context)', color='blue')
        axes[i].plot(time_pred, sample['true'], label='Ground Truth', color='green', linestyle='--')
        axes[i].plot(time_pred, sample['pred'], label='Prediction', color='red')
        
        axes[i].set_title(f"Sample {i+1} Forecast")
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xlabel("Timestamps")
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    plot_path = output_dir / "forecast_evaluation.png"
    plt.savefig(plot_path)
    print(f"\nEvaluation plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
