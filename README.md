# KAN Time Series Contrastive Forecasting

This project implements a high-performance time series forecasting pipeline using **Kolmogorov-Arnold Networks (KAN)** and **Contrastive Learning**. It transforms 1D time series data into 2D Wavelet Scalograms (images) to extract robust features before training a forecasting head.

## Project Idea

The core concept is to treat time series as images using the **Continuous Wavelet Transform (CWT)**. By doing so, we can leverage computer vision techniques (CNNs) combined with the non-linear representational power of **KAN** to learn complex temporal patterns.

1.  **Data Representation**: 1D time series are converted into square 2D images (Scalograms) using `PyWavelets`.
2.  **Pre-training**: A `KANEncoder2D` (CNN + KAN) is trained using **Contrastive Learning (InfoNCE Loss)** to learn generalizable features from a large collection of Chronos datasets.
3.  **Forecasting**: A linear head is attached to the pre-trained encoder to predict future values based on the learned representations.

## Key Features

- **Hybrid Architecture**: Combines CNNs for spatial feature extraction with KANs for high-level non-linear mapping.
- **Contrastive Learning**: Uses InfoNCE loss to learn robust features without explicit labels.
- **Mixed Precision (AMP)**: Optimized for NVIDIA GPUs using 16-bit mixed precision.
- **Chronos Integration**: Simplified loading of diverse time series datasets from Hugging Face.
- **Wavelet Transformation**: Produces square 2D representations without manual tensor reshaping.

## Project Structure

- `src/data/`: Dataset and DataModule implementations.
- `src/models/`: KAN Encoder, Contrastive Loss, and Forecast Head.
- `src/utils/`: Wavelet transformation and normalization utilities.
- `config/`: Centralized YAML configuration.
- `outputs/`: Training logs and model checkpoints.
- `tests/`: Unit tests for data, wavelets, and models.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Configuration**:
    The project uses a `.env` file to set `PYTHONPATH`. This ensures that the `src` module is discoverable.
    ```text
    PYTHONPATH=.
    ```

## Getting Started

### 1. Pre-training (Contrastive Learning)

Run the main script to train the encoder on multiple datasets using contrastive learning:
```powershell
python main.py
```
This will save logs and checkpoints to `outputs/lightning_logs/`.

### 2. Forecasting

Once the encoder is trained, update the `encoder_checkpoint` path in `config/config.yaml` and run the forecasting script:
```powershell
python forecast_main.py
```
This script trains the forecast head and evaluates performance using **MSE** and **MAE** on the original (inverse-transformed) scale.

## Testing

The project includes a comprehensive test suite covering data processing, model architectures, and utility functions. We use `pytest` for running tests.

### Running Tests

You can run the entire test suite using:
```bash
python -m pytest tests/
```

For more detailed output (verbose mode):
```bash
python -m pytest -v -s tests/
```

### Test Coverage

The suite includes the following main parts:
- **Normalization**: Verifies `TimeSeriesScaler` logic for `numpy` and `torch`, including `NaN` handling.
- **Data Pipeline**: Tests `ChronosDataset` and `ChronosDataModule` across all modes (Forecast, Contrastive, and standard loading).
- **Models**: Verifies the `Efficient KAN` architecture, `KANEncoder` (1D/2D), and the `ForecastHead` logic (including parameter freezing).
- **Wavelets**: Ensures Continuous Wavelet Transformation correctly produces square 2D scalograms.

## Configuration

All parameters (batch size, image size, learning rate, datasets) are managed in `config/config.yaml`.

> **Note for Windows Users**: `num_workers` is set to `0` in the config to avoid pickling errors associated with the `pykan` library and multiprocessing.
