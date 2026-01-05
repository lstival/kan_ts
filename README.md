# KAN Time Series Project

This project implements a simplified Chronos dataset loader using PyTorch Lightning and follows ML best practices.

## Project Structure

- `data/`: Local data storage.
- `src/`: Source code.
    - `data/`: Data loading and processing.
        - `chronos_dataset.py`: Simplified Chronos dataset loader.
        - `datamodule.py`: PyTorch Lightning DataModule.
    - `models/`: Model definitions.
        - `kan_contrastive.py`: KAN-based contrastive encoder.
        - `lightning_kan.py`: PyTorch Lightning wrapper for KAN.
    - `utils/`: Utility functions.
        - `normalization.py`: Time series normalization.
        - `wavelet_transform.py`: 2D Wavelet transformation.
- `outputs/`: Training logs and model checkpoints.
- `tests/`: Unit tests.
    - `test_chronos_dataset.py`: Test for loading, normalization, and reverse normalization.
- `main.py`: Entry point for training.
- `requirements.txt`: Project dependencies.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Configuration**:
    The project uses a `.env` file to set `PYTHONPATH`. This ensures that the `src` module is discoverable.
    If you are using VS Code, it will automatically pick up the `.env` file.
    For manual terminal usage, you can set it:
    - Windows (PowerShell): `$env:PYTHONPATH="."`
    - Linux/macOS: `export PYTHONPATH=.`

## Configuration

The project uses a YAML configuration file located at `config/config.yaml`. You can adjust data parameters, model architecture, and training settings there.

## How to Run Tests

```bash
python tests/test_chronos_dataset.py
```

## How to Train

```bash
python main.py
```
