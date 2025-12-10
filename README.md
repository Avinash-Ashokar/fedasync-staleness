# Federated Asynchronous Learning (FedAsync & FedBuff)

This repository implements **FedAsync** (Asynchronous Federated Learning) and **FedBuff** (Buffered Asynchronous Federated Learning) using **PyTorch Lightning** and **ResNet-18**. Both frameworks simulate heterogeneous client behavior and perform asynchronous updates to a central server.

## Project Structure

```
fedasync-staleness/
│
├── FedAsync/              # FedAsync implementation
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   ├── config.yaml
│   └── FedAsync.ipynb
│
├── FedBuff/               # FedBuff implementation
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   ├── config.yml
│   └── Fedbuff.ipynb
│
├── TrustWeight/            # TrustWeight implementation
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   └── ...
│
├── utils/                 # Shared utilities
│   ├── helper.py
│   ├── model.py
│   └── partitioning.py
│
├── scripts/               # Hyperparameter tuning scripts
│   ├── resnet18_hyperparameter_tuning.py
│   ├── resnet18_quick_tuning.py
│   └── ...
│
├── experiments/           # Historical experiment results
├── hyperparameter_tuning_results/  # Tuning results
├── Analysis/             # Analysis and plotting scripts
├── results/              # Final model outputs and visualizations
└── logs/                 # Training logs (CSV format)
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Environment

**Windows:**
```bash
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running Experiments

### FedAsync
```bash
python -m FedAsync.run
```

### FedBuff
```bash
python -m FedBuff.run
```

### TrustWeight
```bash
python -m TrustWeight.run
```

## Outputs

| Location | Description |
|----------|-------------|
| `logs/FedAsync.csv` | Global model metrics (aggregations, losses, accuracies) |
| `logs/FedAsyncClientParticipation.csv` | Per-client participation details |
| `results/FedAsyncModel.pt` | Final trained model weights |
| `checkpoints/` | Intermediate model checkpoints |

## Hyperparameter Tuning

Hyperparameter tuning scripts are available in the `scripts/` directory:

```bash
# Quick ResNet-18 tuning
python3 scripts/resnet18_quick_tuning.py

# Full ResNet-18 grid search
python3 scripts/resnet18_hyperparameter_tuning.py
```

Results are saved in `hyperparameter_tuning_results/`.

## Key Features

- **Asynchronous aggregation** — Clients update server immediately after local training
- **Client heterogeneity simulation** — Random per-client delays to mimic real-world latency
- **PyTorch Lightning** — Reproducibility, checkpointing, and clean training
- **ResNet-18 architecture** — Standardized model architecture across all methods
- **Automatic logging** — Global and client-level logs in CSV format
- **Config-driven** — All behavior customizable via YAML configuration files

## Analysis

Analysis scripts are available in the `Analysis/` directory:
- `AccComp.py` - Accuracy comparison across methods
- `AlphaSweep.py` - Alpha parameter sweep analysis
- `StragglerPlot.py` - Straggler effect visualization

## Results

Final results and visualizations are stored in `results/`:
- `Accuracy/` - Accuracy comparison results
- `AlphaSweep/` - Alpha parameter sweep results
- `StragglerSweep/` - Straggler percentage sweep results
