# Experiments Directory

This directory contains all experimental work, historical development, and additional research beyond the core implementation.

## Structure

```
experiments/
├── baseline/          # Baseline training experiments (SqueezeNet → ResNet-18 evolution)
├── notebooks/         # Jupyter notebooks for Google Colab and local execution
├── analysis/         # Analysis scripts and comparison reports
├── archive/          # Historical scripts, test files, and intermediate results
└── outside/          # Google Colab experiment results and additional runs
```

## Evolution History

### Phase 1: SqueezeNet Baseline
- Initial experiments used **SqueezeNet** architecture
- Baseline training script: `baseline/train_cifar10.py`
- Results documented in `archive/` folder

### Phase 2: ResNet-18 Upgrade
- Architecture upgraded to **ResNet-18** (adapted for CIFAR-10)
- Improved data augmentation pipeline
- Enhanced hyperparameter tuning

### Phase 3: Federated Learning Experiments
- FedAsync, FedBuff, and TrustWeight implementations
- Comprehensive experiment runs across multiple configurations
- Results stored in `logs/` directory

## Contents

- **baseline/**: Baseline centralized training experiments
- **notebooks/**: Complete notebooks for all three methods (FedAsync, FedBuff, TrustWeight)
- **analysis/**: Scripts and reports comparing methods, branches, and results
- **archive/**: Historical development files, test scripts, intermediate results
- **outside/**: Additional experiment runs from Google Colab and other sources

