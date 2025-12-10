# Notebooks Directory

This directory contains Jupyter notebooks for running experiments in Google Colab and locally.

## Contents

### Complete Notebooks (Self-Contained)
- `FedAsync_Complete.ipynb`: Complete FedAsync implementation for Google Colab
- `FedBuff_Complete.ipynb`: Complete FedBuff implementation for Google Colab
- `TrustWeight_Complete.ipynb`: Complete TrustWeight implementation for Google Colab
- `TrustWeight_Local.ipynb`: TrustWeight implementation for local execution
- `TrustWeight_Standardized.ipynb`: Standardized TrustWeight notebook

### Features

All complete notebooks include:
- Google Colab setup (mounting Google Drive)
- Data downloading and loading
- Complete method implementation (client, server, strategy)
- Experiment configurations (6 experiments)
- Automated experiment runner
- Comprehensive visualization suite
- Results saving to Google Drive

## Usage

1. **Google Colab**: Upload any `*_Complete.ipynb` notebook
2. **Local**: Use `TrustWeight_Local.ipynb` or run Python scripts directly
3. **Experiments**: All notebooks support running 6 predefined experiments

## Experiment Configurations

All notebooks include 6 experiment configurations:
- **Exp1**: IID setting (α=1000.0, 0% stragglers)
- **Exp2-Exp6**: Non-IID with increasing straggler percentages (10%, 20%, 30%, 40%, 50%)

