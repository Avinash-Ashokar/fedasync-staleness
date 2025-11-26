# Hyperparameter Tuning Scripts

This directory contains scripts for hyperparameter tuning of different model architectures.

## SqueezeNet Scripts

- `hyperparameter_tuning.py` - Original SqueezeNet tuning script
- `squeezenet_hyperparameter_tuning.py` - Comprehensive SqueezeNet tuning (72 experiments)
- `squeezenet_quick_tuning.py` - Quick SqueezeNet tuning (16 experiments, 30 epochs)

## ResNet-18 Scripts

- `resnet18_hyperparameter_tuning.py` - Full ResNet-18 grid search (72 experiments, 50 epochs)
- `resnet18_quick_tuning.py` - Quick ResNet-18 test (16 experiments, 30 epochs)

## Usage

```bash
# Quick test for ResNet-18
python3 scripts/resnet18_quick_tuning.py

# Full grid search for ResNet-18
python3 scripts/resnet18_hyperparameter_tuning.py

# Quick test for SqueezeNet
python3 scripts/squeezenet_quick_tuning.py
```

## Results

Results are saved in `hyperparameter_tuning_results/`:
- SqueezeNet results: `hyperparameter_tuning_results/*.csv`
- ResNet-18 results: `hyperparameter_tuning_results/resnet18/*.csv`

