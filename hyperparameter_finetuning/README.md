# Hyperparameter Fine-tuning

This directory contains scripts and results for hyperparameter tuning of ResNet-18 and SqueezeNet architectures on CIFAR-10.

## Scripts

- `resnet18_hyperparameter_tuning.py` - Full grid search for ResNet-18 (72 experiments)
- `resnet18_quick_tuning.py` - Quick tuning for ResNet-18 (16 experiments)
- `squeezenet_hyperparameter_tuning.py` - Full grid search for SqueezeNet
- `squeezenet_quick_tuning.py` - Quick tuning for SqueezeNet
- `hyperparameter_tuning.py` - Legacy SqueezeNet tuning script

## Usage

```bash
# Quick ResNet-18 tuning (16 experiments, 30 epochs)
python3 hyperparameter_finetuning/resnet18_quick_tuning.py

# Full ResNet-18 grid search (72 experiments, 50 epochs)
python3 hyperparameter_finetuning/resnet18_hyperparameter_tuning.py

# Quick SqueezeNet tuning
python3 hyperparameter_finetuning/squeezenet_quick_tuning.py
```

## Results

Results are saved in `hyperparameter_finetuning/results/`:
- SqueezeNet hyperparameter tuning: `results/*.csv`
- ResNet-18 hyperparameter tuning: `results/resnet18/*.csv`
- SqueezeNet FedBuff experiments: `results/squeezenet_fedbuff_results/`

