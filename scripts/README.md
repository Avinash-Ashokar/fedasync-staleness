# Hyperparameter Tuning Scripts

Scripts for hyperparameter tuning of ResNet-18 and SqueezeNet architectures.

## Usage

```bash
# Quick ResNet-18 tuning (16 experiments, 30 epochs)
python3 scripts/resnet18_quick_tuning.py

# Full ResNet-18 grid search (72 experiments, 50 epochs)
python3 scripts/resnet18_hyperparameter_tuning.py

# Quick SqueezeNet tuning
python3 scripts/squeezenet_quick_tuning.py
```

## Results

Results are saved in `hyperparameter_tuning_results/`:
- SqueezeNet: `hyperparameter_tuning_results/*.csv`
- ResNet-18: `hyperparameter_tuning_results/resnet18/*.csv`
