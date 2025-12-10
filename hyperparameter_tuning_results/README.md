# Hyperparameter Tuning Results

This directory contains results from hyperparameter tuning experiments.

## Structure

- **Root level**: SqueezeNet hyperparameter tuning results
- **resnet18/**: ResNet-18 hyperparameter tuning results

## SqueezeNet Results

- `quick_results_*.csv` - Quick tuning results (smaller grid, faster)
- `results_*.csv` - Full grid search results
- `quick_summary_*.json` - Summary of best configurations

## ResNet-18 Results

- `resnet18/quick_results_*.csv` - Quick tuning results
- `resnet18/results_*.csv` - Full grid search results

## CSV Format

All CSV files contain:
- `lr`: Learning rate
- `batch_size`: Batch size
- `weight_decay`: Weight decay
- `optimizer`: Optimizer type (adam/sgd)
- `momentum`: Momentum (for SGD)
- `best_test_acc`: Best test accuracy achieved
- `best_epoch`: Epoch where best accuracy was achieved
- `final_test_acc`: Final test accuracy
- `final_train_acc`: Final train accuracy


