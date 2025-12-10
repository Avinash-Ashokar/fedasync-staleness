# Baseline Experiments

This directory contains baseline (centralized) training experiments used to establish performance benchmarks and tune hyperparameters.

## Contents

- `train_cifar10.py`: Baseline training script for CIFAR-10

## Evolution

1. **Initial Architecture**: SqueezeNet
   - Used for initial baseline experiments
   - Results helped establish baseline performance

2. **Upgraded Architecture**: ResNet-18
   - Adapted for CIFAR-10 (3×3 conv, stride 1, no maxpool)
   - Improved performance and convergence

## Purpose

Baseline experiments serve to:
- Establish performance benchmarks
- Tune hyperparameters (learning rate, weight decay, batch size)
- Compare federated learning performance against centralized training
- Validate data augmentation and training pipeline

## Results

Baseline results are stored in:
- `logs/avinash/run_*_baseline/metrics.csv`
- Best baseline accuracy: **80.07%** (centralized training, not FL)

