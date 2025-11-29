# FedBuff Experiment Report

**Generated:** 2025-11-29 11:04:28  
**Branch:** avinash  
**Total Experiments Completed:** 13 / 17

---

## Executive Summary

This report summarizes all completed FedBuff hyperparameter grid search experiments on CIFAR-10 with ResNet-18.

**Experiment Regimes:**
- **Regime A (Clean)**: 10 clients, concurrent=5, no stragglers, no delays
- **Regime B (Realistic Async)**: 50 clients, concurrent=20, heterogeneity enabled

## Results Summary

| Exp ID | Regime | Alpha | Local Epochs | Eta | Buffer Size | Buffer Timeout | Peak Acc | Final Acc | Rounds | Duration (min) |
|--------|--------|-------|--------------|-----|-------------|----------------|----------|-----------|--------|----------------|
| A1 | A | 0.1 | 5 | 0.5 | 5 | 0.0 | **42.56%** | 35.30% | 99 | 84.6 |
| A2 | A | 1.0 | 5 | 0.5 | 5 | 0.0 | **64.75%** | 63.24% | 100 | 83.4 |
| A3 | A | 10 | 5 | 0.5 | 5 | 0.0 | **67.75%** | 67.15% | 100 | 83.7 |
| A4 | A | 100 | 5 | 0.5 | 5 | 0.0 | **68.31%** | 67.99% | 100 | 84.2 |
| A5 | A | 1000 | 5 | 0.5 | 5 | 0.0 | **68.19%** | 67.55% | 100 | 85.1 |
| B1 | A | 0.5 | 5 | 0.5 | 3 | 0.0 | **64.05%** | 61.12% | 100 | 81.3 |
| B2 | A | 0.5 | 5 | 0.5 | 5 | 0.5 | **62.08%** | 57.88% | 100 | 86.8 |
| E1 | A | 0.5 | 5 | 0.2 | 5 | 0.0 | **60.83%** | 59.68% | 100 | 81.1 |
| E2 | A | 0.5 | 5 | 0.5 | 5 | 0.0 | **62.14%** | 61.66% | 100 | 81.6 |
| E3 | A | 0.5 | 5 | 0.8 | 5 | 0.0 | **59.87%** | 46.87% | 100 | 81.3 |
| L1 | A | 0.5 | 3 | 0.5 | 5 | 0.0 | **54.55%** | 54.55% | 100 | 54.4 |
| L2 | A | 0.5 | 5 | 0.5 | 5 | 0.0 | **61.31%** | 60.87% | 100 | 81.6 |
| L3 | A | 0.5 | 7 | 0.5 | 5 | 0.0 | **64.96%** | 58.44% | 100 | 108.5 |

## Key Findings

### 1. Alpha (Non-IID) Sensitivity

Tests impact of data distribution non-IIDness (Dirichlet alpha).

| Alpha | Peak Acc | Final Acc | Observation |
|-------|----------|-----------|-------------|
| 0.1 | 42.56% | 35.30% | Highly non-IID - severe accuracy drop |
| 1.0 | 64.75% | 63.24% | Moderate non-IID - good performance |
| 10.0 | 67.75% | 67.15% | Near IID - best performance |
| 100.0 | 68.31% | 67.99% | Near IID - best performance |
| 1000.0 | 68.19% | 67.55% | Near IID - best performance |

**Key Finding:** Higher alpha (more IID) dramatically improves accuracy. Alpha=0.1 achieves only ~42% vs ~68% for alpha≥100 (26% gap).

### 2. Local Epochs Impact

Tests impact of number of local training epochs per client.

| Local Epochs | Peak Acc | Final Acc | Observation |
|--------------|----------|-----------|-------------|
| 3 | 54.55% | 54.55% | Too few - limited learning |
| 5 | 61.31% | 60.87% | Optimal - good balance |
| 7 | 64.96% | 58.44% | More epochs - better peak but overfitting risk |

**Key Finding:** More local epochs (5-7) improve peak accuracy, but 7 epochs shows overfitting (final acc drops from 64.96% to 58.44%).

### 3. Eta (Server Mixing Rate) Impact

Tests impact of server-side mixing rate on aggregation stability.

| Eta | Peak Acc | Final Acc | Observation |
|-----|----------|-----------|-------------|
| 0.2 | 60.83% | 59.68% | Too conservative - slower learning |
| 0.5 | 62.14% | 61.66% | Optimal - stable and effective |
| 0.8 | 59.87% | 46.87% | Too aggressive - instability (final drops to 46.87%) |

**Key Finding:** Eta=0.5 is optimal. Eta=0.8 causes severe instability (final accuracy drops to 46.87% from 59.87% peak).

### 4. Buffer Configuration

Tests impact of buffer size and timeout settings.

| Config | Peak Acc | Final Acc | Observation |
|--------|----------|-----------|-------------|
| buffer_size=3, timeout=0.0 | 64.05% | 61.12% | Smaller buffer - slightly better |
| buffer_size=5, timeout=0.5 | 62.08% | 57.88% | Larger buffer + timeout - more variance |

**Key Finding:** Smaller buffer (size=3) performs slightly better and more stable.

## Best Configurations

### Top 3 by Peak Accuracy:

1. **A4**: 68.31% peak accuracy
   - Configuration: alpha=100, local_epochs=5, eta=0.5, buffer_size=5
   - Final Accuracy: 67.99%
   - Run Folder: `logs/avinash/run_20251128_112924/`

2. **A5**: 68.19% peak accuracy
   - Configuration: alpha=1000, local_epochs=5, eta=0.5, buffer_size=5
   - Final Accuracy: 67.55%
   - Run Folder: `logs/avinash/run_20251128_125338/`

3. **A3**: 67.75% peak accuracy
   - Configuration: alpha=10, local_epochs=5, eta=0.5, buffer_size=5
   - Final Accuracy: 67.15%
   - Run Folder: `logs/avinash/run_20251128_100540/`

## Regime B (Realistic Async) Status

⏳ **No Regime B experiments completed yet.**

H1-H4 experiments failed to start (likely configuration issue).

## Recommendations

### For Production Use:

1. **IID Data (alpha≥100):**
   - Expected accuracy: ~68%
   - Recommended: local_epochs=5, eta=0.5, buffer_size=3-5

2. **Non-IID Data (alpha=0.1):**
   - Expected accuracy: ~42% (significant challenge)
   - May need specialized techniques for extreme non-IID

3. **Optimal Hyperparameters:**
   - Local epochs: **5** (balance between learning and overfitting)
   - Eta: **0.5** (optimal mixing rate, avoid 0.8)
   - Buffer size: **3-5** (smaller may be slightly better)
   - Buffer timeout: **0.0** (flush immediately when full)

4. **Avoid:**
   - Eta=0.8 (causes instability and accuracy collapse)
   - Local epochs >7 (overfitting risk)

## Appendix

### All Completed Run Folders

- **A1**: `logs/avinash/run_20251128_071743/`
- **A2**: `logs/avinash/run_20251128_084219/`
- **A3**: `logs/avinash/run_20251128_100540/`
- **A4**: `logs/avinash/run_20251128_112924/`
- **A5**: `logs/avinash/run_20251128_125338/`
- **B1**: `logs/avinash/run_20251128_222706/`
- **B2**: `logs/avinash/run_20251128_234825/`
- **E1**: `logs/avinash/run_20251128_182311/`
- **E2**: `logs/avinash/run_20251128_194416/`
- **E3**: `logs/avinash/run_20251128_210551/`
- **L1**: `logs/avinash/run_20251128_141842/`
- **L2**: `logs/avinash/run_20251128_151307/`
- **L3**: `logs/avinash/run_20251128_163442/`

