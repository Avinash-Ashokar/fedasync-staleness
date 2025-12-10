# Comprehensive Experiment Results Table

This document summarizes all experiments conducted for the Federated Learning project.

## Summary Statistics

| Method | Total Runs | Avg Best Acc | Max Best Acc | Avg Final Acc | Max Final Acc |
|--------|------------|--------------|--------------|---------------|---------------|
| FedAsync | 17 | 0.1169 | 0.2854 | 0.1154 | 0.2839 |
| FedBuff | 64 | 0.2740 | 0.6831 | 0.2608 | 0.6799 |
| TrustWeight | 4 | 0.2084 | 0.3689 | 0.2017 | 0.3689 |

## Best Results by Method

### FedAsync
- **Best Accuracy:** 0.2854
- **Configuration:** α=1000.0, 0% stragglers, 20 clients, 300 rounds
- **Run Directory:** `logs/avinash/run_20251129_184737/`

### FedBuff
- **Best Accuracy:** 0.6831
- **Configuration:** α=100.0, 0% stragglers, 10 clients, 100 rounds
- **Run Directory:** `logs/avinash/run_20251128_112924/`

### TrustWeight
- **Best Accuracy:** 0.3689
- **Configuration:** α=1000.0 (IID), 0% stragglers, 20 clients, 104 rounds
- **Run Directory:** `logs/TrustWeight/Exp1/run_20251130_013402/`

## Detailed Results

For complete detailed results, see:
- **CSV File:** `all_experiments_summary.csv`
- **Full Report:** `experiment_summary.txt`

## Key Findings

1. **FedBuff** shows the best overall performance, achieving up to 68.31% accuracy in IID settings (α=100.0, 0% stragglers).

2. **FedAsync** shows limited performance, with best accuracy of 28.54% after 300 rounds in IID settings.

3. **TrustWeight** shows promising results with 36.89% accuracy in IID settings, but has limited experiments (only 4 completed runs).

4. **Non-IID Impact:** All methods show degraded performance with lower alpha values (more non-IID) and higher straggler percentages.

5. **Straggler Impact:** Experiments with 30% stragglers show significantly worse performance across all methods.

## Experiment Breakdown by Alpha

### FedAsync
- α=0.1: 4 runs, avg best acc: 0.1048
- α=0.5: 1 runs, avg best acc: 0.1015
- α=10.0: 4 runs, avg best acc: 0.1007
- α=1000.0: 8 runs, avg best acc: 0.1330

### FedBuff
- α=0.1: 24 runs, avg best acc: 0.1455
- α=0.5: 23 runs, avg best acc: 0.3690
- α=1.0: 2 runs, avg best acc: 0.6378
- α=10.0: 6 runs, avg best acc: 0.2881
- α=100.0: 2 runs, avg best acc: 0.6337
- α=1000.0: 5 runs, avg best acc: 0.2164

### TrustWeight
- α=0.1: 1 runs, avg best acc: 0.1969
- α=1000.0: 3 runs, avg best acc: 0.2122

---

*Generated from comprehensive analysis of all experiment logs*
*Last updated: 2025-11-30*


