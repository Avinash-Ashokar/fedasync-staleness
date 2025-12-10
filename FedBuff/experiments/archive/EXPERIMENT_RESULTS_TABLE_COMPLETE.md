# Complete Comprehensive Experiment Results Table

**Last Updated:** 2025-11-30  
**Total Experiments:** 134

## Summary Statistics

| Method | Total Runs | Avg Best Acc | Max Best Acc | Avg Final Acc | Max Final Acc |
|--------|------------|--------------|--------------|---------------|---------------|
| **FedAsync** | 25 | 0.1360 | 0.2854 | 0.1337 | 0.2839 |
| **FedBuff** | 77 | 0.2737 | **0.6831** | 0.2572 | 0.6799 |
| **TrustWeight** | 32 | 0.1599 | 0.3694 | 0.1429 | 0.3694 |

## Best Results by Method

### FedAsync
- **Best Accuracy:** 0.2854 (28.54%)
- **Configuration:** α=1000.0, 0% stragglers, 20 clients, 300 rounds
- **Location:** `logs/avinash/run_20251129_184737/`

### FedBuff ⭐
- **Best Accuracy:** 0.6831 (68.31%)
- **Configuration:** α=100.0, 0% stragglers, 10 clients, 100 rounds
- **Location:** `logs/avinash/run_20251128_112924/`

### TrustWeight
- **Best Accuracy:** 0.3694 (36.94%)
- **Configuration:** α=1000.0 (IID), 0% stragglers, 20 clients, 504 rounds
- **Location:** `outside/outside/logs/TrustWeight/Exp1/run_20251130_110604/`

## Experiment Breakdown by Alpha

### FedAsync (25 runs)
- **α=0.1:** 9 runs, avg best acc: 0.1383
- **α=0.5:** 1 runs, avg best acc: 0.1015
- **α=10.0:** 4 runs, avg best acc: 0.1007
- **α=1000.0:** 11 runs, avg best acc: 0.1502

### FedBuff (77 runs)
- **α=0.1:** 34 runs, avg best acc: 0.1728
- **α=0.5:** 23 runs, avg best acc: 0.3690
- **α=1.0:** 2 runs, avg best acc: 0.6378
- **α=10.0:** 6 runs, avg best acc: 0.2881
- **α=100.0:** 2 runs, avg best acc: 0.6337
- **α=1000.0:** 8 runs, avg best acc: 0.2802

### TrustWeight (32 runs)
- **α=0.1:** 23 runs, avg best acc: 0.1497
- **α=1000.0:** 9 runs, avg best acc: 0.1861

## Experiments by Location

### Main Logs Directory (`logs/`)
- **FedAsync:** 17 runs
- **FedBuff:** 64 runs
- **TrustWeight:** 4 runs

### Outside Directory (`outside/logs/` and `outside/outside/logs/`)
- **FedAsync:** 8 runs (Exp1-Exp6)
- **FedBuff:** 13 runs (Exp1-Exp6)
- **TrustWeight:** 28 runs (Exp1-Exp6, multiple runs per experiment)

## Key Findings

1. **FedBuff Dominates:** Best overall performance with 68.31% accuracy in IID settings (α=100.0)

2. **TrustWeight Shows Promise:** Achieved 36.94% accuracy with extended training (504 rounds), competitive with FedAsync's best

3. **Non-IID Impact:** All methods show significant degradation with lower alpha values:
   - FedBuff: 0.1728 avg (α=0.1) vs 0.6337 avg (α=100.0)
   - TrustWeight: 0.1497 avg (α=0.1) vs 0.1861 avg (α=1000.0)

4. **Straggler Impact:** Experiments with stragglers (10-50%) show reduced performance across all methods

5. **Extended Training:** TrustWeight experiments with 504 rounds show improvement over shorter runs

6. **Experiment Coverage:**
   - **FedBuff:** Most comprehensive (77 runs) with good coverage across alpha values
   - **TrustWeight:** Good coverage (32 runs) with multiple runs per experiment configuration
   - **FedAsync:** Limited coverage (25 runs) with best results in IID settings

## Detailed Results

For complete detailed results, see:
- **CSV File:** `all_experiments_summary.csv` (134 experiments)
- **Full Report:** `experiment_summary_complete.txt`

## Notable Experiments

### FedBuff Best Performers
1. **68.31%** - α=100.0, 0% stragglers, 100 rounds (`logs/avinash/run_20251128_112924/`)
2. **67.75%** - α=10.0, 0% stragglers, 100 rounds (`logs/avinash/run_20251128_100540/`)
3. **64.96%** - α=0.5, 0% stragglers, 100 rounds (`logs/avinash/run_20251128_163442/`)

### TrustWeight Best Performers
1. **36.94%** - α=1000.0, 0% stragglers, 504 rounds (`outside/outside/logs/TrustWeight/Exp1/run_20251130_110604/`)
2. **36.89%** - α=1000.0, 0% stragglers, 104 rounds (`logs/TrustWeight/Exp1/run_20251130_013402/`)
3. **24.43%** - α=0.1, 30% stragglers, 504 rounds (`outside/outside/logs/TrustWeight/Exp4/run_20251130_150622/`)

### FedAsync Best Performers
1. **28.54%** - α=1000.0, 0% stragglers, 300 rounds (`logs/avinash/run_20251129_184737/`)
2. **23.40%** - α=0.1, 10% stragglers, 98 rounds (`outside/outside/logs/FedBuff/Exp2/run_20251130_023823/`)
3. **15.40%** - α=1000.0, 0% stragglers, 50 rounds (`logs/avinash/run_20251129_183128/`)

---

*This summary includes all experiments from:*
- `logs/avinash/` (main experiment directory)
- `logs/TrustWeight/` (TrustWeight experiments)
- `outside/logs/` (additional experiments)
- `outside/outside/logs/` (nested experiment directory)

