# Main Branch vs Avinash Branch: Experiment Results Comparison

**Analysis Date:** 2025-11-30  
**Main Branch:** `origin/main` (includes merged TrustWeight branch)  
**Avinash Branch:** `avinash` (local branch)

## Executive Summary

| Metric | Main Branch | Avinash Branch |
|--------|-------------|----------------|
| **Total Experiment Results** | 14 organized experiments | **216+ experiment runs** |
| **FedAsync Best Accuracy** | **75.32%** | 28.54% (logs) / 25.99% (outside) |
| **FedBuff Best Accuracy** | **62.24%** | **68.31%** (logs) / **50.15%** (outside) ⭐ |
| **TrustWeight Best Accuracy** | **72.32%** | **36.94%** (outside) / 36.89% (logs) |
| **Results Organization** | Parameter-based (`results/`) | Timestamp-based (`logs/`) + `outside/` (nested) |
| **Experiment Types** | Accuracy, Alpha Sweep, Straggler Sweep | Diverse custom experiments + Google Colab runs |

## Key Findings

### 1. **Performance Comparison**
- **Main branch results are higher** for FedAsync
- Main branch FedAsync: **75.32%** vs Avinash: **28.54%** (2.6x difference)
- **Avinash branch FedBuff outperforms main**: **68.31%** vs Main: **62.24%** ⭐
- Main branch TrustWeight: **72.32%** vs Avinash FL: **36.94%** (main higher)
- Note: Avinash has baseline (centralized) results showing 80.07%, but this is not FL

### 2. **Experimental Setup Differences**
- **Main branch**: Organized, systematic experiments with consistent configurations
- **Avinash branch**: More diverse experiments, iterative improvements, debugging runs

### 3. **Scale Differences**
- **Main branch**: 14 curated experiment results
- **Avinash branch**: **216+ experiment runs** (comprehensive exploration)
  - `logs/avinash/`: 85+ runs
  - `logs/TrustWeight/`: 4+ runs
  - `outside/` folder: **57 runs** (Exp1-Exp6 for all methods)
    - `outside/FedBuff/`: 6 runs (direct, best: 50.15%)
    - `outside/logs/FedBuff/`: 6 runs
    - `outside/logs/TrustWeight/`: 8 runs
    - `outside/outside/logs/FedAsync/`: 8 runs (best: 25.99%)
    - `outside/outside/logs/FedBuff/`: 7 runs (best: 40.03%)
    - `outside/outside/logs/TrustWeight/`: 20 runs (best: 36.94%)
  - **Total CSV files in outside/**: 138 files (59 main + 79 participation)

---

## Detailed Comparison by Method

### FedAsync Results

#### Main Branch
| Experiment | Best Accuracy | Final Accuracy | Rounds |
|------------|---------------|-----------------|--------|
| Accuracy Comparison | **75.32%** | ~75% | 1000 |

#### Avinash Branch
| Experiment | Best Accuracy | Final Accuracy | Rounds | Configuration |
|------------|---------------|-----------------|--------|---------------|
| Best Run | **28.54%** | 28.39% | 300 | α=1000.0, 0% stragglers, 20 clients |
| Average (17 runs) | 11.69% | 11.54% | - | Various configurations |

**Key Differences:**
- Main branch achieves **75.32%** accuracy in 1000 rounds
- Avinash branch best is **28.54%** after 300 rounds
- **Possible reasons**: Different hyperparameters (main uses `lr=0.005`, `local_epochs=2`), longer training (1000 vs 300 rounds), or different model initialization

---

### FedBuff Results

#### Main Branch
| Experiment | Best Accuracy | Final Accuracy | Rounds | Configuration |
|------------|---------------|-----------------|--------|---------------|
| Accuracy Comparison | **62.24%** | ~62% | 999 | Standard |
| Straggler 30% | **33.30%** | ~33% | 500 | 30% stragglers |
| Straggler 50% | **34.66%** | ~35% | 500 | 50% stragglers |

#### Avinash Branch
| Experiment | Best Accuracy | Final Accuracy | Rounds | Configuration | Location |
|------------|---------------|-----------------|--------|---------------|----------|
| Best Run (logs) | **68.31%** | 67.99% | 100 | α=100.0, 0% stragglers, 10 clients | `logs/avinash/` |
| Best Run (outside) | **50.15%** | 49.43% | - | Exp1 (IID) | `outside/FedBuff/Exp1/` |
| Average (77 runs) | 27.37% | 25.72% | - | Various configurations | Multiple |

**Key Differences:**
- **Avinash branch outperforms main branch** in clean IID settings: **68.31%** vs **62.24%** ⭐
- Main branch shows **33-35%** in straggler scenarios (realistic degradation)
- Avinash branch uses different hyperparameters: `lr=0.1`, `local_epochs=5` vs Main: `lr=0.001`, `local_epochs=1`
- Avinash branch shows better performance despite fewer rounds (100 vs 999)

---

### TrustWeight Results

#### Main Branch
| Experiment | Best Accuracy | Final Accuracy | Rounds | Configuration |
|------------|---------------|-----------------|--------|---------------|
| Accuracy Comparison | **72.32%** | ~72% | 1009 | Standard |
| Alpha 0.1 | **58.56%** | ~58% | 504 | α=1000.0 (IID) |
| Alpha 1.0 | 54.47% | ~54% | 504 | α=1.0 |
| Alpha 10.0 | 57.18% | ~57% | 503 | α=10.0 |
| Alpha 100.0 | 56.96% | ~57% | 504 | α=100.0 |
| Alpha 1000.0 | 58.56% | ~59% | 504 | α=1000.0 |
| Straggler 20% | 52.94% | ~53% | 504 | 20% stragglers |
| Straggler 30% | 53.45% | ~53% | 503 | 30% stragglers |
| Straggler 40% | 52.70% | ~53% | 504 | 40% stragglers |
| Straggler 50% | 52.06% | ~52% | 500 | 50% stragglers |

#### Avinash Branch
| Experiment | Best Accuracy | Final Accuracy | Rounds | Configuration | Location | Type |
|------------|---------------|-----------------|--------|---------------|----------|------|
| Best FL Run | **36.94%** | 36.94% | 504 | α=1000.0, 0% stragglers, 20 clients | `outside/outside/logs/` | FL |
| Second Best FL | **36.89%** | 36.89% | 104 | α=1000.0, 0% stragglers, 20 clients | `logs/TrustWeight/` | FL |
| Third Best FL | **24.43%** | 17.12% | 504 | α=0.1, 30% stragglers, 20 clients | `outside/outside/logs/` | FL |
| Baseline (Centralized) | **80.07%** | 80.02% | 150 | Baseline training | `logs/avinash/run_*_baseline/` | Baseline |
| Average (4 runs) | 20.84% | 20.17% | - | Various configurations |

**Key Differences:**
- Main branch outperforms Avinash FL: **72.32%** vs **36.94%** (main 2.0x better)
- Avinash branch has **51 TrustWeight FL runs** across multiple locations
- **Note**: The 80.07% result is from baseline (centralized) training, not FL
- Main branch shows consistent performance degradation with non-IID (alpha) and stragglers
- Avinash branch includes experiments from:
  - `logs/avinash/`: Baseline centralized runs (80.07% best - not FL)
  - `outside/outside/logs/`: Systematic Exp1-Exp6 FL runs (36.94% best)
  - `logs/TrustWeight/`: Local FL experiments (36.89% best)
- Main branch runs for 500-1000 rounds vs Avinash: varies (20-504 rounds)

---

## Experiment Organization Comparison

### Main Branch Organization
```
results/
├── Accuracy/              # Cross-method comparison
│   ├── FedAsync.csv      (75.32% best)
│   ├── FedBuff.csv       (62.24% best)
│   └── TrustWeight.csv   (72.32% best)
├── AlphaSweep/            # Non-IID parameter sweep
│   ├── alpha_0p1/        (53.73% best, α=0.1)
│   ├── alpha_1/          (54.47% best)
│   ├── alpha_10/         (57.18% best)
│   ├── alpha_100/        (56.96% best)
│   └── alpha_1000/       (58.56% best, IID)
└── StragglerSweep/       # Straggler percentage sweep
    ├── 20_pct/           (52.94% best)
    ├── 30_pct/           (53.45% best)
    ├── 40_pct/           (52.70% best)
    └── 50_pct/           (52.06% best)
```

**Characteristics:**
- ✅ Parameter-based organization
- ✅ Systematic experimental design
- ✅ Easy to compare across conditions
- ✅ Curated, high-quality results

### Avinash Branch Organization
```
logs/
├── avinash/
│   └── run_YYYYMMDD_HHMMSS/  # Timestamped runs
│       ├── FedAsync.csv
│       ├── FedBuff.csv
│       └── TrustWeight.csv
├── TrustWeight/
│   └── Exp1-Exp6/              # Experiment-specific folders
│       └── run_YYYYMMDD_HHMMSS/
└── FedAsync.csv, FedBuff.csv, TrustWeight.csv  # Root level results

outside/
├── logs/                       # Additional experiment runs
│   ├── FedAsync/Exp1-Exp6/
│   ├── FedBuff/Exp1-Exp6/
│   └── TrustWeight/Exp1-Exp6/
└── outside/logs/                # Nested experiment runs
    ├── FedAsync/Exp1-Exp6/
    ├── FedBuff/Exp1-Exp6/
    └── TrustWeight/Exp1-Exp6/

FedBuff/
└── [Google Colab results]      # Colab notebook outputs
```

**Characteristics:**
- ✅ Timestamp-based organization
- ✅ Tracks experiment history
- ✅ More comprehensive exploration
- ✅ Includes debugging and iterative runs

---

## Analysis of Performance Gap

### Key Configuration Differences

1. **Hyperparameters (Main Branch)**
   - FedAsync: `lr=0.005`, `local_epochs=2`, `batch_size=128`
   - FedBuff: `lr=0.001`, `local_epochs=1`, `batch_size=64`
   - TrustWeight: `lr=0.005`, `local_epochs=1`, `batch_size=128`

2. **Hyperparameters (Avinash Branch)**
   - FedAsync: `lr=0.005`, `local_epochs=1`, `batch_size=128`
   - FedBuff: `lr=0.1`, `local_epochs=5`, `batch_size=128` (better performance!)
   - TrustWeight: Various settings, limited experiments

3. **Training Duration**
   - Main branch: 500-1000 rounds
   - Avinash branch: 100-300 rounds (shorter)

4. **Model Architecture**
   - Both use ResNet-18 adapted for CIFAR-10
   - Possible differences in initialization or normalization

5. **Experimental Conditions**
   - Main branch: Systematic parameter sweeps
   - Avinash branch: More diverse, iterative exploration

---

## Recommendations

### For Understanding Main Branch Results

1. **Investigate Configuration Files**
   - Check `FedAsync/config.yaml`, `FedBuff/config.yml`, `TrustWeight/config.yaml` in main branch
   - Compare hyperparameters, model architecture, and training settings

2. **Examine Model Architecture**
   - Verify if main branch uses the same ResNet-18 or a different architecture
   - Check model initialization and pretraining status

3. **Review Evaluation Protocol**
   - Verify test set usage and evaluation timing
   - Check if there are any data leakage or evaluation bugs

4. **Compare Experimental Conditions**
   - Number of clients, concurrent clients, buffer settings
   - Data partitioning strategy and seed values

### For Improving Avinash Branch Results

1. **Adopt Main Branch Hyperparameters**
   - If main branch has better hyperparameters, adopt them
   - Focus on learning rate, server learning rate (eta), and local epochs

2. **Investigate Model Architecture**
   - Consider if main branch uses a different model variant
   - Experiment with different architectures or initialization

3. **Systematic Hyperparameter Tuning**
   - Conduct systematic sweeps similar to main branch
   - Focus on alpha values, straggler percentages, and learning rates

4. **Longer Training Runs**
   - Main branch TrustWeight runs for 500-1000 rounds
   - Avinash branch typically runs for 100-300 rounds

---

## Conclusion

### Performance Summary

1. **FedAsync**: Main branch significantly outperforms (75.32% vs 28.54%)
   - Main uses `local_epochs=2` vs Avinash `local_epochs=1`
   - Main runs for 1000 rounds vs Avinash 300 rounds

2. **FedBuff**: ⭐ **Avinash branch outperforms main** (68.31% vs 62.24%)
   - Avinash uses `lr=0.1`, `local_epochs=5` (more aggressive training)
   - Main uses `lr=0.001`, `local_epochs=1` (more conservative)
   - Avinash achieves better results in fewer rounds (100 vs 999)

3. **TrustWeight**: Main branch outperforms (72.32% vs 36.94% FL)
   - Main branch achieves **72.32%** in FL setting
   - Avinash best FL result is **36.94%** (from `outside/outside/logs/`)
   - **Note**: Avinash has baseline (centralized) results showing 80.07%, but this is not federated learning
   - Avinash has **51 TrustWeight FL runs** across multiple locations
   - Main branch runs for 500-1000 rounds vs Avinash: 20-504 rounds
   - Main branch has more systematic sweeps (alpha, straggler percentages)

### Key Insights

1. **Hyperparameter Impact**: Avinash's FedBuff success shows that `lr=0.1` and `local_epochs=5` work better than main's conservative settings
2. **Training Duration**: Main branch benefits from longer training (500-1000 rounds)
3. **Experimental Coverage**: Main branch has systematic sweeps; Avinash has more diverse exploration
4. **FedBuff Optimization**: Avinash branch has found better FedBuff hyperparameters

**Next Steps:**
1. ✅ **Adopt Avinash FedBuff hyperparameters** (`lr=0.1`, `local_epochs=5`) - already successful!
2. **Investigate FedBuff 50.15% result**: Found in `outside/FedBuff/Exp1/` - may be from Google Colab with different config
3. Apply main branch's FedAsync settings (`local_epochs=2`) to improve Avinash FedAsync
4. **Improve TrustWeight FL performance**: Main branch achieves 72.32% vs Avinash 36.94%
   - Consider main branch's hyperparameters (`lr=0.005`, `local_epochs=1`)
   - Run longer experiments (500-1000 rounds) similar to main branch
   - Conduct systematic alpha and straggler sweeps
5. ✅ **Consolidate results from `outside/` folder**: Complete analysis done (see `OUTSIDE_FOLDER_COMPLETE_ANALYSIS.md`)
6. **Document folder structure**: The `outside/` folder has three organizational patterns:
   - Direct method folders (`outside/FedBuff/`) - Google Colab results
   - Logs subfolder (`outside/logs/`) - Standard experiment runs
   - Nested logs (`outside/outside/logs/`) - Comprehensive experiments (best results)
7. **Clarify baseline vs FL results**: The 80.07% baseline result is centralized training, not FL

---

*This comparison is based on available CSV files in both branches. Some results may require further investigation to understand the performance gap.*

