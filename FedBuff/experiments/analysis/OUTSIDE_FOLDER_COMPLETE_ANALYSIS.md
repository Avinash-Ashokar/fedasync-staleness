# Complete Outside Folder Analysis

**Analysis Date:** 2025-11-30  
**Total CSV Files Found:** 138 files  
**Total Experiment Runs:** 57 runs (FedAsync: 8, FedBuff: 21, TrustWeight: 28)

## Folder Structure Overview

The `outside/` folder has a nested structure with experiments organized in multiple locations:

```
outside/
├── FedBuff/
│   └── Exp1-Exp6/              # Direct FedBuff experiments (6 runs)
│       └── run_YYYYMMDD_HHMMSS/
│           └── FedBuff.csv
│
├── logs/
│   ├── FedBuff/
│   │   └── Exp1-Exp6/          # FedBuff in logs subfolder (6 runs)
│   ├── TrustWeight/
│   │   └── Exp1-Exp6/          # TrustWeight in logs subfolder (8 runs)
│   └── comparisons/            # Comparison CSV files (2 files)
│
└── outside/
    └── logs/
        ├── FedAsync/
        │   └── Exp1-Exp6/      # FedAsync experiments (8 runs)
        ├── FedBuff/
        │   └── Exp1-Exp6/      # FedBuff nested experiments (7 runs)
        └── TrustWeight/
            └── Exp1-Exp6/      # TrustWeight nested experiments (20 runs)
```

## Complete Results by Method

### FedAsync Results

**Total Runs:** 8  
**Best Accuracy:** **25.99%**  
**Location:** `outside/outside/logs/FedAsync/Exp1/run_20251130_055731/`

#### By Location:
- **outside/outside/logs/FedAsync/**: 8 runs, best: **25.99%**

#### By Experiment:
| Experiment | Runs | Best Accuracy | Location |
|------------|------|---------------|----------|
| Exp1 | 3 | **25.99%** | `outside/outside/logs/` |
| Exp2 | 1 | 14.38% | `outside/outside/logs/` |
| Exp3 | 1 | 17.13% | `outside/outside/logs/` |
| Exp4 | 1 | 16.20% | `outside/outside/logs/` |
| Exp5 | 1 | 17.91% | `outside/outside/logs/` |
| Exp6 | 1 | 16.89% | `outside/outside/logs/` |

#### Top 3 FedAsync Results:
1. **25.99%** - `outside/outside/logs/FedAsync/Exp1/run_20251130_055731/FedAsync.csv`
2. **19.70%** - `outside/outside/logs/FedAsync/Exp1/run_20251130_050745/FedAsync.csv`
3. **17.91%** - `outside/outside/logs/FedAsync/Exp5/run_20251130_113149/FedAsync.csv`

---

### FedBuff Results

**Total Runs:** 21  
**Best Accuracy:** **50.15%** ⭐  
**Location:** `outside/FedBuff/Exp1/run_20251130_052803/`

#### By Location:

1. **outside/FedBuff/Exp1-Exp6/** (Direct experiments)
   - 6 runs, best: **50.15%**
   - These appear to be Google Colab or direct notebook runs
   - Structure: `outside/FedBuff/Exp1/run_YYYYMMDD_HHMMSS/FedBuff.csv`

2. **outside/logs/FedBuff/Exp1-Exp6/** (Logs subfolder)
   - 6 runs, best: **37.99%**
   - Structure: `outside/logs/FedBuff/Exp1/run_YYYYMMDD_HHMMSS/FedBuff.csv`

3. **outside/outside/logs/FedBuff/Exp1-Exp6/** (Nested logs)
   - 7 runs, best: **40.03%**
   - Structure: `outside/outside/logs/FedBuff/Exp1/run_YYYYMMDD_HHMMSS/FedBuff.csv`

4. **outside/logs/comparisons/** (Comparison files)
   - 2 CSV files (comparison summaries)

#### By Experiment:
| Experiment | Runs | Best Accuracy | Best Location |
|------------|------|---------------|---------------|
| Exp1 | 4 | **50.15%** | `outside/FedBuff/Exp1/` |
| Exp2 | 3 | 33.30% | `outside/FedBuff/Exp2/` |
| Exp3 | 3 | 34.42% | `outside/FedBuff/Exp3/` |
| Exp4 | 3 | 34.66% | `outside/FedBuff/Exp4/` |
| Exp5 | 3 | 33.02% | `outside/FedBuff/Exp5/` |
| Exp6 | 3 | 34.00% | `outside/FedBuff/Exp6/` |

#### Top 5 FedBuff Results:
1. **50.15%** - `outside/FedBuff/Exp1/run_20251130_052803/FedBuff.csv` ⭐
2. **40.03%** - `outside/outside/logs/FedBuff/Exp1/run_20251130_053538/FedBuff.csv`
3. **37.99%** - `outside/logs/FedBuff/Exp1/run_20251130_022110/FedBuff.csv`
4. **34.66%** - `outside/FedBuff/Exp4/run_20251130_091249/FedBuff.csv`
5. **34.42%** - `outside/FedBuff/Exp3/run_20251130_075829/FedBuff.csv`

---

### TrustWeight Results

**Total Runs:** 28  
**Best Accuracy:** **36.94%**  
**Location:** `outside/outside/logs/TrustWeight/Exp1/run_20251130_110604/`

#### By Location:

1. **outside/logs/TrustWeight/Exp1-Exp6/** (Logs subfolder)
   - 8 runs, best: **13.63%**
   - Structure: `outside/logs/TrustWeight/Exp1/run_YYYYMMDD_HHMMSS/TrustWeight.csv`

2. **outside/outside/logs/TrustWeight/Exp1-Exp6/** (Nested logs)
   - 20 runs, best: **36.94%**
   - Structure: `outside/outside/logs/TrustWeight/Exp1/run_YYYYMMDD_HHMMSS/TrustWeight.csv`
   - **This is the main location with best results**

#### By Experiment:
| Experiment | Runs | Best Accuracy | Best Location |
|------------|------|---------------|---------------|
| Exp1 | 6 | **36.94%** | `outside/outside/logs/` |
| Exp2 | 6 | 24.34% | `outside/outside/logs/` |
| Exp3 | 4 | 23.86% | `outside/outside/logs/` |
| Exp4 | 4 | 24.43% | `outside/outside/logs/` |
| Exp5 | 4 | 22.26% | `outside/outside/logs/` |
| Exp6 | 4 | 21.64% | `outside/outside/logs/` |

#### Top 5 TrustWeight Results:
1. **36.94%** - `outside/outside/logs/TrustWeight/Exp1/run_20251130_110604/TrustWeight.csv` ⭐
2. **24.43%** - `outside/outside/logs/TrustWeight/Exp4/run_20251130_150622/TrustWeight.csv`
3. **24.34%** - `outside/outside/logs/TrustWeight/Exp2/run_20251130_122421/TrustWeight.csv`
4. **23.86%** - `outside/outside/logs/TrustWeight/Exp3/run_20251130_134529/TrustWeight.csv`
5. **22.26%** - `outside/outside/logs/TrustWeight/Exp5/run_20251130_162652/TrustWeight.csv`

---

## Key Findings

### 1. **FedBuff Best Result (50.15%)**
- **Location:** `outside/FedBuff/Exp1/run_20251130_052803/`
- This is significantly higher than other FedBuff results
- Appears to be from a direct experiment run (possibly Google Colab)
- **This is the highest FedBuff result found in the outside folder**

### 2. **TrustWeight Best Result (36.94%)**
- **Location:** `outside/outside/logs/TrustWeight/Exp1/run_20251130_110604/`
- This matches the best TrustWeight result found in the comprehensive scan
- From Exp1 (IID setting, α=1000.0, 0% stragglers)
- 504 rounds of training

### 3. **FedAsync Best Result (25.99%)**
- **Location:** `outside/outside/logs/FedAsync/Exp1/run_20251130_055731/`
- From Exp1 (IID setting, α=1000.0, 0% stragglers)
- 498 rounds of training

### 4. **Folder Organization Patterns**

The outside folder has three main organizational patterns:

1. **Direct Method Folders** (`outside/FedBuff/`)
   - Contains direct experiment runs
   - Best FedBuff result (50.15%) found here
   - Appears to be Google Colab or notebook outputs

2. **Logs Subfolder** (`outside/logs/`)
   - Contains FedBuff and TrustWeight experiments
   - Lower performance results (likely different configurations)

3. **Nested Logs** (`outside/outside/logs/`)
   - Contains all three methods (FedAsync, FedBuff, TrustWeight)
   - **Best TrustWeight (36.94%) and FedAsync (25.99%) results found here**
   - Most comprehensive experiment coverage

### 5. **Experiment Coverage**

All experiments (Exp1-Exp6) are represented across methods:
- **Exp1**: IID setting (α=1000.0, 0% stragglers) - Best results
- **Exp2**: Non-IID (α=0.1, 10% stragglers)
- **Exp3**: Non-IID (α=0.1, 20% stragglers)
- **Exp4**: Non-IID (α=0.1, 30% stragglers)
- **Exp5**: Non-IID (α=0.1, 40% stragglers)
- **Exp6**: Non-IID (α=0.1, 50% stragglers)

---

## Complete File Inventory

### CSV Files by Type:
- **FedAsync CSV files:** 8 (excluding Participation files)
- **FedBuff CSV files:** 21 (excluding Participation files)
- **TrustWeight CSV files:** 28 (excluding Participation files)
- **Comparison CSV files:** 2
- **Total:** 59 main CSV files + 79 Participation CSV files = **138 CSV files**

### Supporting Files:
- **CONFIG.yaml files:** Found in most run directories
- **COMMIT.txt files:** Found in most run directories
- **Checkpoint directories:** Found in some run directories
- **Comparison reports:** `outside/STRAGLER_IMPROVEMENTS_GUIDE.md`, `outside/COMPARISON_REPORT.md`

---

## Updated Best Results Summary

When including the `outside/` folder results:

| Method | Best Accuracy | Location | Type |
|--------|---------------|----------|------|
| **FedAsync** | **25.99%** | `outside/outside/logs/FedAsync/Exp1/run_20251130_055731/` | FL |
| **FedBuff** | **50.15%** ⭐ | `outside/FedBuff/Exp1/run_20251130_052803/` | FL (Colab?) |
| **TrustWeight** | **36.94%** | `outside/outside/logs/TrustWeight/Exp1/run_20251130_110604/` | FL |

**Note:** The FedBuff result of 50.15% from `outside/FedBuff/Exp1/` is significantly higher than other FedBuff results and may represent a different experimental setup or configuration.

---

## Recommendations

1. **Investigate FedBuff 50.15% result**: This is the highest FedBuff result found. Check the configuration and experimental setup.

2. **Consolidate results**: The nested structure (`outside/outside/logs/`) contains the most comprehensive and best-performing experiments.

3. **Document folder structure**: The three-tier organization (direct, logs, nested logs) should be documented for future reference.

4. **Standardize locations**: Consider consolidating all results into a single location structure for easier analysis.

---

*Complete analysis saved to: `outside_complete_results.json`*

