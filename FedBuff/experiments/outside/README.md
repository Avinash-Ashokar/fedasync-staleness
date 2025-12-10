# Outside Experiments Directory

This directory contains additional experiment runs, primarily from Google Colab executions and other experimental sources.

## Structure

```
outside/
├── FedBuff/              # Direct FedBuff experiments (Google Colab outputs)
├── logs/                 # Standard experiment logs
│   ├── FedBuff/Exp1-Exp6/
│   └── TrustWeight/Exp1-Exp6/
└── outside/logs/         # Nested experiment logs (most comprehensive)
    ├── FedAsync/Exp1-Exp6/
    ├── FedBuff/Exp1-Exp6/
    └── TrustWeight/Exp1-Exp6/
```

## Contents

### Experiment Runs
- **FedAsync**: 8 runs across Exp1-Exp6
- **FedBuff**: 21 runs across Exp1-Exp6 (best: 50.15%)
- **TrustWeight**: 28 runs across Exp1-Exp6 (best: 36.94%)

### Best Results
- **FedAsync**: 25.99% (`outside/logs/FedAsync/Exp1/`)
- **FedBuff**: 50.15% (`FedBuff/Exp1/`) ⭐
- **TrustWeight**: 36.94% (`outside/logs/TrustWeight/Exp1/`)

### Additional Files
- Comparison reports and analysis scripts
- Visualization plots (PNG files)
- Configuration files (YAML)
- Commit tracking files (COMMIT.txt)

## Organization

The folder has three organizational patterns:
1. **Direct method folders** (`FedBuff/`): Google Colab direct outputs
2. **Logs subfolder** (`logs/`): Standard experiment runs
3. **Nested logs** (`outside/logs/`): Most comprehensive experiments

## Total Files

- **138 CSV files** (59 main + 79 participation)
- Multiple visualization plots
- Configuration and tracking files

For detailed analysis, see: `experiments/analysis/OUTSIDE_FOLDER_COMPLETE_ANALYSIS.md`

