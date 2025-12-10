# Logs Directory

This directory contains all experiment results, logs, and outputs from federated learning runs.

## Structure

```
logs/
├── avinash/              # Main experiment runs (timestamped)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── FedAsync.csv
│       ├── FedBuff.csv
│       ├── TrustWeight.csv
│       ├── CONFIG.yaml
│       └── COMMIT.txt
├── TrustWeight/         # TrustWeight-specific experiments
│   └── Exp1-Exp6/
│       └── run_YYYYMMDD_HHMMSS/
└── [Root CSV files]     # Legacy root-level logs
```

## Contents

### Experiment Results
- **FedAsync**: 25 runs, best: 28.54%
- **FedBuff**: 77 runs, best: 68.31% ⭐
- **TrustWeight**: 32 runs, best: 36.94%

### File Types
- `*.csv`: Experiment metrics (accuracy, loss, rounds, time)
- `*ClientParticipation.csv`: Per-client participation details
- `CONFIG.yaml`: Experiment configuration
- `COMMIT.txt`: Git commit tracking

### Organization

- **Timestamped runs** (`logs/avinash/run_*`): Chronological experiment history
- **Experiment-specific** (`logs/TrustWeight/Exp*`): Organized by experiment ID
- **Root level**: Legacy logs from early experiments

## Best Results

- **FedAsync**: 28.54% (`logs/avinash/run_20251129_184737/`)
- **FedBuff**: 68.31% (`logs/avinash/run_20251128_112924/`)
- **TrustWeight**: 36.94% (`experiments/outside/outside/logs/TrustWeight/Exp1/`)

For comprehensive analysis, see: `experiments/analysis/MAIN_VS_AVINASH_RESULTS_COMPARISON.md`

