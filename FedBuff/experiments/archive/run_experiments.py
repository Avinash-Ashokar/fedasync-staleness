#!/usr/bin/env python3
"""
Run all experiments from remaining.csv sequentially.
Each experiment runs for 100 rounds.

Regime A (Clean): A1-A5, L1-L3, E1-E3, B1-B2
  - 10 clients, concurrent=5
  - No stragglers, no delays
  - Variable alpha from CSV

Regime B (Realistic Async): H1-H4
  - 50 clients, concurrent=20
  - Heterogeneity enabled (stragglers, delays)
  - Fixed alpha=0.1 (non-IID)
"""
import csv
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime
import time


def load_experiments(csv_path: str) -> list:
    """Load experiment configurations from CSV."""
    experiments = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append({
                'id': row['ID'],
                'alpha': float(row['alpha']),
                'local_epochs': int(row['local_epochs']),
                'eta': float(row['eta']),
                'buffer_size': int(row['buffer_size']),
                'buffer_timeout': float(row['buffer_timeout']),
                'stragglers': int(row['stragglers']),
            })
    return experiments


def update_config(config_path: str, exp: dict) -> None:
    """Update FedBuff config.yml with experiment parameters."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Determine regime based on experiment ID
    is_regime_b = exp['id'].startswith('H')
    
    if is_regime_b:
        # Regime B: Realistic async (20 clients, concurrent=10, heterogeneity enabled)
        cfg['clients']['total'] = 20  # Reduced from 50 for testing
        cfg['clients']['concurrent'] = 10  # Reduced from 20 for testing
        cfg['partition_alpha'] = 0.1  # Non-IID
        # Enable heterogeneity
        cfg['clients']['straggler_fraction'] = 0.3
        cfg['clients']['struggle_percent'] = 30
        cfg['clients']['delay_slow_range'] = [0.8, 2.0]
        cfg['clients']['delay_fast_range'] = [0.0, 0.2]
        cfg['clients']['jitter_per_round'] = 0.1
        cfg['clients']['fix_delays_per_client'] = True
    else:
        # Regime A: Clean (10 clients, no stragglers)
        cfg['clients']['total'] = 10
        cfg['clients']['concurrent'] = 5
        # Disable heterogeneity
        cfg['clients']['straggler_fraction'] = 0.0
        cfg['clients']['struggle_percent'] = 0
        cfg['clients']['delay_slow_range'] = [0.0, 0.0]
        cfg['clients']['delay_fast_range'] = [0.0, 0.0]
        cfg['clients']['jitter_per_round'] = 0.0
        cfg['clients']['fix_delays_per_client'] = True
    
    # Update hyperparameters (common to both regimes)
    if not is_regime_b:  # Only update alpha for Regime A
        cfg['partition_alpha'] = exp['alpha']
    cfg['clients']['local_epochs'] = exp['local_epochs']
    cfg['buff']['eta'] = exp['eta']
    cfg['buff']['buffer_size'] = exp['buffer_size']
    cfg['buff']['buffer_timeout_s'] = exp['buffer_timeout']
    cfg['train']['max_rounds'] = 100  # Fixed 100 rounds per experiment
    
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_experiment(exp_id: str, config_path: str) -> tuple:
    """Run a single experiment and return (success, run_folder)."""
    print(f"\n{'='*70}")
    print(f"Running experiment: {exp_id}")
    print(f"{'='*70}")
    
    # Clean checkpoints before each run
    checkpoint_dir = Path("checkpoints/FedBuff")
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Run FedBuff
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "FedBuff.run"],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per experiment (100 rounds takes ~60-90 min)
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Find the latest run folder
            log_dir = Path("logs/avinash")
            if log_dir.exists():
                run_folders = sorted(log_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if run_folders:
                    run_folder = run_folders[0]
                    print(f"✅ Experiment {exp_id} completed successfully")
                    print(f"   Duration: {duration/60:.1f} minutes")
                    print(f"   Run folder: {run_folder}")
                    return True, str(run_folder)
        
        print(f"❌ Experiment {exp_id} failed (return code: {result.returncode})")
        if result.stderr:
            print(f"   Error: {result.stderr[:500]}")
        return False, None
        
    except subprocess.TimeoutExpired:
        print(f"❌ Experiment {exp_id} timed out after 2 hours")
        return False, None
    except Exception as e:
        print(f"❌ Experiment {exp_id} failed with exception: {e}")
        return False, None


def main():
    csv_path = "remaining.csv"
    config_path = "FedBuff/config.yml"
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    if not Path(config_path).exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)
    
    experiments = load_experiments(csv_path)
    regime_a = [e for e in experiments if not e['id'].startswith('H')]
    regime_b = [e for e in experiments if e['id'].startswith('H')]
    print(f"Loaded {len(experiments)} experiments from {csv_path}")
    print(f"  Regime A (Clean): {len(regime_a)} experiments")
    print(f"  Regime B (Realistic Async): {len(regime_b)} experiments")
    
    # Load already completed experiments from existing results
    results_log = Path("experiment_results.csv")
    completed_exp_ids = set()
    existing_results = []
    
    if results_log.exists():
        with open(results_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results.append(row)
                if row.get('success', '').lower() == 'true':
                    completed_exp_ids.add(row['exp_id'])
    
    # Create/append to results log
    file_mode = 'a' if results_log.exists() and existing_results else 'w'
    with open(results_log, file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow([
                'exp_id', 'alpha', 'local_epochs', 'eta', 'buffer_size', 
                'buffer_timeout', 'stragglers', 'success', 'run_folder', 
                'duration_min', 'timestamp'
            ])
    
    if completed_exp_ids:
        print(f"\nSkipping {len(completed_exp_ids)} already completed experiments: {sorted(completed_exp_ids)}")
    
    total_start = time.time()
    
    for i, exp in enumerate(experiments, 1):
        # Skip if already completed successfully
        if exp['id'] in completed_exp_ids:
            print(f"\n{'#'*70}")
            print(f"Skipping experiment {i}/{len(experiments)}: {exp['id']} (already completed)")
            print(f"{'#'*70}")
            continue
        is_regime_b = exp['id'].startswith('H')
        regime = "B (Realistic Async)" if is_regime_b else "A (Clean)"
        
        print(f"\n{'#'*70}")
        print(f"Experiment {i}/{len(experiments)}: {exp['id']} - Regime {regime}")
        print(f"{'#'*70}")
        print(f"Configuration:")
        if is_regime_b:
            print(f"  Regime: B (50 clients, concurrent=20, heterogeneity enabled)")
            print(f"  alpha: 0.1 (fixed for Regime B)")
        else:
            print(f"  Regime: A (10 clients, concurrent=5, no stragglers)")
            print(f"  alpha: {exp['alpha']}")
        print(f"  local_epochs: {exp['local_epochs']}")
        print(f"  eta: {exp['eta']}")
        print(f"  buffer_size: {exp['buffer_size']}")
        print(f"  buffer_timeout: {exp['buffer_timeout']}")
        print(f"  stragglers: {exp['stragglers']} ({'enabled' if exp['stragglers'] == 1 else 'disabled'})")
        
        # Update config
        update_config(config_path, exp)
        
        # Run experiment
        exp_start = time.time()
        success, run_folder = run_experiment(exp['id'], config_path)
        exp_duration = (time.time() - exp_start) / 60
        
        # Log results
        with open(results_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                exp['id'], exp['alpha'], exp['local_epochs'], exp['eta'],
                exp['buffer_size'], exp['buffer_timeout'], exp['stragglers'],
                success, run_folder or '', exp_duration,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        if not success:
            print(f"\n⚠️  Experiment {exp['id']} failed. Continuing with next experiment...")
    
    total_duration = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Total duration: {total_duration:.1f} minutes")
    print(f"Results log: {results_log}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

