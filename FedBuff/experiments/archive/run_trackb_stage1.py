#!/usr/bin/env python3
"""Run Track B Stage 1 experiments: FedAsync and FedBuff with alpha ∈ {0.1, 10.0, 1000.0} and straggler_fraction ∈ {0.0, 0.1, 0.3, 0.5}"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def run_experiment(method: str, alpha: float, strag_frac: float, seed: int = 1) -> str:
    """Run a single experiment and return the run folder path."""
    if method == "FedAsync":
        config_path = "FedAsync/config.yaml"
        run_cmd = ["python", "-m", "FedAsync.run"]
    elif method == "FedBuff":
        config_path = "FedBuff/config.yml"
        run_cmd = ["python", "-m", "FedBuff.run"]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Load config
    cfg = load_config(config_path)
    
    # Update config for this experiment
    cfg["partition_alpha"] = alpha
    cfg["clients"]["straggler_fraction"] = strag_frac
    cfg["seed"] = seed
    
    # Save updated config
    save_config(cfg, config_path)
    
    # Run experiment
    print(f"[{method}] alpha={alpha}, strag_frac={strag_frac}, seed={seed}")
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {method} failed with alpha={alpha}, strag_frac={strag_frac}")
        print(result.stderr)
        return None
    
    # Find the most recent run folder
    logs_dir = Path(cfg["io"]["logs_dir"]) / "avinash"
    run_folders = sorted(logs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_folders:
        return str(run_folders[0])
    return None

def main():
    methods = ["FedAsync", "FedBuff"]
    alphas = [0.1, 10.0, 1000.0]
    strag_fractions = [0.0, 0.1, 0.3, 0.5]
    seed = 1
    
    results = []
    total_runs = len(methods) * len(alphas) * len(strag_fractions)
    run_count = 0
    
    print(f"Starting Track B Stage 1: {total_runs} runs")
    print("=" * 60)
    
    for method in methods:
        for alpha in alphas:
            for strag_frac in strag_fractions:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] Running {method} with alpha={alpha}, strag_frac={strag_frac}")
                
                run_path = run_experiment(method, alpha, strag_frac, seed)
                if run_path:
                    results.append({
                        "method": method,
                        "alpha": alpha,
                        "strag_frac": strag_frac,
                        "seed": seed,
                        "run_path": run_path
                    })
                    print(f"✓ Completed: {run_path}")
                else:
                    print(f"✗ Failed: {method}, alpha={alpha}, strag_frac={strag_frac}")
    
    print("\n" + "=" * 60)
    print(f"Completed {len(results)}/{total_runs} runs")
    print("\nResults summary:")
    for r in results:
        print(f"  {r['method']:10s} alpha={r['alpha']:6.1f} strag={r['strag_frac']:.1f} -> {r['run_path']}")

if __name__ == "__main__":
    main()

