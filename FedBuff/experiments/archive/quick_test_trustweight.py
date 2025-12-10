#!/usr/bin/env python3
"""
Quick test script for TrustWeight with all 6 experiments.
Uses reduced rounds for faster testing.
"""

import sys
import os

# Add the current directory to path to import from solution.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from solution.py
# We'll execute the key parts
exec(open('solution.py').read())

# Override max_rounds for quick testing
print("\n" + "="*70)
print("QUICK TEST MODE - Reduced rounds for faster testing")
print("="*70)

# Reduce rounds for all experiments
for exp_id in experiments:
    original_rounds = experiments[exp_id]["train"]["max_rounds"]
    experiments[exp_id]["train"]["max_rounds"] = min(20, original_rounds)  # Cap at 20 for quick test
    print(f"  {exp_id}: {original_rounds} → {experiments[exp_id]['train']['max_rounds']} rounds")

print("\n" + "="*70)
print("Starting Quick Test Run...")
print("="*70)

# Run all 6 experiments
experiment_results = {}

total_start = time.time()

for exp_id in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5", "Exp6"]:
    exp_start = time.time()
    try:
        run_dir = run_single_experiment(exp_id, experiments)
        if run_dir:
            experiment_results[exp_id] = {
                "run_dir": run_dir,
                "status": "completed",
                "duration_min": (time.time() - exp_start) / 60.0
            }
            print(f"✅ {exp_id} completed in {experiment_results[exp_id]['duration_min']:.2f} minutes")
        else:
            experiment_results[exp_id] = {"status": "failed", "duration_min": (time.time() - exp_start) / 60.0}
    except Exception as e:
        print(f"❌ {exp_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        experiment_results[exp_id] = {"status": "error", "error": str(e), "duration_min": (time.time() - exp_start) / 60.0}
    
    print(f"\n{'─'*70}\n")

total_duration = (time.time() - total_start) / 60.0

print("="*70)
print("QUICK TEST COMPLETED")
print("="*70)
print(f"Total time: {total_duration:.2f} minutes ({total_duration/60:.2f} hours)")
print("\nResults summary:")
for exp_id, result in experiment_results.items():
    if result.get("status") == "completed":
        run_dir = result['run_dir']
        csv_path = run_dir / "TrustWeight.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            best_acc = df['test_acc'].max()
            final_acc = df['test_acc'].iloc[-1]
            print(f"  {exp_id}: ✅ {result['duration_min']:.2f} min | Best: {best_acc:.4f} | Final: {final_acc:.4f} | {run_dir}")
        else:
            print(f"  {exp_id}: ✅ {result['duration_min']:.2f} min | {run_dir}")
    else:
        print(f"  {exp_id}: ❌ {result.get('status', 'unknown')}")
print("="*70)


