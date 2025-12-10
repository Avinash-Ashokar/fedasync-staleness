#!/usr/bin/env python3
"""
Quick test to compare original TrustWeight vs improved version with auto-tuning and compression.
"""
import os
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path to import original TrustWeight
sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence logs for cleaner output
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTNING_DISABLE_RICH"] = "1"
logging.basicConfig(level=logging.ERROR)

import torch
import yaml
from concurrent.futures import ThreadPoolExecutor

# Import both versions
from TrustWeight.config import load_config as load_original_config
from TrustWeight.run import main as run_original

# Import improved version
sys.path.insert(0, str(Path(__file__).parent))
from TrustWeight.config import load_config as load_improved_config
from TrustWeight.run import main as run_improved


def run_quick_test(version_name: str, run_func, config_path: str, max_rounds: int = 5):
    """Run a quick test and return metrics."""
    print(f"\n{'='*80}")
    print(f"Testing {version_name}")
    print(f"{'='*80}")
    
    # Modify config for quick test
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Save original config
    original_max_rounds = cfg_dict['train']['max_rounds']
    cfg_dict['train']['max_rounds'] = max_rounds
    cfg_dict['eval']['target_accuracy'] = 1.0  # Disable early stopping
    
    # Write temporary config
    temp_config = Path(config_path).parent / f"temp_config_{version_name.lower()}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(cfg_dict, f)
    
    # Set environment variable for config path
    os.environ['TRUSTWEIGHT_CONFIG'] = str(temp_config)
    
    start_time = time.time()
    try:
        # Run the test
        run_func()
        elapsed = time.time() - start_time
        
        # Read results
        log_path = Path(cfg_dict['io']['client_participation_csv'])
        if log_path.exists():
            import csv
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    total_agg = int(last_row.get('total_agg', 0))
                    avg_staleness = sum(float(r.get('staleness', 0)) for r in rows) / len(rows)
                    
                    return {
                        'version': version_name,
                        'total_aggregations': total_agg,
                        'avg_staleness': avg_staleness,
                        'elapsed_time': elapsed,
                        'num_updates': len(rows),
                        'success': True
                    }
        
        return {
            'version': version_name,
            'success': False,
            'error': 'No log file found'
        }
    except Exception as e:
        return {
            'version': version_name,
            'success': False,
            'error': str(e)
        }
    finally:
        # Restore original config
        if temp_config.exists():
            temp_config.unlink()


def main():
    print("="*80)
    print("TrustWeight Comparison Test")
    print("="*80)
    print("\nThis test compares:")
    print("  1. Original TrustWeight (main branch)")
    print("  2. Improved TrustWeight (auto-tuning + compression)")
    print("\nRunning quick tests (5 aggregations each)...")
    
    base_dir = Path(__file__).parent.parent
    original_config = base_dir / "TrustWeight" / "config.yaml"
    improved_config = Path(__file__).parent / "TrustWeight" / "config.yaml"
    
    results = []
    
    # Test original version
    if original_config.exists():
        result_original = run_quick_test(
            "Original TrustWeight",
            run_original,
            str(original_config),
            max_rounds=5
        )
        results.append(result_original)
        time.sleep(2)  # Brief pause between tests
    else:
        print(f"⚠️  Original config not found: {original_config}")
    
    # Test improved version
    if improved_config.exists():
        result_improved = run_quick_test(
            "Improved TrustWeight (Auto-tuning + Compression)",
            run_improved,
            str(improved_config),
            max_rounds=5
        )
        results.append(result_improved)
    else:
        print(f"⚠️  Improved config not found: {improved_config}")
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['version']}:")
        if result.get('success'):
            print(f"  ✅ Total Aggregations: {result.get('total_aggregations', 0)}")
            print(f"  ⏱️  Elapsed Time: {result.get('elapsed_time', 0):.2f}s")
            print(f"  📊 Average Staleness: {result.get('avg_staleness', 0):.3f}")
            print(f"  📈 Number of Updates: {result.get('num_updates', 0)}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    if len(results) == 2 and all(r.get('success') for r in results):
        orig = results[0]
        impr = results[1]
        
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        
        time_diff = impr.get('elapsed_time', 0) - orig.get('elapsed_time', 0)
        time_pct = (time_diff / orig.get('elapsed_time', 1)) * 100
        
        print(f"\n⏱️  Time Difference: {time_diff:+.2f}s ({time_pct:+.1f}%)")
        print(f"📊 Staleness Difference: {impr.get('avg_staleness', 0) - orig.get('avg_staleness', 0):+.3f}")
        print(f"📈 Updates Processed: {impr.get('num_updates', 0) - orig.get('num_updates', 0):+d}")
        
        if time_diff < 0:
            print("\n✅ Improved version is FASTER (compression working!)")
        else:
            print("\n⚠️  Improved version is slower (compression overhead or auto-tuning cost)")
        
        staleness_diff = impr.get('avg_staleness', 0) - orig.get('avg_staleness', 0)
        if abs(staleness_diff) < 0.1:
            print("✅ Staleness handling similar (auto-tuning may need more rounds to show effect)")
        else:
            print(f"📊 Staleness difference: {staleness_diff:+.3f} (auto-tuning may be adapting)")


if __name__ == "__main__":
    main()

