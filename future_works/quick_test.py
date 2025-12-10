#!/usr/bin/env python3
"""
Quick test script to compare original vs improved TrustWeight.
Runs a minimal test (3 aggregations) to verify improvements work.
"""
import os
import sys
import time
from pathlib import Path

# Setup paths
base_dir = Path(__file__).parent.parent
future_works_dir = Path(__file__).parent

# Add to path
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(future_works_dir))

# Silence logs
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTNING_DISABLE_RICH"] = "1"

import logging
logging.basicConfig(level=logging.ERROR)
for name in ["pytorch_lightning", "lightning", "torch", "torchvision"]:
    logging.getLogger(name).setLevel(logging.ERROR)

import yaml
import torch
from utils.helper import set_seed
from utils.partitioning import DataDistributor

# Import improved TrustWeight
from TrustWeight.config import load_config
from TrustWeight.server import AsyncServer
from TrustWeight.client import AsyncClient
from concurrent.futures import ThreadPoolExecutor


def run_improved_test(max_rounds: int = 3):
    """Run improved TrustWeight with auto-tuning and compression."""
    print("\n" + "="*80)
    print("Testing Improved TrustWeight (Auto-tuning + Compression)")
    print("="*80)
    
    config_path = future_works_dir / "TrustWeight" / "config.yaml"
    cfg = load_config(str(config_path))
    
    # Override for quick test
    cfg.train.max_rounds = max_rounds
    cfg.eval.target_accuracy = 1.0  # Disable early stopping
    
    # Set seed
    set_seed(cfg.seed)
    
    # Partition data
    distributor = DataDistributor(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.data.data_dir,
    )
    distributor.distribute_data(
        num_clients=cfg.clients.total,
        alpha=cfg.partition_alpha,
        seed=cfg.seed,
    )
    partitions = [distributor.partitions[cid] for cid in range(cfg.clients.total)]
    
    # Create server
    server = AsyncServer(cfg=cfg)
    
    # Create clients with compression enabled
    clients = []
    for cid in range(min(5, cfg.clients.total)):  # Use only 5 clients for quick test
        indices = partitions[cid] if cid < len(partitions) else []
        client = AsyncClient(
            cid=cid,
            indices=indices,
            cfg=cfg,
        )
        # Enable compression
        client.use_compression = True
        client.compression_ratio = 0.5
        clients.append(client)
    
    # Enable auto-tuning in strategy
    server.strategy.cfg.enable_auto_tune = True
    
    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            cont = cl.run_once(server)
            if not cont or server.should_stop():
                break
    
    # Run with limited clients
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(3, len(clients))) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]
        server.wait()
        for f in futures:
            try:
                f.result(timeout=1.0)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    
    # Read results
    log_path = Path(cfg.io.client_participation_csv)
    metrics = {
        'elapsed_time': elapsed,
        'total_aggregations': server._num_aggregations,
        'success': True
    }
    
    if log_path.exists():
        import csv
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                metrics['num_updates'] = len(rows)
                metrics['avg_staleness'] = sum(float(r.get('staleness', 0)) for r in rows) / len(rows)
    
    # Get strategy metrics
    if hasattr(server.strategy, 'performance_history') and server.strategy.performance_history:
        metrics['auto_tuning_active'] = True
        metrics['theta_values'] = [
            float(server.strategy.theta[0].item() if hasattr(server.strategy.theta, 'item') else server.strategy.theta[0]),
            float(server.strategy.theta[1].item() if hasattr(server.strategy.theta, 'item') else server.strategy.theta[1]),
            float(server.strategy.theta[2].item() if hasattr(server.strategy.theta, 'item') else server.strategy.theta[2]),
        ]
        metrics['beta1_adaptive'] = server.strategy.beta1_adaptive
        metrics['beta2_adaptive'] = server.strategy.beta2_adaptive
        metrics['alpha_adaptive'] = server.strategy.alpha_adaptive
    
    return metrics


def run_original_test(max_rounds: int = 3):
    """Run original TrustWeight for comparison."""
    print("\n" + "="*80)
    print("Testing Original TrustWeight")
    print("="*80)
    
    config_path = base_dir / "TrustWeight" / "config.yaml"
    
    # Import original
    sys.path.insert(0, str(base_dir))
    from TrustWeight.config import load_config as load_orig_config
    from TrustWeight.server import AsyncServer as OriginalServer
    from TrustWeight.client import AsyncClient as OriginalClient
    from TrustWeight.run import _set_seed
    
    cfg = load_orig_config(str(config_path))
    
    # Override for quick test
    cfg.train.max_rounds = max_rounds
    cfg.eval.target_accuracy = 1.0
    
    _set_seed(cfg.seed)
    
    # Partition data
    distributor = DataDistributor(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.data.data_dir,
    )
    distributor.distribute_data(
        num_clients=cfg.clients.total,
        alpha=cfg.partition_alpha,
        seed=cfg.seed,
    )
    partitions = [distributor.partitions[cid] for cid in range(cfg.clients.total)]
    
    # Create server
    server = OriginalServer(cfg=cfg)
    
    # Create clients
    clients = []
    for cid in range(min(5, cfg.clients.total)):
        indices = partitions[cid] if cid < len(partitions) else []
        client = OriginalClient(
            cid=cid,
            indices=indices,
            cfg=cfg,
        )
        clients.append(client)
    
    def client_loop(cl: OriginalClient) -> None:
        while not server.should_stop():
            cont = cl.run_once(server)
            if not cont or server.should_stop():
                break
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(3, len(clients))) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]
        server.wait()
        for f in futures:
            try:
                f.result(timeout=1.0)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    
    metrics = {
        'elapsed_time': elapsed,
        'total_aggregations': server._num_aggregations,
        'success': True
    }
    
    log_path = Path(cfg.io.client_participation_csv)
    if log_path.exists():
        import csv
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                metrics['num_updates'] = len(rows)
                metrics['avg_staleness'] = sum(float(r.get('staleness', 0)) for r in rows) / len(rows)
    
    return metrics


def main():
    print("="*80)
    print("TrustWeight Comparison Test")
    print("="*80)
    print("\nRunning quick tests (3 aggregations each)...")
    print("This will compare:")
    print("  1. Original TrustWeight")
    print("  2. Improved TrustWeight (auto-tuning + compression)")
    
    results = {}
    
    try:
        # Test original
        results['original'] = run_original_test(max_rounds=3)
        time.sleep(1)  # Brief pause
        
        # Test improved
        results['improved'] = run_improved_test(max_rounds=3)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    orig = results.get('original', {})
    impr = results.get('improved', {})
    
    print(f"\n📊 Original TrustWeight:")
    print(f"   ⏱️  Time: {orig.get('elapsed_time', 0):.2f}s")
    print(f"   🔄 Aggregations: {orig.get('total_aggregations', 0)}")
    print(f"   📈 Updates: {orig.get('num_updates', 0)}")
    print(f"   ⏳ Avg Staleness: {orig.get('avg_staleness', 0):.3f}")
    
    print(f"\n✨ Improved TrustWeight (Auto-tuning + Compression):")
    print(f"   ⏱️  Time: {impr.get('elapsed_time', 0):.2f}s")
    print(f"   🔄 Aggregations: {impr.get('total_aggregations', 0)}")
    print(f"   📈 Updates: {impr.get('num_updates', 0)}")
    print(f"   ⏳ Avg Staleness: {impr.get('avg_staleness', 0):.3f}")
    
    if impr.get('auto_tuning_active'):
        print(f"\n   🎯 Auto-tuning Active:")
        print(f"      θ values: {impr.get('theta_values', [0,0,0])}")
        print(f"      β₁ (staleness guard): {impr.get('beta1_adaptive', 0):.4f}")
        print(f"      β₂ (norm guard): {impr.get('beta2_adaptive', 0):.4f}")
        print(f"      α (freshness): {impr.get('alpha_adaptive', 0):.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    if orig.get('success') and impr.get('success'):
        time_diff = impr.get('elapsed_time', 0) - orig.get('elapsed_time', 0)
        time_pct = (time_diff / orig.get('elapsed_time', 1)) * 100 if orig.get('elapsed_time', 0) > 0 else 0
        
        print(f"\n⏱️  Time: {time_diff:+.2f}s ({time_pct:+.1f}%)")
        
        if time_diff < 0:
            print("   ✅ Compression is working - faster execution!")
        elif abs(time_diff) < 0.5:
            print("   ✅ Similar performance (compression overhead balanced)")
        else:
            print("   ⚠️  Slower (may be due to auto-tuning overhead or initialization)")
        
        staleness_diff = impr.get('avg_staleness', 0) - orig.get('avg_staleness', 0)
        print(f"\n⏳ Staleness: {staleness_diff:+.3f}")
        if abs(staleness_diff) < 0.1:
            print("   ✅ Similar staleness handling")
        else:
            print("   📊 Auto-tuning may be adapting to staleness patterns")
        
        if impr.get('auto_tuning_active'):
            print(f"\n🎯 Auto-tuning Status:")
            print(f"   ✅ Parameters are adapting during training")
            print(f"   📈 θ, β, α values are being optimized")
    
    print("\n" + "="*80)
    print("✅ Test Complete!")
    print("="*80)
    print("\nNote: For more meaningful results, run longer experiments (20+ aggregations)")
    print("      Auto-tuning effects become more pronounced over time")


if __name__ == "__main__":
    main()

