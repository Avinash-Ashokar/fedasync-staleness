#!/usr/bin/env python3
"""
Test to compare accuracy improvements between original and improved TrustWeight.
Runs a longer test to get meaningful accuracy metrics.
"""
import os
import sys
import time
from pathlib import Path

# Setup paths
base_dir = Path(__file__).parent.parent
future_works_dir = Path(__file__).parent

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
import csv
from utils.helper import set_seed
from utils.partitioning import DataDistributor
from concurrent.futures import ThreadPoolExecutor


def run_test_with_accuracy(version_name: str, use_improved: bool, max_rounds: int = 10):
    """Run test and collect accuracy metrics."""
    print(f"\n{'='*80}")
    print(f"Testing {version_name}")
    print(f"{'='*80}")
    
    if use_improved:
        from TrustWeight.config import load_config
        from TrustWeight.server import AsyncServer
        from TrustWeight.client import AsyncClient
        config_path = future_works_dir / "TrustWeight" / "config.yaml"
    else:
        sys.path.insert(0, str(base_dir))
        from TrustWeight.config import load_config
        from TrustWeight.server import AsyncServer
        from TrustWeight.client import AsyncClient
        config_path = base_dir / "TrustWeight" / "config.yaml"
    
    cfg = load_config(str(config_path))
    
    # Override for test
    cfg.train.max_rounds = max_rounds
    cfg.eval.target_accuracy = 1.0  # Disable early stopping
    cfg.eval.interval_seconds = 5  # More frequent evaluation
    
    # Use unique log files
    if use_improved:
        cfg.io.client_participation_csv = str(Path(cfg.io.client_participation_csv).parent / "TrustWeightImproved.csv")
        cfg.io.global_log_csv = str(Path(cfg.io.global_log_csv).parent / "TrustWeightImprovedEval.csv")
    else:
        cfg.io.client_participation_csv = str(Path(cfg.io.client_participation_csv).parent / "TrustWeightOriginal.csv")
        cfg.io.global_log_csv = str(Path(cfg.io.global_log_csv).parent / "TrustWeightOriginalEval.csv")
    
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
    
    # Create clients
    clients = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        client = AsyncClient(cid=cid, indices=indices, cfg=cfg)
        
        # Enable features for improved version
        if use_improved:
            client.use_compression = True
            client.compression_ratio = 0.5
            server.strategy.cfg.enable_auto_tune = True
        
        clients.append(client)
    
    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            cont = cl.run_once(server)
            if not cont or server.should_stop():
                break
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=cfg.clients.concurrent) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]
        server.wait()
        for f in futures:
            try:
                f.result(timeout=1.0)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    
    # Read accuracy from eval log
    eval_log = Path(cfg.io.global_log_csv)
    accuracies = []
    if eval_log.exists():
        with open(eval_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    acc = float(row.get('test_acc', 0))
                    if acc > 0:
                        accuracies.append(acc)
                except:
                    pass
    
    # Read participation log
    part_log = Path(cfg.io.client_participation_csv)
    updates = []
    if part_log.exists():
        with open(part_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                updates.append(row)
    
    return {
        'version': version_name,
        'elapsed_time': elapsed,
        'total_aggregations': server._num_aggregations,
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1] if accuracies else 0.0,
        'best_accuracy': max(accuracies) if accuracies else 0.0,
        'num_updates': len(updates),
        'avg_staleness': sum(float(u.get('staleness', 0)) for u in updates) / len(updates) if updates else 0.0,
        'success': True
    }


def main():
    print("="*80)
    print("TrustWeight Accuracy Comparison Test")
    print("="*80)
    print("\nRunning longer test (10 aggregations) to measure accuracy...")
    
    results = {}
    
    try:
        # Test original
        results['original'] = run_test_with_accuracy("Original TrustWeight", use_improved=False, max_rounds=10)
        time.sleep(2)
        
        # Test improved
        results['improved'] = run_test_with_accuracy("Improved TrustWeight", use_improved=True, max_rounds=10)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print comparison
    print("\n" + "="*80)
    print("ACCURACY COMPARISON RESULTS")
    print("="*80)
    
    orig = results.get('original', {})
    impr = results.get('improved', {})
    
    print(f"\n📊 Original TrustWeight:")
    print(f"   ⏱️  Time: {orig.get('elapsed_time', 0):.2f}s")
    print(f"   🔄 Aggregations: {orig.get('total_aggregations', 0)}")
    print(f"   📈 Updates: {orig.get('num_updates', 0)}")
    print(f"   🎯 Final Accuracy: {orig.get('final_accuracy', 0):.4f}")
    print(f"   ⭐ Best Accuracy: {orig.get('best_accuracy', 0):.4f}")
    print(f"   ⏳ Avg Staleness: {orig.get('avg_staleness', 0):.3f}")
    
    print(f"\n✨ Improved TrustWeight (Auto-tuning + Compression):")
    print(f"   ⏱️  Time: {impr.get('elapsed_time', 0):.2f}s")
    print(f"   🔄 Aggregations: {impr.get('total_aggregations', 0)}")
    print(f"   📈 Updates: {impr.get('num_updates', 0)}")
    print(f"   🎯 Final Accuracy: {impr.get('final_accuracy', 0):.4f}")
    print(f"   ⭐ Best Accuracy: {impr.get('best_accuracy', 0):.4f}")
    print(f"   ⏳ Avg Staleness: {impr.get('avg_staleness', 0):.3f}")
    
    # Accuracy analysis
    print("\n" + "="*80)
    print("ACCURACY IMPROVEMENT ANALYSIS")
    print("="*80)
    
    if orig.get('final_accuracy', 0) > 0 and impr.get('final_accuracy', 0) > 0:
        final_diff = impr.get('final_accuracy', 0) - orig.get('final_accuracy', 0)
        best_diff = impr.get('best_accuracy', 0) - orig.get('best_accuracy', 0)
        
        print(f"\n🎯 Final Accuracy: {final_diff:+.4f} ({final_diff*100:+.2f}%)")
        if final_diff > 0:
            print("   ✅ Improved version achieves higher final accuracy!")
        elif abs(final_diff) < 0.01:
            print("   ✅ Similar final accuracy (compression maintains quality)")
        else:
            print("   ⚠️  Lower final accuracy (may need tuning)")
        
        print(f"\n⭐ Best Accuracy: {best_diff:+.4f} ({best_diff*100:+.2f}%)")
        if best_diff > 0:
            print("   ✅ Improved version reaches higher peak accuracy!")
        
        # Convergence speed
        orig_accs = orig.get('accuracies', [])
        impr_accs = impr.get('accuracies', [])
        if len(orig_accs) > 0 and len(impr_accs) > 0:
            # Find when each reached 0.5 accuracy
            orig_50 = next((i for i, acc in enumerate(orig_accs) if acc >= 0.5), None)
            impr_50 = next((i for i, acc in enumerate(impr_accs) if acc >= 0.5), None)
            
            if orig_50 is not None and impr_50 is not None:
                print(f"\n⚡ Convergence Speed (to 50% accuracy):")
                print(f"   Original: {orig_50} aggregations")
                print(f"   Improved: {impr_50} aggregations")
                if impr_50 < orig_50:
                    print(f"   ✅ Improved version converges {orig_50 - impr_50} aggregations faster!")
                elif impr_50 == orig_50:
                    print(f"   ✅ Similar convergence speed")
                else:
                    print(f"   ⚠️  Improved version converges {impr_50 - orig_50} aggregations slower")
    
    print("\n" + "="*80)
    print("✅ Accuracy Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

