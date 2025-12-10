#!/usr/bin/env python3
"""
Run extended test of improved TrustWeight to see long-term improvements.
"""
import os
import sys
import time
from pathlib import Path

# Setup paths
base_dir = Path(__file__).parent.parent
future_works_dir = Path(__file__).parent

sys.path.insert(0, str(future_works_dir))

# Silence logs
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTNING_DISABLE_RICH"] = "1"

import logging
logging.basicConfig(level=logging.ERROR)
for name in ["pytorch_lightning", "lightning", "torch", "torchvision"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from utils.helper import set_seed
from utils.partitioning import DataDistributor
from TrustWeight.config import load_config
from TrustWeight.server import AsyncServer
from TrustWeight.client import AsyncClient
from concurrent.futures import ThreadPoolExecutor


def main():
    print("="*80)
    print("Extended Test: Improved TrustWeight (Auto-tuning + Compression)")
    print("="*80)
    print("\nRunning for 30 aggregations to see long-term improvements...")
    print("Features enabled:")
    print("  ✅ Auto-tuning of θ, β, α parameters")
    print("  ✅ Update compression (50% ratio)")
    print()
    
    config_path = future_works_dir / "TrustWeight" / "config.yaml"
    cfg = load_config(str(config_path))
    
    # Override for extended test
    cfg.train.max_rounds = 30
    cfg.eval.target_accuracy = 0.80  # Stop if we reach 80% accuracy
    cfg.eval.interval_seconds = 10  # More frequent evaluation
    
    # Use unique log files in future_works/logs
    logs_dir = future_works_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    cfg.io.client_participation_csv = str(logs_dir / "TrustWeightImproved_Extended_Participation.csv")
    cfg.io.global_log_csv = str(logs_dir / "TrustWeightImproved_Extended_Eval.csv")
    
    set_seed(cfg.seed)
    
    # Partition data
    print("📊 Partitioning data...")
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
    print("🚀 Creating server with auto-tuning enabled...")
    server = AsyncServer(cfg=cfg)
    server.strategy.cfg.enable_auto_tune = True
    
    # Create clients with compression
    print(f"👥 Creating {cfg.clients.total} clients with compression enabled...")
    clients = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        client = AsyncClient(cid=cid, indices=indices, cfg=cfg)
        client.use_compression = True
        client.compression_ratio = 0.5
        clients.append(client)
    
    def client_loop(cl: AsyncClient) -> None:
        while not server.should_stop():
            cont = cl.run_once(server)
            if not cont or server.should_stop():
                break
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print()
    
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
    
    # Analyze results
    print("\n" + "="*80)
    print("EXTENDED TEST RESULTS")
    print("="*80)
    
    eval_log = Path(cfg.io.global_log_csv)
    accuracies = []
    if eval_log.exists():
        import csv
        with open(eval_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    acc = float(row.get('test_acc', 0))
                    agg = int(row.get('total_agg', 0))
                    if acc > 0:
                        accuracies.append((agg, acc))
                except:
                    pass
    
    print(f"\n⏱️  Total Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"🔄 Total Aggregations: {server._num_aggregations}")
    
    if accuracies:
        print(f"\n📈 Accuracy Progression:")
        for agg, acc in accuracies:
            print(f"   Aggregation {agg:3d}: {acc:.4f} ({acc*100:.2f}%)")
        
        final_acc = accuracies[-1][1]
        initial_acc = accuracies[0][1] if len(accuracies) > 1 else 0.0
        improvement = final_acc - initial_acc
        
        print(f"\n🎯 Accuracy Improvement:")
        print(f"   Initial: {initial_acc:.4f} ({initial_acc*100:.2f}%)")
        print(f"   Final:   {final_acc:.4f} ({final_acc*100:.2f}%)")
        print(f"   Gain:    {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # Auto-tuning parameters
        if hasattr(server.strategy, 'theta'):
            theta = server.strategy.theta
            if hasattr(theta, 'data'):
                theta_vals = [float(theta.data[i].item()) for i in range(3)]
            else:
                theta_vals = [float(theta[i].item() if hasattr(theta[i], 'item') else theta[i]) for i in range(3)]
            
            print(f"\n🎛️  Final Auto-tuning Parameters:")
            print(f"   θ (quality weights): [{theta_vals[0]:.4f}, {theta_vals[1]:.4f}, {theta_vals[2]:.4f}]")
            print(f"   β₁ (staleness guard): {server.strategy.beta1_adaptive:.4f}")
            print(f"   β₂ (norm guard): {server.strategy.beta2_adaptive:.4f}")
            print(f"   α (freshness): {server.strategy.alpha_adaptive:.4f}")
    
    # Read participation log
    part_log = Path(cfg.io.client_participation_csv)
    if part_log.exists():
        import csv
        with open(part_log, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\n📊 Client Participation:")
        print(f"   Total updates: {len(rows)}")
        if rows:
            staleness = [float(r.get('staleness', 0)) for r in rows if r.get('staleness', '')]
            if staleness:
                print(f"   Average staleness: {sum(staleness)/len(staleness):.3f}")
                print(f"   Max staleness: {max(staleness):.1f}")
                print(f"   Min staleness: {min(staleness):.1f}")
    
    print(f"\n✅ Results saved to:")
    print(f"   • {eval_log}")
    print(f"   • {part_log}")
    print("\n" + "="*80)
    print("✅ Extended test complete!")
    print("="*80)


if __name__ == "__main__":
    main()

