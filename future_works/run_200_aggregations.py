#!/usr/bin/env python3
"""
Run extended test of improved TrustWeight for 200 aggregations.
Results are versioned with timestamp to avoid overwriting.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

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
    # Create versioned log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_logs_dir = future_works_dir / "logs" / f"run_200agg_{timestamp}"
    versioned_logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Extended Test: Improved TrustWeight (200 Aggregations)")
    print("="*80)
    print(f"\n📁 Results will be saved to: {versioned_logs_dir}")
    print("\nRunning for 200 aggregations to see long-term improvements...")
    print("Features enabled:")
    print("  ✅ Auto-tuning of θ, β, α parameters")
    print("  ✅ Update compression (50% ratio)")
    print()
    
    config_path = future_works_dir / "TrustWeight" / "config.yaml"
    cfg = load_config(str(config_path))
    
    # Override for extended test
    cfg.train.max_rounds = 200
    cfg.eval.target_accuracy = 0.80  # Stop if we reach 80% accuracy
    cfg.eval.interval_seconds = 10  # More frequent evaluation
    
    # Use versioned log files
    cfg.io.client_participation_csv = str(versioned_logs_dir / "Participation.csv")
    cfg.io.global_log_csv = str(versioned_logs_dir / "Eval.csv")
    
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
    print("Starting training (200 aggregations)...")
    print("="*80)
    print("This may take a while. Progress will be shown periodically.")
    print()
    
    start_time = time.time()
    last_print_time = start_time
    last_agg = 0
    
    with ThreadPoolExecutor(max_workers=cfg.clients.concurrent) as executor:
        futures = [executor.submit(client_loop, cl) for cl in clients]
        
        # Monitor progress
        while not server.should_stop():
            time.sleep(30)  # Check every 30 seconds
            current_agg = server._num_aggregations
            elapsed = time.time() - start_time
            
            if current_agg > last_agg:
                print(f"[Progress] Aggregation {current_agg}/200 | Time: {elapsed/60:.1f} min")
                last_agg = current_agg
                last_print_time = time.time()
            
            # Check if all futures are done (shouldn't happen before stop)
            if all(f.done() for f in futures):
                break
        
        server.wait()
        for f in futures:
            try:
                f.result(timeout=1.0)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    
    # Analyze results
    print("\n" + "="*80)
    print("200-AGGREGATION TEST RESULTS")
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
    
    print(f"\n⏱️  Total Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes, {elapsed/3600:.2f} hours)")
    print(f"🔄 Total Aggregations: {server._num_aggregations}")
    
    if accuracies:
        print(f"\n📈 Accuracy Progression (key milestones):")
        milestones = [5, 10, 25, 50, 75, 100, 150, 200]
        for target_agg in milestones:
            # Find closest evaluation
            closest = None
            min_diff = float('inf')
            for agg, acc in accuracies:
                diff = abs(agg - target_agg)
                if diff < min_diff:
                    min_diff = diff
                    closest = (agg, acc)
            
            if closest:
                agg, acc = closest
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
    
    # Create summary file
    summary_file = versioned_logs_dir / "SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("200-AGGREGATION TEST SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)\n")
        f.write(f"Total Aggregations: {server._num_aggregations}\n")
        if accuracies:
            f.write(f"\nFinal Accuracy: {accuracies[-1][1]:.4f} ({accuracies[-1][1]*100:.2f}%)\n")
            f.write(f"Initial Accuracy: {accuracies[0][1]:.4f} ({accuracies[0][1]*100:.2f}%)\n")
            f.write(f"Improvement: {accuracies[-1][1] - accuracies[0][1]:+.4f} ({((accuracies[-1][1] - accuracies[0][1])*100):+.2f}%)\n")
        f.write(f"\nResults saved to: {versioned_logs_dir}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Results saved to:")
    print(f"   • {eval_log}")
    print(f"   • {part_log}")
    print(f"   • {summary_file}")
    print(f"\n📁 All results versioned in: {versioned_logs_dir}")
    print("\n" + "="*80)
    print("✅ 200-aggregation test complete!")
    print("="*80)


if __name__ == "__main__":
    main()

