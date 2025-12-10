#!/usr/bin/env python3
"""Display TrustWeight configuration for 20-round test."""

import yaml
from pathlib import Path

# Read config
config_path = Path("TrustWeight/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("="*70)
print("TRUSTWEIGHT CONFIGURATION (20 Rounds Test)")
print("="*70)

print("\n📊 DATA CONFIGURATION:")
print(f"  Dataset: {config['data']['dataset']}")
print(f"  Data directory: {config['data']['data_dir']}")
print(f"  Number of classes: {config['data']['num_classes']}")
print(f"  Partition alpha: {config['partition_alpha']} ({'IID' if config['partition_alpha'] >= 1000 else 'Non-IID'})")

print("\n👥 CLIENT CONFIGURATION:")
print(f"  Total clients: {config['clients']['total']}")
print(f"  Concurrent clients: {config['clients']['concurrent']}")
print(f"  Local epochs: {config['clients']['local_epochs']}")
print(f"  Batch size: {config['clients']['batch_size']}")
print(f"  Learning rate: {config['clients']['lr']}")
print(f"  Weight decay: {config['clients']['weight_decay']}")
print(f"  Gradient clipping: {config['clients']['grad_clip']}")
print(f"  Straggler percentage: {config['clients']['struggle_percent']}%")
print(f"  Delay slow range: {config['clients']['delay_slow_range']} seconds")
print(f"  Delay fast range: {config['clients']['delay_fast_range']} seconds")
print(f"  Jitter per round: {config['clients']['jitter_per_round']} seconds")

print("\n🖥️  SERVER CONFIGURATION:")
print(f"  Max rounds: {config['train']['max_rounds']}")
print(f"  Update clip norm: {config['train']['update_clip_norm']}")
print(f"  Target accuracy: {config['eval']['target_accuracy']}")
print(f"  Eval interval: {config['eval']['interval_seconds']} seconds")

print("\n⚙️  TRUSTWEIGHT SPECIFIC:")
print("  (These are set in the server initialization)")
print("  - Buffer size: 5 (default)")
print("  - Buffer timeout: 5.0s (default)")
print("  - Eta (server learning rate): 1.0 (default)")
print("  - Strategy uses:")
print("    * Freshness alpha: 0.1")
print("    * Momentum gamma: 0.9")
print("    * Beta1 (guard on staleness): 0.0")
print("    * Beta2 (guard on ||u||): 0.0")
print("    * Theta (quality weights): [0.0, 0.0, 0.0]")

print("\n🔧 OTHER SETTINGS:")
print(f"  Seed: {config['seed']}")
print(f"  Client delay: {config['server_runtime']['client_delay']} seconds")

print("\n📁 OUTPUT PATHS:")
print(f"  Logs directory: {config['io']['logs']}")
print(f"  Global log CSV: {config['io']['global_log_csv']}")
print(f"  Client participation CSV: {config['io']['client_participation_csv']}")
print(f"  Checkpoints: {config['io']['checkpoints_dir']}")
print(f"  Final model: {config['io']['final_model_path']}")

print("\n" + "="*70)
print("To run this test:")
print("  python3 -m TrustWeight.run")
print("="*70)


