#!/usr/bin/env python3
"""
Comparison script for TrustWeight vs FedBuff results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths
logs_dir = Path("logs")
trustweight_dir = logs_dir / "TrustWeight"
fedbuff_dir = logs_dir / "FedBuff"

# Experiment configurations
experiments = {
    "Exp1": {"name": "IID (alpha=1000), no stragglers", "alpha": 1000.0, "stragglers": 0},
    "Exp2": {"name": "alpha=0.1, 10% stragglers", "alpha": 0.1, "stragglers": 10},
    "Exp3": {"name": "alpha=0.1, 20% stragglers", "alpha": 0.1, "stragglers": 20},
    "Exp4": {"name": "alpha=0.1, 30% stragglers", "alpha": 0.1, "stragglers": 30},
    "Exp5": {"name": "alpha=0.1, 40% stragglers", "alpha": 0.1, "stragglers": 40},
    "Exp6": {"name": "alpha=0.1, 50% stragglers", "alpha": 0.1, "stragglers": 50},
}

def get_latest_run(exp_dir):
    """Get the most recent run directory for an experiment."""
    if not exp_dir.exists():
        return None
    runs = sorted(exp_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None

def load_experiment_data(algorithm, exp_id):
    """Load CSV data for a specific algorithm and experiment."""
    if algorithm == "TrustWeight":
        base_dir = trustweight_dir
        csv_name = "TrustWeight.csv"
    else:
        base_dir = fedbuff_dir
        csv_name = "FedBuff.csv"
    
    exp_dir = base_dir / exp_id
    run_dir = get_latest_run(exp_dir)
    
    if run_dir is None:
        return None
    
    csv_path = run_dir / csv_name
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    df['time_min'] = df['time'] / 60.0  # Convert to minutes
    return df

# Load all data
print("Loading experiment data...")
all_data = {}

for exp_id in experiments.keys():
    tw_data = load_experiment_data("TrustWeight", exp_id)
    fb_data = load_experiment_data("FedBuff", exp_id)
    
    if tw_data is not None and fb_data is not None:
        all_data[exp_id] = {
            "TrustWeight": tw_data,
            "FedBuff": fb_data
        }
        print(f"✅ Loaded {exp_id}: TrustWeight ({len(tw_data)} rows), FedBuff ({len(fb_data)} rows)")
    else:
        print(f"⚠️  Missing data for {exp_id}")

if len(all_data) == 0:
    print("❌ No data found to compare!")
    exit(1)

# Create comparison directory
comparison_dir = logs_dir / "comparisons"
comparison_dir.mkdir(exist_ok=True)

# ========== 1. Summary Statistics Table ==========
print("\n" + "="*100)
print("COMPARISON SUMMARY: TrustWeight vs FedBuff")
print("="*100)

summary_data = []

for exp_id in experiments.keys():
    if exp_id not in all_data:
        continue
    
    tw_df = all_data[exp_id]["TrustWeight"]
    fb_df = all_data[exp_id]["FedBuff"]
    exp_config = experiments[exp_id]
    
    # TrustWeight metrics
    tw_best_acc = tw_df['test_acc'].max()
    tw_final_acc = tw_df['test_acc'].iloc[-1]
    tw_best_round = tw_df.loc[tw_df['test_acc'].idxmax(), 'total_agg']
    tw_final_round = tw_df['total_agg'].iloc[-1]
    tw_time = tw_df['time_min'].iloc[-1]
    
    # FedBuff metrics
    fb_best_acc = fb_df['test_acc'].max()
    fb_final_acc = fb_df['test_acc'].iloc[-1]
    fb_best_round = fb_df.loc[fb_df['test_acc'].idxmax(), 'total_agg']
    fb_final_round = fb_df['total_agg'].iloc[-1]
    fb_time = fb_df['time_min'].iloc[-1]
    
    # Calculate differences
    acc_diff = tw_best_acc - fb_best_acc
    time_diff = tw_time - fb_time
    
    summary_data.append({
        "Exp": exp_id,
        "Config": exp_config["name"],
        "TW_Best_Acc": tw_best_acc,
        "FB_Best_Acc": fb_best_acc,
        "Acc_Diff": acc_diff,
        "TW_Final_Acc": tw_final_acc,
        "FB_Final_Acc": fb_final_acc,
        "TW_Time": tw_time,
        "FB_Time": fb_time,
        "Time_Diff": time_diff,
        "TW_Rounds": tw_final_round,
        "FB_Rounds": fb_final_round,
    })

# Print summary table
summary_df = pd.DataFrame(summary_data)
print("\nDetailed Comparison:")
print(summary_df.to_string(index=False, float_format='%.4f'))

# Save summary to CSV
summary_df.to_csv(comparison_dir / "trustweight_vs_fedbuff_summary.csv", index=False)
print(f"\n✅ Summary saved to: {comparison_dir / 'trustweight_vs_fedbuff_summary.csv'}")

# ========== 2. Create Comparison Plots ==========
print("\n📊 Generating comparison plots...")

# Plot 1: Best Accuracy Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 Best Accuracy by Experiment
exp_ids = [exp for exp in experiments.keys() if exp in all_data]
tw_best_accs = [all_data[exp]["TrustWeight"]['test_acc'].max() for exp in exp_ids]
fb_best_accs = [all_data[exp]["FedBuff"]['test_acc'].max() for exp in exp_ids]

x = np.arange(len(exp_ids))
width = 0.35

axes[0, 0].bar(x - width/2, tw_best_accs, width, label='TrustWeight', alpha=0.8, color='#2ca02c')
axes[0, 0].bar(x + width/2, fb_best_accs, width, label='FedBuff', alpha=0.8, color='#1f77b4')
axes[0, 0].set_xlabel('Experiment', fontsize=12)
axes[0, 0].set_ylabel('Best Test Accuracy', fontsize=12)
axes[0, 0].set_title('Best Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(exp_ids, fontsize=10)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0, max(max(tw_best_accs), max(fb_best_accs)) * 1.1])

# Add value labels
for i, (tw, fb) in enumerate(zip(tw_best_accs, fb_best_accs)):
    axes[0, 0].text(i - width/2, tw, f'{tw:.3f}', ha='center', va='bottom', fontsize=8)
    axes[0, 0].text(i + width/2, fb, f'{fb:.3f}', ha='center', va='bottom', fontsize=8)

# 1.2 Accuracy Difference (TrustWeight - FedBuff)
acc_diffs = [tw - fb for tw, fb in zip(tw_best_accs, fb_best_accs)]
colors = ['green' if d > 0 else 'red' for d in acc_diffs]
axes[0, 1].bar(exp_ids, acc_diffs, alpha=0.7, color=colors, edgecolor='black')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Experiment', fontsize=12)
axes[0, 1].set_ylabel('Accuracy Difference (TW - FB)', fontsize=12)
axes[0, 1].set_title('Accuracy Advantage: TrustWeight vs FedBuff', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, diff in enumerate(acc_diffs):
    axes[0, 1].text(i, diff, f'{diff:+.3f}', ha='center', va='bottom' if diff > 0 else 'top', fontsize=9)

# 1.3 Time Comparison
tw_times = [all_data[exp]["TrustWeight"]['time_min'].iloc[-1] for exp in exp_ids]
fb_times = [all_data[exp]["FedBuff"]['time_min'].iloc[-1] for exp in exp_ids]

axes[1, 0].bar(x - width/2, tw_times, width, label='TrustWeight', alpha=0.8, color='#ff7f0e')
axes[1, 0].bar(x + width/2, fb_times, width, label='FedBuff', alpha=0.8, color='#9467bd')
axes[1, 0].set_xlabel('Experiment', fontsize=12)
axes[1, 0].set_ylabel('Total Time (minutes)', fontsize=12)
axes[1, 0].set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(exp_ids, fontsize=10)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (tw, fb) in enumerate(zip(tw_times, fb_times)):
    axes[1, 0].text(i - width/2, tw, f'{tw:.1f}m', ha='center', va='bottom', fontsize=8)
    axes[1, 0].text(i + width/2, fb, f'{fb:.1f}m', ha='center', va='bottom', fontsize=8)

# 1.4 Accuracy Trajectory Comparison (All experiments)
for exp_id in exp_ids:
    tw_df = all_data[exp_id]["TrustWeight"]
    fb_df = all_data[exp_id]["FedBuff"]
    axes[1, 1].plot(tw_df['total_agg'], tw_df['test_acc'], '--', linewidth=1.5, alpha=0.7, label=f'TW-{exp_id}')
    axes[1, 1].plot(fb_df['total_agg'], fb_df['test_acc'], '-', linewidth=1.5, alpha=0.7, label=f'FB-{exp_id}')

axes[1, 1].set_xlabel('Round', fontsize=12)
axes[1, 1].set_ylabel('Test Accuracy', fontsize=12)
axes[1, 1].set_title('Accuracy Trajectory: All Experiments', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=8, ncol=2, loc='lower right')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig(comparison_dir / "trustweight_vs_fedbuff_comparison.png", dpi=150, bbox_inches='tight')
print(f"✅ Comparison plot saved: {comparison_dir / 'trustweight_vs_fedbuff_comparison.png'}")
plt.close()

# ========== 3. Individual Experiment Comparisons ==========
print("\n📊 Generating individual experiment comparison plots...")

for exp_id in exp_ids:
    tw_df = all_data[exp_id]["TrustWeight"]
    fb_df = all_data[exp_id]["FedBuff"]
    exp_config = experiments[exp_id]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3.1 Accuracy vs Rounds
    axes[0, 0].plot(tw_df['total_agg'], tw_df['test_acc'], 'o-', linewidth=2, markersize=4, 
                   label='TrustWeight', color='#2ca02c')
    axes[0, 0].plot(fb_df['total_agg'], fb_df['test_acc'], 's-', linewidth=2, markersize=4, 
                   label='FedBuff', color='#1f77b4')
    axes[0, 0].set_xlabel('Round', fontsize=11)
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0, 0].set_title(f'{exp_id}: Accuracy vs Rounds', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.0])
    
    # 3.2 Accuracy vs Time
    axes[0, 1].plot(tw_df['time_min'], tw_df['test_acc'], 'o-', linewidth=2, markersize=4, 
                   label='TrustWeight', color='#2ca02c')
    axes[0, 1].plot(fb_df['time_min'], fb_df['test_acc'], 's-', linewidth=2, markersize=4, 
                   label='FedBuff', color='#1f77b4')
    axes[0, 1].set_xlabel('Wall Clock Time (minutes)', fontsize=11)
    axes[0, 1].set_ylabel('Test Accuracy', fontsize=11)
    axes[0, 1].set_title(f'{exp_id}: Accuracy vs Wall Clock Time', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.0])
    
    # 3.3 Loss vs Rounds
    axes[1, 0].plot(tw_df['total_agg'], tw_df['test_loss'], 'o-', linewidth=2, markersize=4, 
                   label='TrustWeight', color='#d62728')
    axes[1, 0].plot(fb_df['total_agg'], fb_df['test_loss'], 's-', linewidth=2, markersize=4, 
                   label='FedBuff', color='#ff7f0e')
    axes[1, 0].set_xlabel('Round', fontsize=11)
    axes[1, 0].set_ylabel('Test Loss', fontsize=11)
    axes[1, 0].set_title(f'{exp_id}: Loss vs Rounds', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3.4 Metrics Summary
    axes[1, 1].axis('off')
    summary_text = f"""
{exp_id}: {exp_config['name']}

TrustWeight:
  Best Accuracy: {tw_df['test_acc'].max():.4f} (Round {tw_df.loc[tw_df['test_acc'].idxmax(), 'total_agg']})
  Final Accuracy: {tw_df['test_acc'].iloc[-1]:.4f}
  Total Rounds: {tw_df['total_agg'].iloc[-1]}
  Total Time: {tw_df['time_min'].iloc[-1]:.2f} minutes

FedBuff:
  Best Accuracy: {fb_df['test_acc'].max():.4f} (Round {fb_df.loc[fb_df['test_acc'].idxmax(), 'total_agg']})
  Final Accuracy: {fb_df['test_acc'].iloc[-1]:.4f}
  Total Rounds: {fb_df['total_agg'].iloc[-1]}
  Total Time: {fb_df['time_min'].iloc[-1]:.2f} minutes

Difference:
  Accuracy: {tw_df['test_acc'].max() - fb_df['test_acc'].max():+.4f}
  Time: {tw_df['time_min'].iloc[-1] - fb_df['time_min'].iloc[-1]:+.2f} minutes
"""
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(comparison_dir / f"{exp_id}_trustweight_vs_fedbuff.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"✅ Individual comparison plots saved to: {comparison_dir}")

# ========== 4. Overall Statistics ==========
print("\n" + "="*100)
print("OVERALL STATISTICS")
print("="*100)

avg_tw_acc = np.mean([all_data[exp]["TrustWeight"]['test_acc'].max() for exp in exp_ids])
avg_fb_acc = np.mean([all_data[exp]["FedBuff"]['test_acc'].max() for exp in exp_ids])
avg_tw_time = np.mean([all_data[exp]["TrustWeight"]['time_min'].iloc[-1] for exp in exp_ids])
avg_fb_time = np.mean([all_data[exp]["FedBuff"]['time_min'].iloc[-1] for exp in exp_ids])

print(f"\nAverage Best Accuracy:")
print(f"  TrustWeight: {avg_tw_acc:.4f}")
print(f"  FedBuff:     {avg_fb_acc:.4f}")
print(f"  Difference:  {avg_tw_acc - avg_fb_acc:+.4f} ({((avg_tw_acc - avg_fb_acc) / avg_fb_acc * 100):+.2f}%)")

print(f"\nAverage Training Time:")
print(f"  TrustWeight: {avg_tw_time:.2f} minutes")
print(f"  FedBuff:    {avg_fb_time:.2f} minutes")
print(f"  Difference: {avg_tw_time - avg_fb_time:+.2f} minutes ({((avg_tw_time - avg_fb_time) / avg_fb_time * 100):+.2f}%)")

# Count wins
tw_wins = sum(1 for exp in exp_ids if all_data[exp]["TrustWeight"]['test_acc'].max() > all_data[exp]["FedBuff"]['test_acc'].max())
fb_wins = len(exp_ids) - tw_wins

print(f"\nExperiment Wins:")
print(f"  TrustWeight: {tw_wins}/{len(exp_ids)}")
print(f"  FedBuff:     {fb_wins}/{len(exp_ids)}")

print("\n" + "="*100)
print("✅ Comparison complete! All results saved to:", comparison_dir)
print("="*100)


