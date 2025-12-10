#!/usr/bin/env python3
"""
Compare TrustWeight and FedBuff at round 20
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

def get_data_at_round(df, target_round):
    """Get data at or closest to target round."""
    # Find the row with total_agg closest to target_round
    df_sorted = df.sort_values('total_agg')
    closest_idx = (df_sorted['total_agg'] - target_round).abs().idxmin()
    closest_round = df_sorted.loc[closest_idx, 'total_agg']
    
    # Get the row
    row = df_sorted.loc[closest_idx]
    
    return row, closest_round

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

# ========== Extract data at round 20 ==========
target_round = 20
print(f"\n{'='*100}")
print(f"COMPARISON AT ROUND {target_round}")
print("="*100)

round_20_data = []

for exp_id in experiments.keys():
    if exp_id not in all_data:
        continue
    
    tw_df = all_data[exp_id]["TrustWeight"]
    fb_df = all_data[exp_id]["FedBuff"]
    exp_config = experiments[exp_id]
    
    # Get data at round 20 (or closest)
    tw_row, tw_actual_round = get_data_at_round(tw_df, target_round)
    fb_row, fb_actual_round = get_data_at_round(fb_df, target_round)
    
    round_20_data.append({
        "Exp": exp_id,
        "Config": exp_config["name"],
        "TW_Round": tw_actual_round,
        "TW_Acc": tw_row['test_acc'],
        "TW_Loss": tw_row['test_loss'],
        "TW_Time": tw_row['time_min'],
        "TW_Train_Acc": tw_row['avg_train_acc'],
        "TW_Train_Loss": tw_row['avg_train_loss'],
        "FB_Round": fb_actual_round,
        "FB_Acc": fb_row['test_acc'],
        "FB_Loss": fb_row['test_loss'],
        "FB_Time": fb_row['time_min'],
        "FB_Train_Acc": fb_row['avg_train_acc'],
        "FB_Train_Loss": fb_row['avg_train_loss'],
        "Acc_Diff": tw_row['test_acc'] - fb_row['test_acc'],
        "Time_Diff": tw_row['time_min'] - fb_row['time_min'],
    })

# Create DataFrame
round_20_df = pd.DataFrame(round_20_data)

# Print summary table
print("\nDetailed Comparison at Round 20:")
print(round_20_df.to_string(index=False, float_format='%.4f'))

# Save to CSV
round_20_df.to_csv(comparison_dir / f"trustweight_vs_fedbuff_at_round_{target_round}.csv", index=False)
print(f"\n✅ Round {target_round} comparison saved to: {comparison_dir / f'trustweight_vs_fedbuff_at_round_{target_round}.csv'}")

# ========== Statistics ==========
print("\n" + "="*100)
print("STATISTICS AT ROUND 20")
print("="*100)

avg_tw_acc = round_20_df['TW_Acc'].mean()
avg_fb_acc = round_20_df['FB_Acc'].mean()
avg_tw_time = round_20_df['TW_Time'].mean()
avg_fb_time = round_20_df['FB_Time'].mean()

print(f"\nAverage Test Accuracy at Round 20:")
print(f"  TrustWeight: {avg_tw_acc:.4f} ({avg_tw_acc*100:.2f}%)")
print(f"  FedBuff:     {avg_fb_acc:.4f} ({avg_fb_acc*100:.2f}%)")
print(f"  Difference:  {avg_tw_acc - avg_fb_acc:+.4f} ({((avg_tw_acc - avg_fb_acc) / avg_fb_acc * 100):+.2f}%)")

print(f"\nAverage Training Time to Reach Round 20:")
print(f"  TrustWeight: {avg_tw_time:.2f} minutes")
print(f"  FedBuff:     {avg_fb_time:.2f} minutes")
print(f"  Difference:  {avg_tw_time - avg_fb_time:+.2f} minutes ({((avg_tw_time - avg_fb_time) / avg_fb_time * 100):+.2f}%)")

# Count wins
tw_wins = sum(1 for diff in round_20_df['Acc_Diff'] if diff > 0)
fb_wins = len(round_20_df) - tw_wins

print(f"\nAccuracy Wins at Round 20:")
print(f"  TrustWeight: {tw_wins}/{len(round_20_df)}")
print(f"  FedBuff:     {fb_wins}/{len(round_20_df)}")

# ========== Create Visualization ==========
print("\n📊 Generating comparison plots at round 20...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

exp_ids = round_20_df['Exp'].values
tw_accs = round_20_df['TW_Acc'].values
fb_accs = round_20_df['FB_Acc'].values
tw_times = round_20_df['TW_Time'].values
fb_times = round_20_df['FB_Time'].values

x = np.arange(len(exp_ids))
width = 0.35

# 1. Accuracy Comparison at Round 20
axes[0, 0].bar(x - width/2, tw_accs, width, label='TrustWeight', alpha=0.8, color='#2ca02c')
axes[0, 0].bar(x + width/2, fb_accs, width, label='FedBuff', alpha=0.8, color='#1f77b4')
axes[0, 0].set_xlabel('Experiment', fontsize=12)
axes[0, 0].set_ylabel('Test Accuracy', fontsize=12)
axes[0, 0].set_title(f'Test Accuracy at Round 20', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(exp_ids, fontsize=10)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0, max(max(tw_accs), max(fb_accs)) * 1.15])

# Add value labels
for i, (tw, fb) in enumerate(zip(tw_accs, fb_accs)):
    axes[0, 0].text(i - width/2, tw, f'{tw:.3f}', ha='center', va='bottom', fontsize=8)
    axes[0, 0].text(i + width/2, fb, f'{fb:.3f}', ha='center', va='bottom', fontsize=8)

# 2. Accuracy Difference
acc_diffs = round_20_df['Acc_Diff'].values
colors = ['green' if d > 0 else 'red' for d in acc_diffs]
axes[0, 1].bar(exp_ids, acc_diffs, alpha=0.7, color=colors, edgecolor='black')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Experiment', fontsize=12)
axes[0, 1].set_ylabel('Accuracy Difference (TW - FB)', fontsize=12)
axes[0, 1].set_title('Accuracy Advantage at Round 20', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, diff in enumerate(acc_diffs):
    axes[0, 1].text(i, diff, f'{diff:+.3f}', ha='center', va='bottom' if diff > 0 else 'top', fontsize=9)

# 3. Time to Reach Round 20
axes[1, 0].bar(x - width/2, tw_times, width, label='TrustWeight', alpha=0.8, color='#ff7f0e')
axes[1, 0].bar(x + width/2, fb_times, width, label='FedBuff', alpha=0.8, color='#9467bd')
axes[1, 0].set_xlabel('Experiment', fontsize=12)
axes[1, 0].set_ylabel('Time to Round 20 (minutes)', fontsize=12)
axes[1, 0].set_title('Time to Reach Round 20', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(exp_ids, fontsize=10)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (tw, fb) in enumerate(zip(tw_times, fb_times)):
    axes[1, 0].text(i - width/2, tw, f'{tw:.1f}m', ha='center', va='bottom', fontsize=8)
    axes[1, 0].text(i + width/2, fb, f'{fb:.1f}m', ha='center', va='bottom', fontsize=8)

# 4. Accuracy Trajectory up to Round 20
for exp_id in exp_ids:
    tw_df = all_data[exp_id]["TrustWeight"]
    fb_df = all_data[exp_id]["FedBuff"]
    
    # Filter to rounds <= 20
    tw_subset = tw_df[tw_df['total_agg'] <= 20].sort_values('total_agg')
    fb_subset = fb_df[fb_df['total_agg'] <= 20].sort_values('total_agg')
    
    if len(tw_subset) > 0:
        axes[1, 1].plot(tw_subset['total_agg'], tw_subset['test_acc'], 'o-', 
                      linewidth=2, markersize=4, alpha=0.7, label=f'TW-{exp_id}')
    if len(fb_subset) > 0:
        axes[1, 1].plot(fb_subset['total_agg'], fb_subset['test_acc'], 's-', 
                      linewidth=2, markersize=4, alpha=0.7, label=f'FB-{exp_id}')

axes[1, 1].set_xlabel('Round', fontsize=12)
axes[1, 1].set_ylabel('Test Accuracy', fontsize=12)
axes[1, 1].set_title('Accuracy Trajectory: Rounds 1-20', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=7, ncol=3, loc='lower right')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 0.4])
axes[1, 1].axvline(x=20, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Round 20')

plt.tight_layout()
plt.savefig(comparison_dir / f"trustweight_vs_fedbuff_at_round_{target_round}.png", dpi=150, bbox_inches='tight')
print(f"✅ Round {target_round} comparison plot saved: {comparison_dir / f'trustweight_vs_fedbuff_at_round_{target_round}.png'}")
plt.close()

# ========== Individual Experiment Trajectories ==========
print("\n📊 Generating individual experiment trajectories up to round 20...")

for exp_id in exp_ids:
    tw_df = all_data[exp_id]["TrustWeight"]
    fb_df = all_data[exp_id]["FedBuff"]
    exp_config = experiments[exp_id]
    
    # Filter to rounds <= 20
    tw_subset = tw_df[tw_df['total_agg'] <= 20].sort_values('total_agg')
    fb_subset = fb_df[fb_df['total_agg'] <= 20].sort_values('total_agg')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy trajectory
    if len(tw_subset) > 0:
        axes[0].plot(tw_subset['total_agg'], tw_subset['test_acc'], 'o-', 
                   linewidth=2, markersize=5, label='TrustWeight', color='#2ca02c')
    if len(fb_subset) > 0:
        axes[0].plot(fb_subset['total_agg'], fb_subset['test_acc'], 's-', 
                   linewidth=2, markersize=5, label='FedBuff', color='#1f77b4')
    
    axes[0].axvline(x=20, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Round', fontsize=11)
    axes[0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0].set_title(f'{exp_id}: Accuracy Trajectory (Rounds 1-20)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 0.4])
    
    # Loss trajectory
    if len(tw_subset) > 0:
        axes[1].plot(tw_subset['total_agg'], tw_subset['test_loss'], 'o-', 
                   linewidth=2, markersize=5, label='TrustWeight', color='#d62728')
    if len(fb_subset) > 0:
        axes[1].plot(fb_subset['total_agg'], fb_subset['test_loss'], 's-', 
                   linewidth=2, markersize=5, label='FedBuff', color='#ff7f0e')
    
    axes[1].axvline(x=20, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Round', fontsize=11)
    axes[1].set_ylabel('Test Loss', fontsize=11)
    axes[1].set_title(f'{exp_id}: Loss Trajectory (Rounds 1-20)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / f"{exp_id}_trajectory_to_round_{target_round}.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"✅ Individual trajectory plots saved to: {comparison_dir}")

# ========== Summary Table ==========
print("\n" + "="*100)
print("DETAILED METRICS AT ROUND 20")
print("="*100)

summary_cols = ['Exp', 'TW_Round', 'TW_Acc', 'TW_Loss', 'TW_Time', 
                'FB_Round', 'FB_Acc', 'FB_Loss', 'FB_Time', 'Acc_Diff']
print(round_20_df[summary_cols].to_string(index=False, float_format='%.4f'))

print("\n" + "="*100)
print("✅ Round 20 comparison complete! All results saved to:", comparison_dir)
print("="*100)


