#!/usr/bin/env python3
"""
Script to apply straggler-focused improvements to TrustWeight notebook.
This updates Exp5 and Exp6 configs and adds code improvements.
"""

import json
from pathlib import Path

notebook_path = Path("trustweight.ipynb")

# Read notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find the cell with experiment configs (cell index 4)
config_cell = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'Exp5' in cell['source'][0] if cell['source'] else False:
        config_cell = i
        break

if config_cell is None:
    print("❌ Could not find experiment config cell")
    exit(1)

# Get the source code
source_lines = notebook['cells'][config_cell]['source']

# Update Exp5 config
exp5_start = None
exp5_end = None
for i, line in enumerate(source_lines):
    if '"Exp5":' in line:
        exp5_start = i
    if exp5_start and '"io": get_paths("Exp5")' in line:
        exp5_end = i + 1
        break

if exp5_start and exp5_end:
    # Find trustweight section in Exp5
    for i in range(exp5_start, exp5_end):
        if '"trustweight":' in source_lines[i]:
            # Update the trustweight config
            for j in range(i, min(i+10, exp5_end)):
                if '"eta": 0.5,' in source_lines[j]:
                    source_lines[j] = '            "eta": 0.4,\n'
                elif '"theta": [1.0, -0.1, 0.2],' in source_lines[j]:
                    source_lines[j] = '            "theta": [0.8, -0.05, 0.1],\n'
                elif '"freshness_alpha": 0.1,' in source_lines[j]:
                    source_lines[j] = '            "freshness_alpha": 0.3,\n'
                elif '"beta1": 0.0,' in source_lines[j]:
                    source_lines[j] = '            "beta1": 0.15,\n'
                elif '"beta2": 0.0' in source_lines[j] and j < exp5_end - 5:
                    source_lines[j] = '            "beta2": 0.02\n'

# Update Exp6 config
exp6_start = None
exp6_end = None
for i, line in enumerate(source_lines):
    if '"Exp6":' in line:
        exp6_start = i
    if exp6_start and '"io": get_paths("Exp6")' in line:
        exp6_end = i + 1
        break

if exp6_start and exp6_end:
    # Find trustweight section in Exp6
    for i in range(exp6_start, exp6_end):
        if '"trustweight":' in source_lines[i]:
            # Update the trustweight config
            for j in range(i, min(i+10, exp6_end)):
                if '"eta": 0.5,' in source_lines[j]:
                    source_lines[j] = '            "eta": 0.4,\n'
                elif '"theta": [1.0, -0.1, 0.2],' in source_lines[j]:
                    source_lines[j] = '            "theta": [0.8, -0.05, 0.1],\n'
                elif '"freshness_alpha": 0.1,' in source_lines[j]:
                    source_lines[j] = '            "freshness_alpha": 0.3,\n'
                elif '"beta1": 0.0,' in source_lines[j]:
                    source_lines[j] = '            "beta1": 0.15,\n'
                elif '"beta2": 0.0' in source_lines[j] and j < exp6_end - 5:
                    source_lines[j] = '            "beta2": 0.02\n'

# Update the cell
notebook['cells'][config_cell]['source'] = source_lines

# Now update the TrustWeightedAsyncStrategy.aggregate method
strategy_cell = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'TrustWeightedAsyncStrategy' in ''.join(cell['source']):
        strategy_cell = i
        break

if strategy_cell:
    source_lines = notebook['cells'][strategy_cell]['source']
    new_source = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        new_source.append(line)
        
        # Add staleness penalty to quality term
        if 'feats = torch.stack(' in line and 'delta_losses' in source_lines[i+1] if i+1 < len(source_lines) else False:
            # Insert effective_delta calculation before feats
            new_source.append('        # Add scaled staleness as a negative feature (straggler robustness)\n')
            new_source.append('        lambda_stale = 0.05\n')
            new_source.append('        effective_delta = delta_losses - lambda_stale * taus\n')
            new_source.append('\n')
            # Modify the feats line
            i += 1
            if i < len(source_lines):
                # Replace delta_losses with effective_delta
                new_source.append('        feats = torch.stack(\n')
                i += 1
                if i < len(source_lines) and 'delta_losses' in source_lines[i]:
                    new_source.append('            [effective_delta, norm_u_tensor, cos_tensor],\n')
                    i += 1
                    if i < len(source_lines):
                        new_source.append(source_lines[i])  # dim=1 line
                        i += 1
                        continue
        
        # Add clamping for quality_logits
        if 'quality_logits = feats @ self.theta.to(device)' in line:
            new_source.append(line)
            i += 1
            # Insert clamping after quality_logits
            new_source.append('        # Clamp logits to avoid exploding weights (straggler robustness)\n')
            new_source.append('        quality_logits = torch.clamp(quality_logits, min=-3.0, max=3.0)\n')
            continue
        
        i += 1
    
    notebook['cells'][strategy_cell]['source'] = new_source

# Update AsyncServer._aggregate to add hard staleness threshold
server_cell = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'def _aggregate(self, updates:' in ''.join(cell['source']):
        server_cell = i
        break

if server_cell:
    source_lines = notebook['cells'][server_cell]['source']
    new_source = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        new_source.append(line)
        
        # Add hard staleness threshold after tau_i calculation
        if 'tau_i = float(max(0, version_now - u.base_version))' in line:
            new_source.append(line)
            i += 1
            # Insert hard cutoff
            new_source.append('            \n')
            new_source.append('            # Drop extremely stale updates (hard threshold for straggler robustness)\n')
            new_source.append('            max_tau = 10.0\n')
            new_source.append('            if tau_i > max_tau:\n')
            new_source.append('                print(f"[Server] Dropping client {u.client_id} update (tau={tau_i:.1f} > {max_tau})")\n')
            new_source.append('                continue\n')
            new_source.append('            \n')
            continue
        
        i += 1
    
    notebook['cells'][server_cell]['source'] = new_source

# Update AsyncClient to reduce epochs for slow clients
client_cell = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'class AsyncClient:' in ''.join(cell['source']):
        client_cell = i
        break

if client_cell:
    source_lines = notebook['cells'][client_cell]['source']
    new_source = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        
        # Replace local_epochs assignment
        if 'self.local_epochs = cfg["clients"]["local_epochs"]' in line:
            new_source.append('        # Reduce epochs for slow clients (straggler robustness)\n')
            new_source.append('        base_epochs = cfg["clients"]["local_epochs"]\n')
            new_source.append('        if self.is_slow:\n')
            new_source.append('            self.local_epochs = max(1, base_epochs // 2)  # half epochs for slow clients\n')
            new_source.append('        else:\n')
            new_source.append('            self.local_epochs = base_epochs\n')
            i += 1
            continue
        
        new_source.append(line)
        i += 1
    
    notebook['cells'][client_cell]['source'] = new_source

# Save updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Applied straggler-focused improvements to TrustWeight notebook:")
print("   - Updated Exp5 & Exp6 configs (eta=0.4, freshness_alpha=0.3, beta1=0.15, beta2=0.02)")
print("   - Added hard staleness threshold (max_tau=10)")
print("   - Added staleness penalty to quality term (lambda_stale=0.05)")
print("   - Added quality logits clamping ([-3, 3])")
print("   - Reduced local epochs for slow clients (half epochs)")


