# Step-by-Step Implementation of Straggler Improvements

## Quick Summary
Apply these changes to make TrustWeight more robust against stragglers (Exp5 & Exp6).

---

## Step 1: Update Exp5 & Exp6 Configs

**Location**: Cell 4 (experiment configurations)

**Find Exp5's `trustweight` section** (around line 1377-1385) and change:
```python
"trustweight": {
    "buffer_size": 5,
    "buffer_timeout_s": 0.0,
    "use_sample_weighing": True,
    "eta": 0.4,                 # ← Changed from 0.5
    "theta": [0.8, -0.05, 0.1], # ← Changed from [1.0, -0.1, 0.2]
    "freshness_alpha": 0.3,     # ← Changed from 0.1
    "beta1": 0.15,              # ← Changed from 0.0
    "beta2": 0.02               # ← Changed from 0.0
},
```

**Find Exp6's `trustweight` section** (around line 1423-1431) and make the same changes.

---

## Step 2: Add Hard Staleness Threshold

**Location**: Cell 4 (AsyncServer class), in the `_aggregate` method

**Find this line** (around line 1027):
```python
tau_i = float(max(0, version_now - u.base_version))
staleness_list.append(tau_i)
```

**Add after it**:
```python
tau_i = float(max(0, version_now - u.base_version))

# Drop extremely stale updates (hard threshold for straggler robustness)
max_tau = 10.0
if tau_i > max_tau:
    print(f"[Server] Dropping client {u.client_id} update (tau={tau_i:.1f} > {max_tau})")
    continue

staleness_list.append(tau_i)
```

---

## Step 3: Add Staleness Penalty to Quality Term

**Location**: Cell 3 (TrustWeightedAsyncStrategy class), in the `aggregate` method

**Find this section** (around line 656-663):
```python
# Quality term: exp(θᵀ [ΔL̃ᵢ, ||uᵢ||, cos(uᵢ, m_t)])
# θ: (ΔL weight, -||u|| penalty, cosine alignment)
feats = torch.stack(
    [delta_losses, norm_u_tensor, cos_tensor],
    dim=1,
)  # [B, 3]
quality_logits = feats @ self.theta.to(device)
quality = torch.exp(quality_logits)
```

**Replace with**:
```python
# Add scaled staleness as a negative feature (straggler robustness)
lambda_stale = 0.05
effective_delta = delta_losses - lambda_stale * taus

# Quality term: exp(θᵀ [ΔL̃ᵢ, ||uᵢ||, cos(uᵢ, m_t)])
# θ: (ΔL weight, -||u|| penalty, cosine alignment)
feats = torch.stack(
    [effective_delta, norm_u_tensor, cos_tensor],  # ← Changed delta_losses to effective_delta
    dim=1,
)  # [B, 3]
quality_logits = feats @ self.theta.to(device)

# Clamp logits to avoid exploding weights (straggler robustness)
quality_logits = torch.clamp(quality_logits, min=-3.0, max=3.0)
quality = torch.exp(quality_logits)
```

---

## Step 4: Reduce Epochs for Slow Clients (Optional)

**Location**: Cell 2 (AsyncClient class), in `__init__`

**Find this line** (around line 371):
```python
self.local_epochs = cfg["clients"]["local_epochs"]
```

**Replace with**:
```python
# Reduce epochs for slow clients (straggler robustness)
base_epochs = cfg["clients"]["local_epochs"]
if self.is_slow:
    self.local_epochs = max(1, base_epochs // 2)  # half epochs for slow clients
else:
    self.local_epochs = base_epochs
```

---

## Verification Checklist

After making changes, verify:

- [ ] Exp5 `trustweight` config has: eta=0.4, freshness_alpha=0.3, beta1=0.15, beta2=0.02
- [ ] Exp6 `trustweight` config has: eta=0.4, freshness_alpha=0.3, beta1=0.15, beta2=0.02
- [ ] `_aggregate` method has hard staleness threshold (max_tau=10)
- [ ] `aggregate` method uses `effective_delta` instead of `delta_losses`
- [ ] `aggregate` method clamps `quality_logits` to [-3, 3]
- [ ] (Optional) `AsyncClient.__init__` reduces epochs for slow clients

---

## Expected Impact

After these changes:
- **Exp5 (40% stragglers)**: Should show improved stability and accuracy
- **Exp6 (50% stragglers)**: Should match or beat FedBuff
- **Learning curves**: Should be smoother with fewer spikes
- **Staleness handling**: Very stale updates (τ > 10) are completely dropped

---

## Testing

1. Run Exp5 and Exp6 with new settings
2. Compare with FedBuff results
3. Check if TrustWeight degrades less as straggler % increases
4. Verify curves are more stable

---

## Notes

- These changes are **only for Exp5 & Exp6** (high straggler scenarios)
- Exp1-Exp4 keep original settings
- The improvements make TrustWeight more "aggressive" against stale updates
- This should help TrustWeight outperform FedBuff in high-straggler cases


