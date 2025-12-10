# TrustWeight Straggler-Focused Improvements Guide

This guide implements the feedback to make TrustWeight more robust against stragglers (40-50% in Exp5 & Exp6).

## Summary of Changes

### 1. Config Changes for Exp5 & Exp6

Update the `trustweight` section in Exp5 and Exp6:

```python
"trustweight": {
    "buffer_size": 5,
    "buffer_timeout_s": 0.0,
    "use_sample_weighing": True,
    "eta": 0.4,                 # Changed from 0.5 (lower LR for stability)
    "theta": [0.8, -0.05, 0.1], # Changed from [1.0, -0.1, 0.2] (gentler quality term)
    "freshness_alpha": 0.3,     # Changed from 0.1 (MUCH stronger staleness penalty)
    "beta1": 0.15,              # Changed from 0.0 (now actually use Guard)
    "beta2": 0.02               # Changed from 0.0
},
```

**Effect**: For τ=5 (staleness of 5 rounds):
- Old: s(τ) = exp(-0.1·5) ≈ 0.61
- New: s(τ) = exp(-0.3·5) ≈ 0.22 → stale updates heavily crushed

---

### 2. Code Changes

#### 2.1. Hard Staleness Threshold in `AsyncServer._aggregate`

**Location**: In the `_aggregate` method, after computing `tau_i`

**Add this code** after the line `tau_i = float(max(0, version_now - u.base_version))`:

```python
# Drop extremely stale updates (hard threshold for straggler robustness)
max_tau = 10.0
if tau_i > max_tau:
    print(f"[Server] Dropping client {u.client_id} update (tau={tau_i:.1f} > {max_tau})")
    continue
```

**Effect**: Updates with staleness > 10 rounds are completely ignored.

---

#### 2.2. Staleness Penalty in Quality Term

**Location**: In `TrustWeightedAsyncStrategy.aggregate`, before building `feats`

**Replace this section**:
```python
# Quality term: exp(θᵀ [ΔL̃ᵢ, ||uᵢ||, cos(uᵢ, m_t)])
feats = torch.stack(
    [delta_losses, norm_u_tensor, cos_tensor],
    dim=1,
)  # [B, 3]
quality_logits = feats @ self.theta.to(device)
quality = torch.exp(quality_logits)
```

**With this**:
```python
# Add scaled staleness as a negative feature (straggler robustness)
lambda_stale = 0.05
effective_delta = delta_losses - lambda_stale * taus

# Quality term: exp(θᵀ [ΔL̃ᵢ, ||uᵢ||, cos(uᵢ, m_t)])
feats = torch.stack(
    [effective_delta, norm_u_tensor, cos_tensor],
    dim=1,
)  # [B, 3]
quality_logits = feats @ self.theta.to(device)
```

**Effect**: Even if ΔL is big, large τ reduces the effective quality, preventing stale-but-locally-good updates from being over-rewarded.

---

#### 2.3. Clamp Quality Logits

**Location**: In `TrustWeightedAsyncStrategy.aggregate`, after computing `quality_logits`

**Add this** right after `quality_logits = feats @ self.theta.to(device)`:

```python
# Clamp logits to avoid exploding weights (straggler robustness)
quality_logits = torch.clamp(quality_logits, min=-3.0, max=3.0)
quality = torch.exp(quality_logits)
```

**Effect**: No client's quality term exceeds exp(3) ≈ 20 or goes below exp(-3) ≈ 0.05, preventing a single straggler from dominating.

---

#### 2.4. Reduce Local Epochs for Slow Clients (Optional)

**Location**: In `AsyncClient.__init__`, replace:

```python
self.local_epochs = cfg["clients"]["local_epochs"]
```

**With**:
```python
# Reduce epochs for slow clients (straggler robustness)
base_epochs = cfg["clients"]["local_epochs"]
if self.is_slow:
    self.local_epochs = max(1, base_epochs // 2)  # half epochs for slow clients
else:
    self.local_epochs = base_epochs
```

**Effect**: Slow clients do less local work on stale models, reducing drift from the current global direction.

---

## Testing

After applying these changes:

1. **Rerun Exp5 & Exp6** with the new settings
2. **Compare with FedBuff** on:
   - Best test accuracy
   - Time to reach 50%/60% accuracy
   - Curve stability (fewer spikes)

**Expected Results**:
- TrustWeight should degrade less severely as straggler % increases
- In 50% straggler case, TrustWeight should match or surpass FedBuff
- More stable learning curves with fewer weird spikes

---

## Quick Reference: What Each Change Does

| Change | Purpose | Impact |
|--------|---------|--------|
| `freshness_alpha: 0.3` | Stronger staleness penalty | Stale updates get much lower weight |
| `beta1: 0.15, beta2: 0.02` | Enable Guard terms | Large/stale updates have sideways movement crushed |
| `eta: 0.4` | Lower learning rate | More stable when punishing stale updates |
| `theta: [0.8, -0.05, 0.1]` | Gentler quality term | Prevents one client from dominating |
| Hard threshold (tau > 10) | Drop extremely stale updates | Completely ignore very outdated updates |
| Staleness in quality term | Penalize stale updates in quality | Even good local updates get penalized if stale |
| Clamp logits | Prevent exploding weights | Bounds the influence of any single client |
| Reduce epochs for slow | Less local work on stale models | Reduces drift from current global direction |

---

## Files to Modify

1. **trustweight.ipynb**:
   - Cell 4: Update Exp5 & Exp6 `trustweight` configs
   - Cell 3 (TrustWeightedAsyncStrategy): Add staleness penalty and clamping
   - Cell 4 (AsyncServer._aggregate): Add hard staleness threshold
   - Cell 2 (AsyncClient): Optionally reduce epochs for slow clients

---

## Notes

- These changes are **only applied to Exp5 & Exp6** (40-50% stragglers)
- Exp1-Exp4 keep their original settings
- The improvements make TrustWeight more "aggressive" against stale updates
- This should help TrustWeight outperform FedBuff in high-straggler scenarios


