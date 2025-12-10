# FedBuff vs TrustWeight Implementation Comparison

## Overview
This document compares `damn.py` (FedBuff implementation) and `solution.py` (TrustWeight implementation) to highlight their key differences.

---

## 1. **Client Implementation Differences**

### FedBuff (`damn.py` - `LocalBuffClient`)
- **Training Framework**: Uses **PyTorch Lightning** (`LitCifar` class)
- **Model Management**: Maintains a persistent `LitCifar` instance with `self.lit`
- **Parameter Conversion**: Uses `_to_list()` and `_from_list()` to convert between state_dict and list formats
- **Method Name**: `fit_once(server)` - called once per round
- **Training**: Uses `pl.Trainer.fit()` for training
- **Metrics Collection**: Uses `self.lit.get_epoch_metrics()` to get train metrics
- **Test Evaluation**: Uses separate `_evaluate()` function with external testloader
- **Submission**: Submits `train_loss`, `train_acc`, `test_loss`, `test_acc` (no `delta_loss`)

### TrustWeight (`solution.py` - `AsyncClient`)
- **Training Framework**: **Direct PyTorch** (no Lightning)
- **Model Management**: Creates a new model instance each round via `_build_model()`
- **Parameter Conversion**: Works directly with state_dict (OrderedDict)
- **Method Name**: `run_once(server)` - called once per round
- **Training**: Direct SGD optimizer loop in `_train_local()`
- **Metrics Collection**: Manual accumulation in `_evaluate_on_loader()`
- **Test Evaluation**: Uses same `_evaluate_on_loader()` for both train and test
- **Delta Loss Calculation**: **Computes `delta_loss = loss_before - loss_after`** (critical for TrustWeight)
- **Submission**: Submits `delta_loss`, `loss_before`, `loss_after`, `train_acc`, `test_loss`, `test_acc`

**Key Difference**: TrustWeight computes `delta_loss` (loss improvement) which is essential for the quality term in aggregation.

---

## 2. **Server Implementation Differences**

### FedBuff (`damn.py` - `BufferedFedServer`)

#### Aggregation Logic (`_flush_buffer()`)
```python
# Simple weighted average aggregation
weight = num_samples / total_samples  # Sample-based weighting
aggregated[i] += weight * client_params[i]

# Update with eta mixing
if eta >= 1.0:
    merged = aggregated
else:
    merged = (1.0 - eta) * global + eta * aggregated
```

**Characteristics**:
- **Simple aggregation**: Weighted average based on sample counts
- **No staleness awareness**: Doesn't track or use staleness (τ)
- **No quality metrics**: Doesn't use loss improvement or update quality
- **No momentum**: No server-side momentum tracking
- **No projection/guard**: Direct parameter mixing

#### Version Tracking
- Uses `t_round` (integer round counter)
- Returns `(params_list, t_round)` from `get_global()`
- No version history maintained

#### Buffer Management
- Buffer stores raw client parameters (`new_params` as list of tensors)
- Flushes when buffer size reached or timeout

### TrustWeight (`solution.py` - `AsyncServer`)

#### Aggregation Logic (`_aggregate()`)
```python
# Complex trust-weighted aggregation via TrustWeightedAsyncStrategy
# 1. Compute update vectors: u_i = new_params - base_params
# 2. Compute staleness: τ_i = current_version - base_version
# 3. Apply TrustWeightedAsyncStrategy.aggregate():
#    - Freshness: s(τ) = exp(-α τ)
#    - Quality: exp(θᵀ [ΔL̃_i, ||u_i||, cos(u_i, m_t)])
#    - Projection: Proj_m(u_i)
#    - Guard: 1 / (1 + β1*τ + β2*||u_i||)
#    - Final: w_{t+1} = w_t + η * Σ Weight_i * [Proj + Guard*sideways]
```

**Characteristics**:
- **Staleness-aware**: Tracks version history and computes staleness (τ)
- **Quality-aware**: Uses `delta_loss`, update magnitude, and cosine similarity
- **Momentum-based**: Maintains server momentum `m_t` for projection
- **Projection + Guard**: Projects updates onto momentum, applies guard to sideways components
- **Sophisticated weighting**: Combines freshness, quality, and data share

#### Version Tracking
- Maintains `_model_versions` list (full version history)
- Uses `_version` counter
- Returns `(version, state_dict)` from `get_global_model()`
- Computes staleness: `τ_i = current_version - base_version`

#### Buffer Management
- Buffer stores `ClientUpdateState` dataclass objects
- Computes update vectors (`u_i = new - base`) during aggregation
- Applies update clipping (`update_clip_norm`)
- Filters NaN/Inf updates

---

## 3. **Strategy Component**

### FedBuff
- **No separate strategy class**
- Aggregation logic directly in `_flush_buffer()`
- Simple weighted average

### TrustWeight
- **`TrustWeightedAsyncStrategy` class** (lines 451-601 in `solution.py`)
- Encapsulates all aggregation math:
  - `_proj_m()`: Projection onto momentum
  - `_guard()`: Guard factor computation
  - `_freshness()`: Staleness decay
  - `aggregate()`: Main aggregation method
- Uses `theta` vector for quality weighting: `[delta_loss_coef, norm_coef, cosine_coef]`

---

## 4. **Configuration Differences**

### FedBuff Config Structure
```python
"buff": {
    "buffer_size": 5,
    "buffer_timeout_s": 0.0,
    "use_sample_weighing": True,
    "eta": 0.5
}
```

### TrustWeight Config Structure
```python
"trustweight": {
    "buffer_size": 5,
    "buffer_timeout_s": 0.0,
    "use_sample_weighing": True,
    "eta": 0.5,
    "theta": [1.0, -0.1, 0.2]  # Quality term weights
}
```

**Key Difference**: TrustWeight has `theta` parameter for quality weighting.

---

## 5. **Client Submission Differences**

### FedBuff `submit_update()` Signature
```python
server.submit_update(
    client_id, base_version, new_params, num_samples,
    train_time_s, train_loss, train_acc, test_loss, test_acc
)
```

### TrustWeight `submit_update()` Signature
```python
server.submit_update(
    client_id, base_version, new_params, num_samples,
    train_time_s, delta_loss, loss_before, loss_after,
    train_acc, test_loss, test_acc
)
```

**Key Difference**: TrustWeight requires `delta_loss`, `loss_before`, `loss_after` for quality computation.

---

## 6. **Logging Differences**

### FedBuff Client Participation CSV
```csv
client_id, local_train_loss, local_train_acc, local_test_loss, local_test_acc, total_agg
```

### TrustWeight Client Participation CSV
```csv
client_id, local_train_loss, local_train_acc, local_test_loss, local_test_acc, total_agg, staleness
```

**Key Difference**: TrustWeight logs `staleness` (τ) per client update.

---

## 7. **Data Structure Differences**

### FedBuff Buffer Entry
```python
{
    "client_id": int,
    "base_version": int,
    "new_params": List[torch.Tensor],  # Already converted to list
    "num_samples": int,
    "train_loss": float,
    "train_acc": float,
    "test_loss": float,
    "test_acc": float,
}
```

### TrustWeight Buffer Entry (`ClientUpdateState`)
```python
@dataclass
class ClientUpdateState:
    client_id: int
    base_version: int
    new_params: OrderedDict[str, torch.Tensor]  # State dict format
    num_samples: int
    train_time_s: float
    delta_loss: float  # Critical for quality term
    loss_before: float
    loss_after: float
    train_acc: float
    test_loss: float
    test_acc: float
    arrival_ts: float
```

**Key Differences**:
- TrustWeight stores `OrderedDict` (state_dict), FedBuff stores `List[torch.Tensor]`
- TrustWeight includes `delta_loss`, `loss_before`, `loss_after`
- TrustWeight includes `arrival_ts` timestamp

---

## 8. **Model Version History**

### FedBuff
- **No version history**
- Only tracks current global model
- `base_version` is just a round number, not used for staleness computation

### TrustWeight
- **Maintains full version history** (`_model_versions` list)
- Each aggregation creates a new version snapshot
- Uses version history to compute update vectors: `u_i = new_params - base_version_params`
- Computes staleness: `τ_i = current_version - base_version`

---

## 9. **Update Vector Computation**

### FedBuff
- **No update vector computation**
- Directly aggregates client model parameters
- No difference between base and new model

### TrustWeight
- **Computes update vectors** during aggregation:
  ```python
  base_vec = _flatten_state_by_template(base_state, template)
  new_vec = _flatten_state_by_template(new_params, template)
  ui = new_vec - base_vec  # Update vector
  ```
- Uses update vectors for projection, guard, and quality computation

---

## 10. **Summary of Core Algorithmic Differences**

| Aspect | FedBuff | TrustWeight |
|--------|---------|-------------|
| **Aggregation** | Simple weighted average | Trust-weighted with freshness, quality, projection, guard |
| **Staleness** | Not used | Exponential decay: `exp(-α τ)` |
| **Quality** | Not used | Quality term: `exp(θᵀ [ΔL, ||u||, cos(u,m)])` |
| **Momentum** | None | Server momentum `m_t` for projection |
| **Projection** | None | Projects updates onto momentum direction |
| **Guard** | None | Guards sideways components: `1/(1 + β1*τ + β2*||u||)` |
| **Version History** | No | Yes (for staleness and update vector computation) |
| **Delta Loss** | Not computed | Computed and used for quality |
| **Update Clipping** | No | Yes (`update_clip_norm`) |
| **Training Framework** | PyTorch Lightning | Direct PyTorch |

---

## 11. **Code Organization**

### FedBuff
- Client and server in same file
- No separate strategy class
- Simpler, more straightforward structure

### TrustWeight
- Client, server, and strategy in same file
- Strategy class encapsulates complex math
- More modular but more complex structure

---

## Conclusion

**FedBuff** is a simpler buffered asynchronous FL method that:
- Aggregates client updates using sample-weighted averaging
- Uses a simple `eta` mixing parameter
- No staleness or quality awareness

**TrustWeight** is a sophisticated trust-weighted method that:
- Uses staleness-aware freshness weighting
- Incorporates quality metrics (loss improvement, update magnitude, alignment)
- Projects updates onto server momentum
- Applies guard factors to sideways components
- Maintains version history for accurate staleness computation

The key innovation in TrustWeight is the combination of freshness, quality, projection, and guard terms to intelligently weight client updates based on their staleness, training quality, and alignment with global momentum.


