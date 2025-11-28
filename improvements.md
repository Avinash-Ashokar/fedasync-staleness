# FedBuff Implementation – Issues & Improvements

## 1. Root cause of low global accuracy

**Symptom:**  

Global `test_acc` remains around 9–10% (random for CIFAR-10), even though local client training shows reasonable `train_acc` (e.g., 0.6–0.7).

### 1.1. Server step size (eta) too small

In `BufferedFedServer._flush_buffer`, the global model is updated as:

```python
merged = [(1.0 - self.eta) * gi + self.eta * aggregated[i] for i, gi in enumerate(g)]
```

With eta = 0.00125 and only max_rounds = 10, the global model moves by only ~1.25% toward the aggregated client model after all rounds. This is effectively "no learning" at the server.

**Fixes / decisions:**

- **Baseline FedBuff (FedAvg):**
  Set eta = 1.0 and simply replace the global model with the aggregated one:
  ```python
  merged = aggregated
  ```

- **Relaxed update variant:**
  Keep the convex combination but use a larger eta (e.g. 0.05–0.1) and increase max_rounds (e.g. 100–300).

**Config change example:**

```yaml
buff:
  buffer_size: 5
  buffer_timeout_s: 10.0
  use_sample_weighing: true
  eta: 1.0    # for baseline, or 0.1 for relaxed updates
```

### 1.2. Client checkpoints overriding global model

In `LocalBuffClient.fit_once`, we load the latest global model:

```python
params, version = server.get_global()
self._from_list(params)
```

but then call Lightning with:

```python
ckpt = self.ckpt_path if Path(self.ckpt_path).exists() else None
trainer.fit(self.lit, train_dataloaders=self.loader, ckpt_path=ckpt)
```

If `last.ckpt` exists, Lightning restores from the checkpoint, which overwrites the freshly loaded global weights. From round 2 onward, each client continues from its own previous local model instead of the server's updated global model.

**Fix:**

- Remove `ckpt_path` when calling `trainer.fit`.
- Optionally disable checkpointing for normal FL runs.

**Updated code:**

```python
def fit_once(self, server) -> bool:
    params, version = server.get_global()
    self._from_list(params)
    
    self._sleep_delay()
    
    epochs = int(self.cfg["clients"]["local_epochs"])
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=self.accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        callbacks=[],
    )
    start = time.time()
    trainer.fit(self.lit, train_dataloaders=self.loader)
    duration = time.time() - start
    ...
```

## 2. Logging behaviour (zeros in avg_train_loss/acc)

The CSV log shows many rows with `avg_train_loss = 0.0` and `avg_train_acc = 0.0`. This is due to how we accumulate and clear training stats:

- `_train_loss_acc_accum` is only updated when `_flush_buffer()` is called (i.e., when a buffer of client updates is aggregated).
- `_periodic_eval_and_log()` computes the average and then clears `_train_loss_acc_accum`.

If the evaluation timer fires but no new buffer flush has happened since the last log, we log zeros. This is expected and not a functional bug.

**Possible improvement (optional):**

- Replace the "clear after log" strategy with an exponential moving average over time to avoid zero spikes.

## 3. Recommended training hyperparameters / next experiments

1. **Increase max_rounds**
   Start with:
   ```yaml
   train:
     max_rounds: 200
   ```
   and adjust when convergence speed is understood.

2. **Client training config**
   - Try `local_epochs = 5`.
   - Start with `lr = 0.01–0.05`, possibly with a simple LR scheduler in `LitCifar.configure_optimizers`.

3. **Hardware utilization**
   - Make sure clients and server actually use GPU/MPS when available (check device type in logs).

4. **Partitioning notes**
   - `partition_alpha = 0.1` creates a non-IID split. This is fine but leads to slower convergence than IID. Keep this in mind when interpreting curves.

## 4. Checklist

- [x] Remove `ckpt_path` from `trainer.fit` in `LocalBuffClient`.
- [x] Disable or simplify per-client checkpointing for standard runs.
- [x] Set eta to a meaningful value (1.0 for FedAvg-style FedBuff, or ≥0.05 for relaxed updates).
- [ ] Increase max_rounds for more realistic training.
- [ ] Rerun experiment and verify global test_acc rises above random baseline (~10%) on CIFAR-10.

