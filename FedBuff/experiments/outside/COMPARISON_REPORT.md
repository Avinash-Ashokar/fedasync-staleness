# TrustWeight vs FedBuff Comparison Report

## Executive Summary

This report compares the performance of **TrustWeight** and **FedBuff** federated learning algorithms across 6 different experimental configurations on the CIFAR-10 dataset.

### Key Findings

- **FedBuff significantly outperforms TrustWeight in accuracy** (53% better on average)
- **TrustWeight is much faster** (72% faster, ~4 minutes vs ~15 minutes)
- **FedBuff wins all 6 experiments** in terms of best accuracy achieved
- **TrustWeight appears to have stopped early** (24 rounds vs ~100 rounds for FedBuff)

---

## Detailed Results

### Overall Statistics

| Metric | TrustWeight | FedBuff | Difference |
|--------|------------|---------|------------|
| **Average Best Accuracy** | 0.1231 (12.31%) | 0.2618 (26.18%) | **-0.1387 (-53.00%)** |
| **Average Training Time** | 4.15 minutes | 15.06 minutes | **-10.91 minutes (-72.45%)** |
| **Experiment Wins** | 0/6 | 6/6 | - |

### Experiment-by-Experiment Comparison

#### Exp1: IID (alpha=1000), no stragglers
- **TrustWeight**: Best Acc = 13.55%, Final Acc = 13.46%, Time = 3.95 min, Rounds = 24
- **FedBuff**: Best Acc = 37.99%, Final Acc = 37.50%, Time = 15.27 min, Rounds = 99
- **Winner**: FedBuff (by 24.44% accuracy)

#### Exp2: alpha=0.1, 10% stragglers
- **TrustWeight**: Best Acc = 12.05%, Final Acc = 10.11%, Time = 4.18 min, Rounds = 24
- **FedBuff**: Best Acc = 23.40%, Final Acc = 19.64%, Time = 14.84 min, Rounds = 98
- **Winner**: FedBuff (by 11.35% accuracy)

#### Exp3: alpha=0.1, 20% stragglers
- **TrustWeight**: Best Acc = 12.04%, Final Acc = 10.16%, Time = 4.16 min, Rounds = 24
- **FedBuff**: Best Acc = 22.44%, Final Acc = 19.21%, Time = 14.86 min, Rounds = 100
- **Winner**: FedBuff (by 10.40% accuracy)

#### Exp4: alpha=0.1, 30% stragglers
- **TrustWeight**: Best Acc = 12.06%, Final Acc = 10.25%, Time = 4.18 min, Rounds = 24
- **FedBuff**: Best Acc = 23.29%, Final Acc = 18.65%, Time = 15.29 min, Rounds = 100
- **Winner**: FedBuff (by 11.23% accuracy)

#### Exp5: alpha=0.1, 40% stragglers
- **TrustWeight**: Best Acc = 12.06%, Final Acc = 9.93%, Time = 4.17 min, Rounds = 24
- **FedBuff**: Best Acc = 23.93%, Final Acc = 21.31%, Time = 14.78 min, Rounds = 99
- **Winner**: FedBuff (by 11.87% accuracy)

#### Exp6: alpha=0.1, 50% stragglers
- **TrustWeight**: Best Acc = 12.07%, Final Acc = 10.10%, Time = 4.23 min, Rounds = 24
- **FedBuff**: Best Acc = 26.02%, Final Acc = 19.81%, Time = 15.31 min, Rounds = 100
- **Winner**: FedBuff (by 13.95% accuracy)

---

## Analysis

### Accuracy Performance

1. **FedBuff consistently achieves 2x better accuracy** across all experiments
2. **TrustWeight accuracy is very low** (~12%), suggesting the model may not be learning effectively
3. **FedBuff shows better performance with more stragglers** (Exp6 achieves highest accuracy at 26.02%)

### Training Efficiency

1. **TrustWeight is 3.6x faster** than FedBuff (4.15 min vs 15.06 min average)
2. **TrustWeight stops much earlier** (24 rounds vs ~100 rounds)
3. This suggests TrustWeight may have:
   - A different stopping criterion
   - A bug causing early termination
   - Different hyperparameters affecting convergence

### Convergence Patterns

- **FedBuff**: Shows steady improvement over ~100 rounds, reaching 20-38% accuracy
- **TrustWeight**: Stops at 24 rounds with minimal learning (accuracy remains near random ~12%)

---

## Possible Explanations

### Why TrustWeight Performs Poorly

1. **Early Stopping**: TrustWeight may have stopped too early (24 rounds) before meaningful learning occurred
2. **Hyperparameter Issues**: Learning rate, aggregation method, or trust weighting mechanism may need tuning
3. **Implementation Differences**: TrustWeight may have different aggregation logic that's less effective
4. **Stopping Criterion**: May have hit a stopping condition (time limit, target accuracy, etc.) prematurely

### Why FedBuff Performs Better

1. **More Training Rounds**: ~100 rounds allows for better convergence
2. **Buffered Aggregation**: Collecting multiple updates before aggregating may provide more stable learning
3. **Better Hyperparameters**: The configuration may be better tuned for this task

---

## Recommendations

1. **Investigate TrustWeight Early Stopping**: Check why TrustWeight stops at 24 rounds
2. **Tune TrustWeight Hyperparameters**: Adjust learning rate, trust weighting mechanism, or aggregation parameters
3. **Run TrustWeight for More Rounds**: Allow TrustWeight to train for similar number of rounds as FedBuff
4. **Compare Convergence Rates**: Analyze accuracy per round to see if TrustWeight learns faster initially
5. **Check Implementation**: Verify TrustWeight aggregation logic matches the intended algorithm

---

## Files Generated

All comparison results are saved in `logs/comparisons/`:

- `trustweight_vs_fedbuff_summary.csv` - Detailed numerical comparison
- `trustweight_vs_fedbuff_comparison.png` - Overall comparison plots
- `Exp1-6_trustweight_vs_fedbuff.png` - Individual experiment comparisons

---

## Conclusion

**FedBuff significantly outperforms TrustWeight** in terms of accuracy across all experimental configurations. However, TrustWeight is much faster, suggesting it may have stopped early or has different convergence characteristics. Further investigation is needed to understand why TrustWeight stops at 24 rounds and whether extending its training would improve results.

**For accuracy-critical applications**: Use **FedBuff**
**For time-critical applications**: Investigate why TrustWeight stops early and tune for better accuracy


