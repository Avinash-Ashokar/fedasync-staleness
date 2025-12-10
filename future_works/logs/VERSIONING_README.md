# Versioned Test Results

This directory contains versioned test results from extended runs of the improved TrustWeight implementation.

## Directory Structure

Each extended test run creates a timestamped directory:
```
logs/
├── run_200agg_YYYYMMDD_HHMMSS/
│   ├── Eval.csv              # Global model evaluation log
│   ├── Participation.csv      # Client participation log
│   └── SUMMARY.txt            # Test summary
└── run_200agg_output.log      # Console output log
```

## Versioning Format

- **Format**: `run_200agg_YYYYMMDD_HHMMSS`
- **Example**: `run_200agg_20251210_143022`
- **Purpose**: Prevents overwriting results from multiple test runs

## Files in Each Versioned Directory

### `Eval.csv`
Global model evaluation results at key aggregation points (every 5 aggregations).

**Columns:**
- `total_agg`: Aggregation number
- `avg_train_loss`: Average training loss across clients
- `avg_train_acc`: Average training accuracy across clients
- `test_loss`: Test set loss
- `test_acc`: Test set accuracy
- `time`: Timestamp

### `Participation.csv`
Per-client participation log showing all client updates.

**Columns:**
- `client_id`: Client identifier
- `local_train_loss`: Client's training loss
- `local_train_acc`: Client's training accuracy
- `local_test_loss`: Client's test loss
- `local_test_acc`: Client's test accuracy
- `total_agg`: Aggregation number when this update was processed
- `staleness`: Staleness (τ) of the update

### `SUMMARY.txt`
Quick reference summary of the test run including:
- Test date and time
- Total execution time
- Total aggregations completed
- Final accuracy
- Accuracy improvement

## Current Test Runs

### Quick Tests (10 aggregations)
- `TrustWeightImproved_Eval.csv` - Initial quick test
- `TrustWeightImproved_Participation.csv` - Initial quick test participation

### Extended Test (30 aggregations)
- `TrustWeightImproved_Extended_Participation.csv` - Extended test participation
- See `EXTENDED_TEST_ANALYSIS.md` for detailed analysis

### Long-term Test (200 aggregations)
- Versioned directories: `run_200agg_YYYYMMDD_HHMMSS/`
- These are the most comprehensive tests showing long-term performance

## Analysis

For detailed analysis of test results, see:
- `../EXTENDED_TEST_ANALYSIS.md` - Analysis of 30-aggregation test
- `../SUMMARY.md` - Overall summary of improvements
- `../TEST_RESULTS.md` - Initial test results

## Best Practices

1. **Never delete versioned directories** - They represent historical test runs
2. **Always use versioned directories** for long tests (50+ aggregations)
3. **Document significant findings** in the main analysis files
4. **Commit results** after each major test run

