# Coverage Threshold Upgrade: 0.5 → 0.9

## Summary

Upgraded coverage quality threshold from 50% to 90% confidence for more meaningful metrics.

## Changes Made

### 1. Primary Threshold (`coverage_env.py` line 1240)
```python
# Before:
covered = (self.state.coverage > 0.5).sum()  # 50% confidence

# After:
covered = (self.state.coverage > 0.9).sum()  # 90% confidence
```

### 2. Multi-Threshold Tracking (`coverage_env.py` line 1222)
Added to info dict:
- `cells_ever_sensed`: Any sensing (>0%)
- `coverage_pct_50`: Low confidence (>50%)
- `coverage_pct_70`: Medium confidence (>70%)
- `coverage_pct_90`: High confidence (>90%) ← **Primary metric**
- `coverage_pct_95`: Very high confidence (>95%)

### 3. Episode Termination (`coverage_env.py` line 1200)
```python
# Before:
done = (self.state.step >= self.max_steps or coverage_pct >= 0.99)

# After:
done = (self.state.step >= self.max_steps or coverage_pct >= 0.95)
```

## Why This Change?

### Problem with 0.5 Threshold
- Single marginal sensing event → cell counted as "covered"
- Inflated coverage percentages
- Doesn't reflect mission-critical quality
- 50% confidence = 1 in 2 chance of false coverage

### Benefits of 0.9 Threshold
- Requires multiple sensing passes or very close sensing
- 90% confidence = mission-ready reliability
- Aligns with Bayesian accumulation design
- More conservative, meaningful metrics

## Impact

### On Metrics
- **Coverage numbers will be lower** (e.g., 35-50% instead of 65-75%)
- **But quality is much higher** (each cell has 90%+ confidence)
- Expected pyramid after training:
  ```
  Sensed:  85-95% (agent explores most of map)
  ≥50%:    75-85% (most explored cells have some confidence)
  ≥70%:    55-70% (many cells have medium confidence)
  ≥90%:    35-50% (good portion have high confidence) ← Primary
  ≥95%:    20-35% (core areas have very high confidence)
  ```

### On Rewards
- **No change** - rewards still use continuous probabilities
- Agent still rewarded for all exploration
- Learning dynamics unchanged

### On Training
- **Slightly harder** early completion (needs 95% of 90%+ cells)
- **More thorough** coverage required
- **Higher quality** final coverage

## Testing

Run verification:
```bash
python test_coverage_threshold.py
```

Expected output:
```
✅ TEST 1 PASSED: Coverage threshold is 0.9
✅ TEST 2 PASSED: Multi-threshold metrics form proper pyramid  
✅ TEST 3 PASSED: Rewards use continuous probabilities
✅ TEST 4 PASSED: Termination threshold is 0.95
✅ TEST 5 PASSED: Quality metrics demonstrate improvement
```

## Validation

After training completes, verify:
- Coverage numbers are lower but pyramid-distributed
- Rewards still substantial for exploration
- Agent completes episodes when 95% of cells have 90%+ confidence
- Final coverage quality is high

## Files Modified

1. **coverage_env.py**:
   - Line 1240: Changed threshold from 0.5 to 0.9
   - Line 1222: Added multi-threshold metrics to info dict
   - Line 1200: Changed termination from 0.99 to 0.95

2. **test_coverage_threshold.py** (new):
   - Comprehensive verification of all changes

## Note

The old `environment.py` already used 0.9 threshold. This change **restores the original, correct design** that was inadvertently changed to 0.5 in the new `coverage_env.py`.
