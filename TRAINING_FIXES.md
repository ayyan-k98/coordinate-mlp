# Training Stability Fixes

## 🔴 Critical Issues Identified

### 1. **Unbounded Q-Values**
- Target clipping is commented out (dqn_agent.py:306)
- Q-values growing to 13+ before exploding to infinity
- Gradient explosions in q_head.value_stream.bias

### 2. **Reward Scaling Problem**
- Rewards sum across all cells: `coverage_gain.sum()`
- 30×30 grid can produce 25+ reward per step
- Episode rewards reach 300+ (should be ~10-50)
- Q-network can't handle this scale variation

### 3. **Overfitting**
- Training coverage: 99%
- Validation coverage: 40-60%
- 40% generalization gap!

### 4. **Weak Gradient Control**
- Gradient norms climbing to 3.2+
- No explicit gradient norm clipping
- AGC at 0.01 is insufficient

---

## ✅ IMMEDIATE FIXES (Apply in Order)

### Fix 1: Enable Target Clipping (CRITICAL)
**File**: `dqn_agent.py` line 306

**Current**:
```python
#targets = torch.clamp(targets, min=-100.0, max=100.0)
```

**Fix**:
```python
targets = torch.clamp(targets, min=-10.0, max=10.0)  # Tighter bounds for stability
```

**Why**: Prevents Q-value divergence by bounding the Bellman targets.

---

### Fix 2: Add Per-Step Reward Clipping (CRITICAL)
**File**: `dqn_agent.py` in `store_transition()` method

**Add after line 211**:
```python
# Clip rewards to prevent scale issues
reward = np.clip(reward, -5.0, 5.0)
```

**Why**: Prevents extreme rewards from breaking value estimation.

---

### Fix 3: Normalize Rewards by Grid Size (RECOMMENDED)
**File**: `coverage_env.py` line 625-626

**Current**:
```python
coverage_gain = (next_state.coverage - state.coverage).sum()
coverage_r = coverage_gain * self.coverage_reward
```

**Fix**:
```python
coverage_gain = (next_state.coverage - state.coverage).sum()
# Normalize by grid size to keep rewards consistent across scales
grid_size = next_state.coverage.shape[0] * next_state.coverage.shape[1]
normalized_gain = coverage_gain / (grid_size / 400)  # 400 = 20x20 baseline
coverage_r = normalized_gain * self.coverage_reward
```

**Why**: Makes rewards scale-invariant for multi-grid training.

---

### Fix 4: Add Gradient Norm Clipping (CRITICAL)
**File**: `dqn_agent.py` after line 321 (after `scaler.unscale_`)

**Add**:
```python
# Hard gradient norm clipping
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
```

**Why**: Prevents gradient explosions during training.

---

### Fix 5: Reduce Learning Rate (HIGH PRIORITY)
**File**: `config.py` line 80

**Current**:
```python
learning_rate: float = 5e-6  # FCN-level conservative rate
```

**Fix**:
```python
learning_rate: float = 1e-6  # Ultra-conservative for stability
```

**Why**: Slower updates prevent value divergence when combined with other fixes.

---

### Fix 6: Slower Target Network Updates (HIGH PRIORITY)
**File**: `config.py` lines 88-89

**Current**:
```python
target_update_tau: float = 0.01
target_update_frequency: int = 1  # Update every N episodes
```

**Fix**:
```python
target_update_tau: float = 0.005  # Slower soft updates
target_update_frequency: int = 5  # Update every 5 episodes
```

**Why**: Provides more stable targets, reducing bootstrapping errors.

---

### Fix 7: Stronger Epsilon Floor (MEDIUM PRIORITY)
**File**: `config.py` line 84

**Current**:
```python
epsilon_end: float = 0.25
```

**Fix**:
```python
epsilon_end: float = 0.15  # More exploration to combat overfitting
```

**Why**: Maintains exploration longer to improve generalization.

---

## 📊 DIAGNOSTIC IMPROVEMENTS

### Add Reward Monitoring
**File**: `train.py` after episode metrics logging

**Add**:
```python
# Track reward statistics
if episode % 10 == 0:
    print(f"  Reward/step: {metrics['reward']/metrics['steps']:.3f}")
    print(f"  Coverage reward component: {metrics.get('reward_coverage', 0):.3f}")
```

### Monitor Q-Value Range
**File**: `dqn_agent.py` in the gradient explosion detection block

**Already present** at lines 352-354, but ensure you're watching these values!

---

## 🧪 TESTING PROTOCOL

### Phase 1: Stability Test (50 episodes)
1. Apply Fixes 1-6
2. Run training for 50 episodes
3. Check:
   - ✅ No gradient explosions
   - ✅ Q-values stay in [-5, 5] range
   - ✅ Episode rewards in [-50, 50] range
   - ✅ Loss < 1.0

### Phase 2: Convergence Test (200 episodes)
1. Continue training
2. Check:
   - ✅ Validation coverage improving
   - ✅ Training-validation gap < 20%
   - ✅ No early stopping due to plateaus

### Phase 3: Full Run (1500 episodes)
1. Complete curriculum
2. Target:
   - Empty maps: >95% coverage
   - Random obstacles: >70% coverage
   - Corridors/caves: >50% coverage

---

## 🚀 OPTIONAL IMPROVEMENTS (Try If Still Unstable)

### A. Disable Probabilistic Environment Temporarily
**File**: `config.py` line 39

```python
USE_PROBABILISTIC_ENV = False  # Revert to deterministic for debugging
```

Test if deterministic environment trains more stably.

### B. Reduce Network Capacity
**File**: `config.py` line 71

```python
hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])  # Half capacity
```

Smaller network = less overfitting potential.

### C. Increase Dropout
**File**: `config.py` line 73

```python
dropout: float = 0.2  # Stronger regularization
```

### D. Use Huber Loss Instead of Smooth L1
**File**: `dqn_agent.py` line 314

```python
loss = nn.functional.huber_loss(q_values, targets, delta=1.0)
```

More robust to outliers.

---

## 📈 EXPECTED OUTCOMES

After applying Fixes 1-6:

**Stability Metrics:**
- Gradient norms: < 2.0 (currently 3.2+)
- Q-value range: [-5, 5] (currently [-1.5, 13])
- Episode rewards: [-20, 50] (currently [-100, 400])
- No gradient explosions

**Performance Metrics:**
- Training coverage: 85-95% (not 99% overfit)
- Validation coverage: 60-80% (not 40%)
- Generalization gap: < 15% (currently 40%)

**Training Dynamics:**
- Completes all 5 curriculum phases
- Reaches episode 1000+ before early stopping
- Best model at episode 800-1200 (not 250)

---

## 🎯 SUCCESS CRITERIA

Training is successful when:
1. ✅ No gradient explosions for 500+ episodes
2. ✅ Validation coverage > 60% on all map types
3. ✅ Training-validation gap < 20%
4. ✅ Agent completes corridor and cave maps (Phase 3-5)
5. ✅ Consistent improvement until episode 1000+

---

## ⚠️ RED FLAGS (Stop Training If You See)

1. 🔴 Gradient explosions after episode 100
2. 🔴 Q-values > 10 or < -10
3. 🔴 Episode rewards > 100
4. 🔴 Loss > 2.0 consistently
5. 🔴 Validation coverage decreasing

If any occur → revert to last checkpoint and apply more aggressive fixes.
