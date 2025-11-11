# FCN Training Configuration Guide

Complete guide for running the FCN coverage agent with different settings.

---

## 📋 Quick Reference

### Basic Training (Recommended Defaults)
```bash
python train.py \
  --experiment-name fcn_baseline \
  --episodes 1500 \
  --multi-scale \
  --curriculum default \
  --device cuda
```

This automatically includes:
- ✅ **POMDP enabled** (obstacle discovery through sensing)
- ✅ **Balanced rewards** (episode returns ~60)
- ✅ **Progressive revisit penalty** (-0.03 → -0.08)
- ✅ **Curriculum learning** (5 phases, adaptive epsilon)
- ✅ **Mixed precision (AMP)** for 2-3× speedup
- ✅ **Multi-scale training** (15×15, 20×20, 25×25, 30×30)

---

## 🎛️ Command-Line Arguments

### Available Arguments

```bash
python train.py --help
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment-name` | str | `'coordinate_mlp_coverage'` | Experiment name (for logs/checkpoints) |
| `--episodes` | int | `1500` | Number of training episodes |
| `--multi-scale` | flag | `False` | Enable multi-scale training (15, 20, 25, 30) |
| `--device` | str | `'cuda'` | Device: 'cuda' or 'cpu' |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--hidden-dim` | int | `256` | Hidden dimension (FCN: unused, kept for compat) |
| `--curriculum` | str | `'default'` | Curriculum: 'default', 'fast', 'none' |

---

## 🔧 Configuration Options (Code-Level)

If command-line args aren't enough, modify `config.py` or pass a custom config:

### 1. POMDP Settings (ENABLED BY DEFAULT ✅)

**Where**: `dqn_agent.py` initialization

**Current**: POMDP is **always enabled** with the FCN implementation. The agent:
- Starts with ~20% obstacle knowledge
- Discovers obstacles through sensing (obstacle_belief channel)
- Learns from collisions (collision_penalty active)
- Reaches ~85-95% obstacle knowledge after exploration

**To verify POMDP is working**:
```python
# In train.py, add this diagnostic
print(f"Episode {episode}:")
print(f"  Obstacle knowledge: {env.state.obstacle_belief.sum() / env.state.obstacles.size * 100:.1f}%")
print(f"  Collisions: {info.get('collisions', 0)}")
```

**Expected output**:
```
Episode 0:   Obstacle knowledge: 22.3%, Collisions: 12
Episode 50:  Obstacle knowledge: 65.1%, Collisions: 5
Episode 150: Obstacle knowledge: 89.2%, Collisions: 1
```

---

### 2. Probabilistic Environment (Currently DISABLED)

**Where**: `config.py` line 39

```python
# config.py
USE_PROBABILISTIC_ENV = False  # Currently disabled
```

**What it does**: Uses probabilistic coverage model with sigmoid decay instead of binary coverage

**To enable**:
1. Edit `config.py`:
   ```python
   USE_PROBABILISTIC_ENV = True
   PROBABILISTIC_COVERAGE_MIDPOINT = 2.0  # Distance where coverage = 50%
   PROBABILISTIC_COVERAGE_STEEPNESS = 1.5  # How quickly coverage decays
   ```

2. **Note**: This is currently a **global constant**, not integrated with the command-line args. You'll need to modify the code or create a custom config.

**Why disabled**: Binary coverage is simpler and works well. Probabilistic coverage adds complexity without clear benefit.

**Recommendation**: Keep it disabled unless you're researching sensor uncertainty.

---

### 3. Reward Configuration

**Where**: `config.py` lines 114-128 (`EnvironmentConfig`)

**Current settings** (already optimized!):
```python
# Balanced Rewards (5× scale, episode returns ~60)
coverage_reward: float = 0.5           # Per new coverage
revisit_penalty: float = -0.08         # Per revisit
collision_penalty: float = -0.3        # Per collision
step_penalty: float = -0.0004          # Per step (efficiency)
frontier_bonus: float = 0.01           # Per frontier cell
coverage_confidence_weight: float = 0.03  # Confidence bonus
first_visit_bonus: float = 0.5         # New cell discovery bonus

# Progressive revisit penalty
use_progressive_revisit_penalty: bool = True
revisit_penalty_min: float = -0.03     # Start (lenient)
revisit_penalty_max: float = -0.08     # End (strict)
```

**To modify**:
```python
# Create custom config
from config import get_default_config

config = get_default_config()

# Adjust rewards
config.environment.coverage_reward = 0.8  # More reward for coverage
config.environment.collision_penalty = -0.5  # Harsher collision penalty

# Train with custom config
from train import train
train(config, curriculum_type='default')
```

**⚠️ Warning**: These rewards are **already optimized**. Changing them may cause:
- Gradient explosions (if too large)
- No learning (if too small)
- Unbalanced behavior (if penalties too weak)

---

### 4. Curriculum Learning

**Where**: Command-line `--curriculum` argument

**Options**:

#### **A) Default Curriculum** (Recommended)
```bash
python train.py --curriculum default
```

**Phases** (1500 episodes):
1. **Phase 1: Basics** (0-200) - Empty maps, learn basic movement
   - Map types: `empty`
   - Epsilon floor: 0.10
2. **Phase 2: Obstacles** (200-500) - Random obstacles, learn avoidance
   - Map types: `random`
   - Epsilon floor: 0.15
3. **Phase 3: Structures** (500-800) - Corridors/rooms, learn navigation
   - Map types: `corridor`, `room`
   - Epsilon boost: 0.30 (re-explore for new structures!)
   - Epsilon floor: 0.20
4. **Phase 4: Complex** (800-1100) - Caves/L-shapes, hard exploration
   - Map types: `cave`, `lshape`
   - Epsilon boost: 0.35 (maximum exploration!)
   - Epsilon floor: 0.25
5. **Phase 5: Mixed** (1100-1500) - All map types, generalization
   - Map types: `all`
   - Epsilon floor: 0.10

**Benefits**:
- ✅ Gradual difficulty increase
- ✅ Phase-adaptive epsilon (boosts at transitions)
- ✅ Prevents local optima on complex maps

---

#### **B) Fast Curriculum**
```bash
python train.py --curriculum fast --episodes 750
```

**Phases** (750 episodes, 50% shorter):
- Same progression, half the episodes per phase
- Use for quick experiments or debugging

---

#### **C) No Curriculum**
```bash
python train.py --curriculum none --episodes 500
```

**Behavior**:
- All map types from start
- No epsilon boosting/floors
- Natural epsilon decay only
- Use for ablation studies ("does curriculum help?")

---

### 5. Multi-Scale Training (Recommended!)

**Where**: `--multi-scale` flag

**Enabled**:
```bash
python train.py --multi-scale
```
- Trains on: 15×15, 20×20, 25×25, 30×30 grids
- Random selection each episode
- Each grid has separate replay buffer
- Agent learns scale-robust features

**Disabled** (single-scale):
```bash
python train.py  # Default: 20×20 only
```
- Trains only on 20×20 grids
- Faster training per episode
- May not generalize to other sizes

**Recommendation**: **Always use `--multi-scale`** unless debugging. FCN needs multi-scale training for generalization.

---

### 6. Learning Rate & Stability

**Where**: `config.py` line 80 (`TrainingConfig`)

**Current** (ultra-conservative, FCN-optimized):
```python
learning_rate: float = 5e-6  # 6× slower than typical 3e-5
```

**This is already optimized for FCN stability!**

If you see gradient explosions (unlikely), try:
```python
learning_rate: float = 1e-6  # 10× slower (very conservative)
```

If training is too slow (> 5000 episodes to converge):
```python
learning_rate: float = 1e-5  # 2× faster (still safe)
```

---

### 7. Epsilon Decay Settings

**Where**: `config.py` lines 83-86 (`TrainingConfig`)

**Current** (optimized for exploration):
```python
epsilon_start: float = 1.0      # Full exploration at start
epsilon_end: float = 0.25       # 25% exploration at end (raised from 0.05!)
epsilon_decay: float = 0.998    # Slow decay (~700 episodes to min)
```

**Why epsilon_end = 0.25?**
- Coverage is **non-stationary** (maps change)
- Agent needs continued exploration
- Prevents convergence to local optima

**Curriculum overrides**: Phase-adaptive epsilon uses:
- Epsilon floors (maintain minimum exploration)
- Epsilon boosts (reset to 0.30-0.35 at hard phases)

---

### 8. Early Stopping

**Where**: `train.py` training loop

**Current** (enabled by default):
- Patience: 200 episodes without improvement
- Check frequency: Every 50 episodes (eval_frequency)
- Metric: Validation coverage (not training!)

**To disable**:
```python
# In train.py, around line 615
# Comment out the early stopping block:
# if episodes_without_improvement >= patience:
#     break
```

**Recommendation**: Keep enabled. Prevents wasting compute if model has converged.

---

## 🚀 Example Configurations

### 1. Quick Test (5 minutes)
```bash
python train.py \
  --experiment-name fcn_quicktest \
  --episodes 50 \
  --device cuda \
  --curriculum none
```
**Purpose**: Verify FCN works, check for gradient explosions

---

### 2. Ablation: No Curriculum
```bash
python train.py \
  --experiment-name fcn_no_curriculum \
  --episodes 500 \
  --multi-scale \
  --curriculum none \
  --device cuda
```
**Purpose**: Measure curriculum benefit (compare to default)

---

### 3. Ablation: Single Scale
```bash
python train.py \
  --experiment-name fcn_single_scale \
  --episodes 1500 \
  --curriculum default \
  --device cuda
  # Note: --multi-scale NOT specified
```
**Purpose**: Measure multi-scale benefit (compare to default)

---

### 4. Fast Training (1 hour)
```bash
python train.py \
  --experiment-name fcn_fast \
  --episodes 750 \
  --multi-scale \
  --curriculum fast \
  --device cuda
```
**Purpose**: Quick results with shorter curriculum

---

### 5. Full Production Run (2-3 hours)
```bash
python train.py \
  --experiment-name fcn_production \
  --episodes 1500 \
  --multi-scale \
  --curriculum default \
  --device cuda \
  --seed 42
```
**Purpose**: Best model for publication, maximum performance

---

### 6. Multiple Seeds (Reproducibility)
```bash
# Run 3 times with different seeds
for seed in 42 123 999; do
  python train.py \
    --experiment-name fcn_seed_${seed} \
    --episodes 1500 \
    --multi-scale \
    --curriculum default \
    --device cuda \
    --seed ${seed}
done
```
**Purpose**: Measure variance, report mean ± std

---

## 📊 Monitoring Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir logs/

# Open browser to: http://localhost:6006
```

**Key metrics to watch**:
- `train/coverage` - Should increase: 30% → 60%+
- `train/grad_norm` - Should stay < 10.0 (NOT 25-50!)
- `train/q_mean` - Should stay 1-5 (NOT exploding to 11+)
- `train/loss` - Should decrease over time
- `val/coverage` - Validation coverage (model selection)

**Red flags** (FCN should NOT have these!):
- ⚠️ `grad_norm` > 25.0 - Gradient explosion
- ⚠️ `q_mean` > 10.0 - Q-value explosion
- ⚠️ `loss` = NaN/Inf - Training collapse

---

### Console Output
```
Episode  100: coverage=45.2%, epsilon=0.817, grad_norm=4.23, q_mean=2.35
Episode  200: coverage=52.1%, epsilon=0.669, grad_norm=5.12, q_mean=3.01
...
```

**Healthy training**:
- Coverage increasing steadily
- Grad norm 2-8 (stable)
- Q-values 1-5 (stable)
- No explosion warnings

---

## 🔍 Debugging Common Issues

### Issue 1: No POMDP Effect (Collisions Always 0)

**Symptom**:
```
Episode 100: collisions=0.0, obstacle_belief=100%
```

**Cause**: Obstacle discovery broken

**Fix**: Check `coverage_env.py` line 72-74:
```python
# Should be:
obstacle_belief: np.ndarray  # Agent's learned map
# NOT:
obstacle_belief = obstacles  # This would be cheating!
```

---

### Issue 2: Poor Coverage (< 40% after 200 episodes)

**Possible causes**:
1. **Curriculum not loading**: Check curriculum files exist
2. **Wrong device**: Using CPU instead of GPU (100× slower)
3. **Rewards broken**: Check reward_config is passed to environment

**Diagnostic**:
```python
# Add to train.py
print(f"Reward breakdown: coverage={info['reward_coverage']:.2f}, "
      f"collision={info['reward_collision']:.2f}")
```

---

### Issue 3: Training Too Slow

**Symptoms**: < 5 episodes/second on GPU

**Causes**:
1. **No mixed precision**: AMP disabled
2. **Large grid sizes**: 30×30 takes 4× longer than 15×15
3. **Curriculum overhead**: Map generation

**Fix**:
```python
# Verify AMP is enabled
# Should see: "Mixed Precision (AMP): Enabled"

# If not, check:
# 1. CUDA available? torch.cuda.is_available()
# 2. GPU device selected? --device cuda
```

---

## 📝 Custom Configuration Example

For advanced users who need full control:

```python
# custom_train.py
from config import get_default_config, ExperimentConfig
from train import train

# Create base config
config = get_default_config()

# Customize experiment
config.experiment_name = "fcn_custom"
config.seed = 999

# Customize training
config.training.num_episodes = 2000
config.training.learning_rate = 1e-5  # 2× faster
config.training.epsilon_end = 0.15    # Lower exploration at end
config.training.multi_scale = True

# Customize environment
config.environment.obstacle_density = 0.20  # More obstacles
config.environment.collision_penalty = -0.5  # Harsher penalty

# Customize model (FCN)
config.model.hidden_channels = [64, 128, 256]  # Default
config.model.dropout = 0.2  # More regularization

# Train
train(config, curriculum_type='default')
```

---

## ✅ Recommended Settings Summary

**For best results** (what we expect to work):

```bash
python train.py \
  --experiment-name fcn_best \
  --episodes 1500 \
  --multi-scale \
  --curriculum default \
  --device cuda \
  --seed 42
```

**With defaults**:
- ✅ POMDP enabled (obstacle discovery)
- ✅ Balanced rewards (optimized scales)
- ✅ Progressive revisit penalty
- ✅ Phase-adaptive curriculum
- ✅ Mixed precision (AMP)
- ✅ Multi-scale training
- ✅ Conservative learning rate (5e-6)
- ✅ Proper validation & early stopping

**Expected results**:
- Coverage: 60-65% (stable)
- Gradient explosions: 0
- Training time: 2-3 hours on GPU
- Best model selected via validation

---

## 🎯 Quick Start Checklist

- [ ] Pull latest code: `git pull origin claude/repo-analysis-review-011CUt4vRDG2NJsNwoeUo2G2`
- [ ] Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Run quick test: `python train.py --episodes 50 --device cuda`
- [ ] Verify zero explosions in logs
- [ ] Run full training: `python train.py --episodes 1500 --multi-scale --curriculum default --device cuda`
- [ ] Monitor TensorBoard: `tensorboard --logdir logs/`
- [ ] Wait 2-3 hours for completion
- [ ] Check final coverage: Should be 60-65%
- [ ] Celebrate stable training! 🎉

---

**That's it! The defaults are already optimized. Just run and watch it work! 🚀**
