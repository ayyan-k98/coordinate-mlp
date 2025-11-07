# Curriculum Learning Implementation

## Overview

This implementation adds **curriculum learning** to progressively train the agent on increasingly difficult map structures, using the previously orphaned `map_generator.py`.

---

## 🎯 **Why Curriculum Learning?**

**Problem:** Training on complex maps from the start can be overwhelming
- Agent struggles to learn basic coverage
- Gets stuck in local optima
- Poor generalization to new scenarios

**Solution:** Progressive difficulty curriculum
- Start simple (empty maps)
- Gradually increase complexity (obstacles → structures)
- End with mixed scenarios (generalization)

**Result:** Faster learning, better final performance, robust policies

---

## 📚 **Curriculum Phases**

### Phase 1: Basics (200 episodes)
**Map Types:** `empty`
**Goal:** Learn fundamental coverage strategies
- No obstacles to avoid
- Focus on efficient path planning
- Establish baseline behaviors

### Phase 2: Obstacles (300 episodes)
**Map Types:** `random`
**Goal:** Learn obstacle avoidance
- Random scattered obstacles
- Collision avoidance
- Frontier detection

### Phase 3: Structures (400 episodes)
**Map Types:** `corridor`, `room`
**Goal:** Learn structured navigation
- Hallways and doorways
- Room exploration
- Systematic coverage patterns

### Phase 4: Complex (400 episodes)
**Map Types:** `cave`, `lshape`
**Goal:** Master complex irregular structures
- Organic cave-like obstacles
- L-shaped barriers
- Advanced path planning

### Phase 5: Mixed (200 episodes)
**Map Types:** All 6 types
**Goal:** Generalize across all scenarios
- Random sampling from all map types
- Test robustness
- Final policy refinement

**Total: 1500 episodes**

---

## 🚀 **Usage**

### Training with Curriculum

```bash
# Full curriculum (default)
python train.py --curriculum default --episodes 1500

# Fast curriculum (for testing)
python train.py --curriculum fast --episodes 750

# No curriculum (baseline)
python train.py --curriculum none

# With multi-scale
python train.py --curriculum default --multi-scale
```

### In Code

```python
from curriculum import CurriculumScheduler, create_default_curriculum
from coverage_env import CoverageEnvironment

# Create curriculum
curriculum_config = create_default_curriculum()
curriculum = CurriculumScheduler(curriculum_config, grid_sizes=[15, 20, 25, 30])

# Training loop
for episode in range(num_episodes):
    # Sample from curriculum
    grid_size = curriculum.sample_grid_size()    # Multi-scale
    map_type = curriculum.sample_map_type()      # Curriculum

    # Create environment
    env = CoverageEnvironment(
        grid_size=grid_size,
        map_type=map_type  # Use curriculum map type
    )

    # Train
    state = env.reset()
    # ... training ...

    # Advance curriculum
    phase_changed = curriculum.step()
    if phase_changed:
        print("Phase transition!")
        curriculum.print_status()
```

---

## 📊 **Map Types**

### 1. Empty
```
┌────────────┐
│            │
│            │
│            │
│            │
└────────────┘
```
**Characteristics:**
- No obstacles
- Pure coverage problem
- Tests basic path planning

### 2. Random
```
┌────────────┐
│ ■  ■       │
│    ■ ■     │
│ ■     ■    │
│   ■   ■    │
└────────────┘
```
**Characteristics:**
- Clustered random obstacles
- ~15% density
- Tests obstacle avoidance

### 3. Corridor
```
┌────────────┐
│████   ████ │
│        ■   │
│████   ████ │
│        ■   │
└────────────┘
```
**Characteristics:**
- Hallway structures
- Narrow passages
- Tests navigation skills

### 4. Room
```
┌────────────┐
│████ ████  │
│█   █    █ │
│█   █████ █│
│█         █│
└────────────┘
```
**Characteristics:**
- Rooms with walls/doors
- Compartmentalized
- Tests systematic exploration

### 5. Cave
```
┌────────────┐
│  ■■■       │
│ ■■  ■■     │
│■  ■   ■■   │
│ ■■■  ■     │
└────────────┘
```
**Characteristics:**
- Irregular organic shapes
- Complex boundaries
- Tests adaptability

### 6. L-Shape
```
┌────────────┐
│████████   │
│        ██  │
│        ██  │
│        ██  │
└────────────┘
```
**Characteristics:**
- Large structural obstacles
- L-shaped barriers
- Tests around-object planning

---

## 🔧 **API Reference**

### CurriculumScheduler

```python
class CurriculumScheduler:
    def __init__(self, config: CurriculumConfig, grid_sizes: List[int])

    def sample_map_type(self) -> str
        """Sample map type from current phase."""

    def sample_grid_size(self) -> int
        """Sample grid size for multi-scale training."""

    def step(self) -> bool
        """Advance one episode. Returns True if phase changed."""

    def get_progress(self) -> dict
        """Get curriculum progress information."""

    def print_status(self)
        """Print current curriculum status."""
```

### CoverageEnvironment (Enhanced)

```python
class CoverageEnvironment:
    def __init__(
        self,
        grid_size: int = 20,
        map_type: str = "random",  # NEW!
        ...
    )

    def reset(self, map_type: Optional[str] = None)
        """Can override map type per episode."""
```

---

## 📈 **Expected Benefits**

### Training Efficiency
- **Faster convergence:** Simple tasks first → stable learning
- **Fewer episodes needed:** Progressive difficulty reduces thrashing
- **Better sample efficiency:** Agent learns incrementally

### Final Performance
- **Higher coverage:** Better policies from structured learning
- **Fewer collisions:** Learned avoidance progressively
- **More robust:** Trained on diverse scenarios

### Generalization
- **Scale invariance:** Multi-scale mixed with curriculum
- **Structure invariance:** Learns across map types
- **Transfer:** Skills from simple maps apply to complex

---

## 🧪 **Ablation Studies**

Test the value of curriculum learning:

```bash
# With curriculum
python train.py --curriculum default --experiment-name "with_curriculum"

# Without curriculum (baseline)
python train.py --curriculum none --experiment-name "no_curriculum"

# Fast curriculum
python train.py --curriculum fast --experiment-name "fast_curriculum"
```

**Compare:**
- Final coverage percentage
- Training stability (reward variance)
- Generalization to unseen map types
- Sample efficiency (episodes to reach goal)

---

## 🎨 **Customizing Curriculum**

### Create Custom Phases

```python
from curriculum import CurriculumPhase, CurriculumConfig

custom_curriculum = CurriculumConfig(
    enabled=True,
    phases=[
        CurriculumPhase(
            name="Easy Start",
            map_types=["empty"],
            num_episodes=100,
            description="Warm up phase"
        ),
        CurriculumPhase(
            name="Main Training",
            map_types=["random", "room", "corridor"],
            num_episodes=800,
            description="Core training"
        ),
        CurriculumPhase(
            name="Hard Finish",
            map_types=["cave", "lshape"],
            num_episodes=100,
            description="Final challenge"
        ),
    ]
)
```

### Adjust Phase Duration

```python
# Shorter curriculum for debugging
debug_curriculum = create_fast_curriculum()  # 750 episodes total

# No curriculum for baseline
baseline = create_no_curriculum()  # All maps from start
```

---

## 📊 **Monitoring Progress**

### Training Output

```
======================================================================
Curriculum Status
======================================================================
Phase: Phase 2: Obstacles (2/5)
Description: Learn obstacle avoidance with random obstacles
Map Types: random
Progress: 150/300 (50.0%)
Overall: 350/1500 (23.3%)
======================================================================

Episode 350: Phase=Phase 2: Obstacles, Map=random, Grid=20x20, Coverage=68.5%
```

### TensorBoard Logging

```python
# Curriculum info logged to TensorBoard
tb_logger.log_scalar('curriculum/phase', phase_idx, episode)
tb_logger.log_text('curriculum/map_type', map_type, episode)
```

---

## ✅ **Benefits Over Previous Implementation**

### Before
- ❌ `map_generator.py` orphaned (unused 204 lines)
- ❌ Only trained on simple random noise
- ❌ No progressive difficulty
- ❌ Limited scenario diversity

### After
- ✅ `map_generator.py` actively used
- ✅ 6 diverse map types (empty → caves)
- ✅ Progressive curriculum (5 phases)
- ✅ Better learning, better generalization

---

## 🔬 **Research Applications**

This curriculum system enables research on:

1. **Progressive Training:** Effect of curriculum on RL learning
2. **Transfer Learning:** Skills from simple → complex maps
3. **Multi-Task Learning:** Single policy for multiple scenarios
4. **Generalization:** Scale + structure invariance
5. **Sample Efficiency:** Fewer episodes to reach performance

---

## 📚 **References**

**Curriculum Learning:**
- Bengio et al. "Curriculum Learning" (ICML 2009)
- Graves et al. "Automated Curriculum Learning" (ICLR 2017)

**Map Generation:**
- PCG (Procedural Content Generation) for training diversity
- Structured environments for realistic scenarios

---

## 🚧 **Future Enhancements**

Potential improvements:

1. **Adaptive Curriculum:** Adjust phase duration based on performance
2. **Difficulty Metrics:** Automatically assess map difficulty
3. **Reverse Curriculum:** Train hard → easy for robustness
4. **Multi-Agent Curriculum:** Coordinate multiple agents
5. **Dynamic Obstacles:** Moving obstacles in later phases

---

## 💡 **Quick Start**

```bash
# 1. Train with default curriculum
python train.py --curriculum default --multi-scale --episodes 1500

# 2. Monitor in TensorBoard
tensorboard --logdir logs/

# 3. Evaluate on all map types
python test.py --checkpoint checkpoints/best.pt

# 4. Compare curriculum vs no curriculum
python train.py --curriculum none --experiment-name baseline
```

---

**The curriculum learning system makes training more efficient, policies more robust, and utilizes the sophisticated map generation capabilities that were previously unused.**
