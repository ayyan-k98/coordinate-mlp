# Visualization Guide

This guide shows how to use the visualization tools to debug and understand agent behavior.

---

## 🎯 Quick Start

### **1. Visualize Your Best Checkpoint**

```bash
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep350.pt \
    --create-report
```

This will:
- Test on all map types (empty, random, corridor, cave)
- Test on grid sizes 20×20 and 30×30
- Run 3 episodes per configuration
- Generate visualizations and a comparison report

**Output**:
```
visualizations/
├── 20x20/
│   ├── empty/
│   │   ├── episode_0/
│   │   │   ├── episode_0_summary.png
│   │   │   └── episode_0_heatmap.png
│   │   ├── episode_1/...
│   │   └── episode_2/...
│   ├── random/...
│   ├── corridor/...
│   └── cave/...
├── 30x30/...
├── comparison_grid.png        # 6-panel analysis
├── evaluation_report.txt      # Detailed statistics
└── all_results.json           # Raw data
```

---

## 📊 Understanding the Outputs

### **Summary Plot** (`episode_N_summary.png`)

Shows 5 key moments in the episode:

```
[Start] → [25%] → [50%] → [75%] → [End]
```

**What to look for**:
- **Good**: Smooth progression, coverage increasing each panel
- **Bad**: Coverage stagnant, agent stuck in one area

**Example interpretations**:
- **Empty map**: Should see steady blue coverage spreading from start
- **Corridor map**: Should see agent navigating hallways systematically
- **Cave map**: Should see exploration of interconnected rooms

---

### **Heatmap** (`episode_N_heatmap.png`)

Shows visit counts with color-coded frequency:
- **White**: Unvisited
- **Yellow**: Visited 1-5 times
- **Orange**: Visited 6-15 times
- **Red**: Visited 16-30 times
- **Dark Red**: Visited 30+ times (⚠️ **looping!**)

**Statistics sidebar shows**:
- Total steps
- Unique cells visited
- Max revisits (high = looping problem)
- Avg revisits (high = inefficient)
- Final coverage

**What to look for**:
- ✅ **Good**: Mostly yellow/orange (uniform exploration)
- ⚠️ **Warning**: Red spots (some revisits, acceptable)
- ❌ **Bad**: Dark red hotspots (severe looping)

---

### **Comparison Grid** (`comparison_grid.png`)

6-panel analysis showing:

1. **Coverage Heatmap** (top-left)
   - Matrix of coverage % by map type × grid size
   - Green = good (>80%), Red = poor (<40%)

2. **Steps Distribution** (top-right)
   - Scatter plot showing episode lengths
   - Higher = less efficient

3. **Coverage Efficiency** (middle-left)
   - Coverage vs Steps for each map type
   - Upper-left is ideal (high coverage, low steps)

4. **Revisit Statistics** (middle-right)
   - Average and max revisits per map type
   - Detects looping behavior

5. **Performance Summary** (bottom)
   - Bar chart with error bars
   - Color-coded: green (good), orange (ok), red (poor)

---

### **Evaluation Report** (`evaluation_report.txt`)

Detailed text analysis including:

**Overall Statistics**:
```
Average Coverage: 46.3% ± 34.0%
Average Steps: 562.0 ± 180.5
```

**Per-Map-Type Breakdown**:
```
CORRIDOR:
  Coverage: 16.8% ± 5.2%   ← Low! Needs investigation
  Steps: 350.0 ± 0.0       ← Maxing out steps
  Max Revisits: 45.2       ← Looping detected!
```

**Best Episodes**:
```
1. empty 20×20: 98.8% coverage in 297 steps
2. random 30×30: 58.7% coverage in 787 steps
...
```

**Worst Episodes**:
```
1. corridor 20×20: 10.8% coverage in 350 steps  ← Failure mode!
   Path: visualizations/20x20/corridor/episode_0/
...
```

**Failure Mode Analysis**:
```
⚠️  LOOPING DETECTED (8 episodes):
  - corridor 20×20: 52 max revisits

⚠️  LOW COVERAGE (12 episodes < 50%):
  - corridor 30×30: 16.3% coverage
```

**Recommendations**:
```
⚠️  Corridor coverage (16.8%) is low. Consider:
   - Increasing frontier bonus
   - Adding exploration bonus for new cells
   - Reducing revisit penalty initially
```

---

## 🔍 Debugging Workflow

### **Step 1: Run Comprehensive Evaluation**

```bash
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep350.pt \
    --num-episodes 5 \
    --create-report
```

### **Step 2: Read the Report**

```bash
cat visualizations/evaluation_report.txt
```

Look for:
- Which map types have low coverage?
- Are there looping issues? (high max revisits)
- Which episodes failed worst?

### **Step 3: Examine Failure Cases**

From the report, find worst episodes:
```
Worst Episodes:
1. corridor 20×20: 10.8% coverage
   Path: visualizations/20x20/corridor/episode_0/
```

Open the visualizations:
```bash
# On your local machine or Colab
from IPython.display import Image, display

# View summary
display(Image('visualizations/20x20/corridor/episode_0/episode_0_summary.png'))

# View heatmap
display(Image('visualizations/20x20/corridor/episode_0/episode_0_heatmap.png'))
```

### **Step 4: Identify Failure Pattern**

**Example: Corridor Failure**

Looking at `episode_0_heatmap.png`:
- **Dark red spot** in bottom-left corner (100+ revisits)
- Agent stuck in a loop
- **White regions** in top-right (never visited)
- Only 10.8% coverage

Looking at `episode_0_summary.png`:
- Start panel: Agent enters corridor
- 25% panel: Still in same corridor section
- 50% panel: **Stuck in loop** (same position as 25%)
- 75% panel: Still looping
- End panel: Never escaped the loop

**Diagnosis**: Agent gets trapped in local loops, doesn't explore frontiers.

**Solution**: Increase frontier bonus, add count-based exploration.

---

## 🛠️ Advanced Usage

### **Compare Multiple Checkpoints**

```bash
# Visualize episode 200
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep200.pt \
    --save-dir visualizations/ep200 \
    --create-report

# Visualize episode 350
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep350.pt \
    --save-dir visualizations/ep350 \
    --create-report

# Compare reports
diff visualizations/ep200/evaluation_report.txt \
     visualizations/ep350/evaluation_report.txt
```

### **Test Specific Scenarios**

```bash
# Only test corridors on large grids
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep350.pt \
    --map-types corridor \
    --grid-sizes 30 40 \
    --num-episodes 10
```

### **Quick Test (No Report)**

```bash
# Fast check on one map type
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep350.pt \
    --map-types empty \
    --grid-sizes 20 \
    --num-episodes 1
```

---

## 📈 Interpreting Results

### **Good Performance**

**Empty Maps**: Should achieve 95%+ coverage
```
EMPTY:
  Coverage: 98.3% ± 1.2%  ✅
  Steps: 250.5 ± 45.2
  Max Revisits: 8.3       ✅ (low looping)
```

**Heatmap**: Mostly yellow/orange, no dark red spots
**Summary**: Smooth coverage progression across all panels

---

### **Moderate Performance**

**Random Obstacles**: Should achieve 60-80% coverage
```
RANDOM:
  Coverage: 58.7% ± 12.3%  ⚠️ (could be better)
  Steps: 650.2 ± 120.4
  Max Revisits: 15.8       ⚠️ (some looping)
```

**Heatmap**: Yellow/orange with a few red spots
**Summary**: Coverage progresses but with some backtracking

---

### **Poor Performance** ❌

**Corridors**: Currently achieving <20% coverage
```
CORRIDOR:
  Coverage: 16.8% ± 5.2%   ❌ FAILURE
  Steps: 350.0 ± 0.0       ❌ (timeout)
  Max Revisits: 52.3       ❌ (severe looping)
```

**Heatmap**: Dark red hotspots, large white regions
**Summary**: Agent stuck in loops, never explores full map

**Action Required**: See recommendations in report!

---

## 🎯 Example Analysis Session

### **Scenario**: Your ep350 checkpoint

```bash
# Run evaluation
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_baseline_ep350.pt \
    --create-report
```

**Report shows**:
```
CORRIDOR:
  Coverage: 16.8%  ❌
  Max Revisits: 52 ❌ (looping!)

Recommendations:
  ⚠️  Increase frontier bonus
  ⚠️  Add count-based exploration
```

**Open worst corridor episode**:
- Heatmap shows: Dark red loop in corner
- Summary shows: Agent stuck from 25% panel onwards

**Conclusion**: Need to fix exploration strategy

**Apply fixes from TRAINING_FIXES.md**:
1. Increase frontier_bonus: 0.2 → 1.0
2. Add count-based exploration bonus
3. Re-train for 200 episodes

**Re-evaluate**:
```bash
python visualize_checkpoint.py \
    --checkpoint checkpoints/fcn_fixed_ep550.pt \
    --map-types corridor \
    --create-report
```

**New results**:
```
CORRIDOR:
  Coverage: 45.2%  ✅ (improved from 16.8%!)
  Max Revisits: 18 ✅ (reduced from 52)
```

**Success!** Continue iterating.

---

## 🚀 Integration with Training

You can also add visualization to your training loop:

```python
# In train.py, after validation:

from visualize_checkpoint import visualize_single_episode

if episode % 100 == 0:
    print(f"\n🎨 Creating visualizations for episode {episode}...")

    for map_type in ['empty', 'corridor']:
        save_dir = f"visualizations/training/ep{episode}_{map_type}"
        metrics = visualize_single_episode(
            agent=agent,
            grid_size=20,
            map_type=map_type,
            save_dir=save_dir,
            episode_idx=0,
            verbose=True
        )

        print(f"  {map_type}: {metrics['coverage']:.1%} coverage")
```

This creates visualizations during training to track improvement!

---

## 📝 Tips & Best Practices

### **1. Start with Quick Checks**

```bash
# Quick test on one map type
python visualize_checkpoint.py \
    --checkpoint checkpoints/best.pt \
    --map-types corridor \
    --grid-sizes 20 \
    --num-episodes 3
```

### **2. Focus on Failures**

Only create full reports when you need detailed analysis:
```bash
# Full analysis with report
python visualize_checkpoint.py \
    --checkpoint checkpoints/best.pt \
    --num-episodes 10 \
    --create-report
```

### **3. Compare Before/After**

Always visualize before and after applying fixes:
```bash
# Before
python visualize_checkpoint.py --checkpoint checkpoints/before.pt --save-dir viz_before
# After
python visualize_checkpoint.py --checkpoint checkpoints/after.pt --save-dir viz_after
# Compare
diff viz_before/evaluation_report.txt viz_after/evaluation_report.txt
```

### **4. Archive Good Examples**

When you get good performance, save the visualizations:
```bash
cp -r visualizations/20x20/corridor/episode_0 examples/good_corridor_example/
```

Use these as baselines for comparison.

---

## 🎬 Next Steps

1. **Run your first visualization**:
   ```bash
   python visualize_checkpoint.py --checkpoint checkpoints/fcn_baseline_ep350.pt --create-report
   ```

2. **Read the report** and identify failure modes

3. **Apply fixes** from `TRAINING_FIXES.md`

4. **Re-train** for 100-200 episodes

5. **Visualize again** and compare improvement

6. **Iterate** until all map types achieve good coverage!

---

## 📚 See Also

- `TRAINING_FIXES.md` - Critical fixes for stability
- `BRAINSTORM_NEXT_STEPS.md` - Long-term improvements
- `visualization.py` - PathVisualizer class documentation
- `visualize_checkpoint.py` - Full script reference

---

**Happy debugging! 🎨**
