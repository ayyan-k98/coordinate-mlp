# Quick Start Guide

## Installation

```powershell
# Navigate to project directory
cd "d:\pro\marl\coordinate mlp"

# Install dependencies
pip install torch numpy matplotlib tensorboard

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Running Tests

```powershell
# Run all unit tests
python run_tests.py

# Or run individual tests
python -m src.models.positional_encoding
python -m src.models.coordinate_network
python -m src.agent.dqn_agent
```

## Training Examples

### 1. Single-Scale Training (Baseline)

Train on 20×20 grid only:

```powershell
python train.py `
    --experiment-name single_scale_baseline `
    --episodes 1500 `
    --device cuda `
    --seed 42
```

Expected results:
- Training time: ~2-3 hours on GPU
- Final coverage: 35-45% on 20×20
- Checkpoint saved to: `checkpoints/single_scale_baseline_best.pt`

### 2. Multi-Scale Training (Recommended)

Train on multiple grid sizes [15, 20, 25, 30]:

```powershell
python train.py `
    --experiment-name multi_scale_curriculum `
    --episodes 2000 `
    --multi-scale `
    --device cuda `
    --seed 42
```

Expected results:
- Training time: ~3-4 hours on GPU
- Final coverage: 38-48% on 20×20
- Better generalization to unseen sizes

### 3. Custom Configuration

```powershell
python train.py `
    --experiment-name custom_large `
    --episodes 2500 `
    --multi-scale `
    --hidden-dim 512 `
    --device cuda
```

## Testing

### Test on Multiple Grid Sizes

```powershell
# Test trained model
python test.py `
    --checkpoint checkpoints/multi_scale_curriculum_best.pt `
    --test-sizes 20 25 30 35 40 50 `
    --num-episodes 20 `
    --save-plots `
    --device cuda
```

Output:
- Console: Detailed metrics per grid size
- File: `results/coordinate_mlp_test_results.json`
- Plot: `results/coordinate_mlp_test_comparison.png`

### Expected Test Results

```
Grid-Size Invariance Analysis
==================================================
Baseline (20×20): 42.5%

Size    Coverage    Degradation       Status
--------------------------------------------------
20×20   42.5%±3.2%      +0.0%       BASELINE
25×25   40.1%±3.5%      -5.6%       ✓ GOOD
30×30   37.8%±4.1%     -11.1%       ✓ GOOD
35×35   35.2%±4.5%     -17.2%       ⚠ OK
40×40   32.9%±5.0%     -22.6%       ⚠ OK
```

## Monitoring Training

### TensorBoard

```powershell
# Start TensorBoard
tensorboard --logdir=logs

# Open browser to: http://localhost:6006
```

Metrics to watch:
- `train/reward`: Should increase over time
- `train/coverage`: Should plateau at 35-45%
- `train/loss`: Should decrease and stabilize
- `train/epsilon`: Should decay to ~0.05

### Console Output

Training prints metrics every 10 episodes:

```
Episode   0: reward=12.3456, steps=245, coverage=0.1523, epsilon=1.0000, memory_size=245
Episode  10: reward=25.7812, steps=312, coverage=0.2341, epsilon=0.9512, memory_size=3450
Episode  20: reward=38.2145, steps=289, coverage=0.3102, epsilon=0.9048, memory_size=6234
...
Episode 1490: reward=56.3421, steps=267, coverage=0.4287, epsilon=0.0500, memory_size=50000
```

## Understanding Results

### Good Performance

✓ Coverage on 20×20: **> 38%**  
✓ Degradation at 40×40: **< 25%**  
✓ Training converges: **by episode 1200**

### Needs Tuning

⚠ Coverage on 20×20: **< 35%**  
→ Try: More episodes, larger hidden_dim, adjust rewards

⚠ Degradation at 40×40: **> 30%**  
→ Try: More multi-scale training, more diverse grid sizes

⚠ Training unstable: **loss spikes, coverage fluctuates**  
→ Try: Lower learning rate, gradient clipping, smaller batch

## Troubleshooting

### Issue: CUDA out of memory

Solution:
```powershell
# Use CPU instead
python train.py --device cpu

# Or reduce batch size (edit src/config.py)
# training.batch_size = 16  # instead of 32
```

### Issue: Training very slow

Possible causes:
- Mock environment is synchronous
- Attention over many cells (40×40 = 1600 cells)
- CPU training

Solutions:
- Use smaller grid sizes initially
- Reduce hidden_dim (256 → 128)
- Use GPU if available

### Issue: Poor performance

Debug checklist:
1. Check epsilon decay (should reach ~0.05)
2. Verify replay buffer is filling (should reach 50k)
3. Check target network updates (every episode)
4. Monitor loss convergence
5. Validate reward function

## Next Steps

### 1. Integrate Real Environment

Replace mock environment in `train.py`:

```python
# Replace this:
env = create_mock_environment(grid_size, config)

# With your actual environment:
from your_package import CoverageEnvironment
env = CoverageEnvironment(
    grid_size=grid_size,
    sensor_range=config.environment.get_sensor_range(grid_size),
    max_steps=config.environment.get_max_steps(grid_size)
)
```

### 2. Hyperparameter Tuning

Key parameters to tune (in `src/config.py`):
- `model.hidden_dim`: 128, 256, 512
- `model.num_freq_bands`: 4, 6, 8
- `model.num_attention_heads`: 4, 8
- `training.learning_rate`: 1e-4, 5e-5, 1e-5
- `training.gamma`: 0.95, 0.99, 0.995

### 3. Ablation Studies

Compare architectures:
```powershell
# Baseline
python train.py --experiment-name baseline --episodes 1500

# No Fourier features
# Edit config to set num_freq_bands=0
python train.py --experiment-name no_fourier --episodes 1500

# Different architectures
python train.py --experiment-name large_model --hidden-dim 512 --episodes 2000
```

### 4. Visualization

After training, visualize attention:

```python
from src.utils.visualization import visualize_attention

# Load agent
agent.load('checkpoints/best.pt')

# Get attention weights
state = env.reset()
q_values, attn = agent.policy_net(state, return_attention=True)

# Visualize
visualize_attention(attn, grid_size=(20, 20), save_path='attention.png')
```

## Command Reference

```powershell
# Training
python train.py --help

# Testing
python test.py --help

# Run all tests
python run_tests.py

# Individual module tests
python -m src.models.coordinate_network
python -m src.agent.dqn_agent

# TensorBoard
tensorboard --logdir=logs --port=6006
```

## Tips for Best Results

1. **Start Simple**: Train single-scale first, then add multi-scale
2. **Monitor Early**: Check TensorBoard after 100 episodes
3. **Save Often**: Checkpoints every 100 episodes
4. **Test Frequently**: Evaluate every 50 episodes
5. **Compare Baselines**: Run multiple seeds, average results

## Getting Help

- Check `README.md` for detailed documentation
- Review module docstrings for API details
- Run unit tests to verify correct installation
- Open GitHub issue for bugs or questions

Good luck with your research! 🚀
