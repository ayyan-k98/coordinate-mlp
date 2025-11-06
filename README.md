# Coordinate MLP for Scale-Invariant Coverage Planning

A PyTorch implementation of a coordinate-based neural architecture using Fourier features for multi-agent reinforcement learning coverage tasks. This architecture aims to achieve **grid-size invariance** by processing spatial information through coordinate-based representations rather than convolutional operations.

## 🎯 Overview

Traditional CNN-based approaches struggle with scale invariance in coverage planning. This project explores an alternative architecture that:

- Uses **Fourier positional encoding** to represent spatial coordinates
- Processes each grid cell independently with **MLPs**
- Aggregates information using **multi-head attention**
- Achieves **coordinate-based scale invariance**

Based on techniques from:
- NeRF (Neural Radiance Fields)
- Transformer architectures
- Implicit neural representations

## 🏗️ Architecture

```
Grid[H, W, 5] → Coordinate Generator → Fourier Encoder
                                             ↓
                                    [Coord Features + Grid Values]
                                             ↓
                                      Cell Feature MLP
                                             ↓
                                     Attention Pooling
                                             ↓
                                      Dueling Q-Network
                                             ↓
                                    Q-values[9 actions]
```

### Key Components

1. **Fourier Positional Encoding** (`src/models/positional_encoding.py`)
   - Converts (x, y) ∈ [-1, 1]² to high-frequency features
   - Output dimension: 2 + 4*L (L = number of frequency bands)
   - Default: 6 bands → 26-dimensional features

2. **Cell Feature MLP** (`src/models/cell_encoder.py`)
   - Processes each cell independently
   - Input: Coordinate features + grid values (26 + 5 = 31 dims)
   - Output: Hidden dimension (default: 256 dims)

3. **Attention Pooling** (`src/models/attention.py`)
   - Multi-head attention over all cells
   - Learns to focus on important regions (frontiers, agent position)
   - Aggregates spatial information

4. **Dueling Q-Network** (`src/models/q_network.py`)
   - Separate value and advantage streams
   - Maps aggregated features to action Q-values

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- (Optional) TensorBoard for logging

### Setup

```bash
# Clone repository
cd "d:/pro/marl/coordinate mlp"

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m src.models.positional_encoding
python -m src.models.coordinate_network
python -m src.agent.dqn_agent
```

## 🚀 Quick Start

### Training

```bash
# Single-scale training (20×20 grid)
python train.py --experiment-name single_scale --episodes 1500

# Multi-scale training (curriculum learning)
python train.py --experiment-name multi_scale --episodes 2000 --multi-scale

# Custom configuration
python train.py --experiment-name custom \
    --episodes 1500 \
    --multi-scale \
    --hidden-dim 512 \
    --device cuda
```

### Testing

```bash
# Test on multiple grid sizes
python test.py --checkpoint checkpoints/multi_scale_best.pt \
    --test-sizes 20 25 30 35 40 \
    --num-episodes 20 \
    --save-plots
```

## 📊 Expected Results

### Performance Targets

| Grid Size | Expected Coverage | Degradation from 20×20 |
|-----------|------------------|------------------------|
| 20×20     | 40-45%          | Baseline              |
| 25×25     | 38-43%          | -5 to -10%            |
| 30×30     | 36-41%          | -10 to -15%           |
| 35×35     | 34-39%          | -15 to -20%           |
| 40×40     | 32-37%          | -20 to -25%           |

**Goal**: Keep degradation < 25% for 2× grid size increase (20×20 → 40×40)

### Comparison with FCN Baseline

| Metric | FCN + Spatial Softmax | Coordinate MLP (Expected) |
|--------|----------------------|---------------------------|
| 20×20 Coverage | 48% | 40-45% |
| 40×40 Coverage | 22% (-55%) | 32-37% (-20%) |
| Parameters | ~350K | ~400K |
| Training Episodes | 1500 | 2000-3000 |

## 🧪 Experiments

### Ablation Studies

```bash
# Without Fourier features
python train.py --experiment-name ablation_no_fourier --num-freq-bands 0

# Different hidden dimensions
python train.py --experiment-name ablation_hidden_128 --hidden-dim 128
python train.py --experiment-name ablation_hidden_512 --hidden-dim 512

# Single-scale vs Multi-scale
python train.py --experiment-name single_scale --episodes 1500
python train.py --experiment-name multi_scale --episodes 2000 --multi-scale
```

## 📁 Project Structure

```
coordinate mlp/
├── src/
│   ├── models/
│   │   ├── positional_encoding.py    # Fourier features
│   │   ├── cell_encoder.py           # Per-cell MLP
│   │   ├── attention.py              # Attention pooling
│   │   ├── q_network.py              # Dueling Q-network
│   │   └── coordinate_network.py     # Main architecture
│   ├── agent/
│   │   ├── replay_buffer.py          # Experience replay
│   │   └── dqn_agent.py              # DQN training logic
│   ├── utils/
│   │   ├── logger.py                 # Logging utilities
│   │   ├── metrics.py                # Performance metrics
│   │   └── visualization.py          # Plotting functions
│   └── config.py                     # Configuration management
├── train.py                          # Training script
├── test.py                           # Evaluation script
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🔬 Technical Details

### Coordinate Normalization

Grid coordinates are normalized to [-1, 1]²:
- Ensures scale invariance
- Allows Fourier encoding to work across sizes
- Example: 20×20 and 40×40 both map to same coordinate space

### Fourier Features

For each coordinate (x, y):
```
features = [x, y, sin(2π·x), cos(2π·x), sin(4π·x), cos(4π·x), ..., sin(64π·x), cos(64π·x)]
```

6 frequency bands (2⁰, 2¹, ..., 2⁵) → 26-dimensional encoding

### Multi-Scale Training

Curriculum learning across grid sizes:
1. Sample grid size uniformly from [15, 20, 25, 30]
2. Scale sensor range and max steps proportionally
3. Agent learns generalizable spatial patterns

### Attention Mechanism

- **Query**: Learnable vector (what to look for)
- **Keys/Values**: Per-cell features
- **Output**: Weighted aggregation of important cells

## 📈 Training Tips

1. **Warmup Period**: Train for 50-100 episodes before updating to fill replay buffer
2. **Epsilon Decay**: Use exponential decay (0.995) to balance exploration/exploitation
3. **Target Network**: Update every episode with soft updates (τ=0.01)
4. **Batch Size**: 32 works well; increase to 64 for more stable training
5. **Gradient Clipping**: Clip to norm=10 to prevent instability

## ⚠️ Known Limitations

1. **Slower than CNN**: O(N²) complexity for N×N grid (MLPs on each cell)
2. **Needs More Data**: Requires 2000-3000 episodes vs 1500 for CNN
3. **Weaker Spatial Bias**: No built-in locality; must learn from scratch
4. **Memory Usage**: Attention over all cells can be memory-intensive

## 🔮 Future Work

- [ ] Sparse attention for large grids (only attend to nearby cells)
- [ ] Hierarchical processing (coarse-to-fine)
- [ ] Pre-training on synthetic data
- [ ] Multi-agent coordination with coordinate-based communication
- [ ] Transfer to continuous action spaces

## 📚 References

1. Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields" (2020)
2. Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions" (2020)
3. Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
4. Vaswani et al. "Attention Is All You Need" (2017)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{coordinate_mlp_coverage,
  title={Coordinate MLP for Scale-Invariant Coverage Planning},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/coordinate-mlp-coverage}}
}
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Integration with actual coverage environments
- Hyperparameter tuning
- Ablation studies
- Comparison with other architectures

## 📄 License

MIT License - see LICENSE file for details

## 🙋 FAQ

**Q: Why not just use CNN?**  
A: CNNs have strong locality bias that hurts scale invariance. This explores an alternative approach inspired by implicit neural representations.

**Q: Will this beat CNN baseline?**  
A: On 20×20, probably not. But it should generalize better to larger/smaller grids with less performance degradation.

**Q: How long does training take?**  
A: ~2-4 hours for 1500 episodes on GPU (mock environment). Real environment will be slower.

**Q: Can I use this for other tasks?**  
A: Yes! The coordinate-based architecture is general. Just replace the environment and reward structure.

---

**Status**: 🚧 Research prototype - use with caution in production!

For questions or issues, please open a GitHub issue or contact the authors.
