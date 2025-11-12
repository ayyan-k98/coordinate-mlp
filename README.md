# FCN Coverage Planning with Curriculum Learning

A PyTorch implementation of a Fully Convolutional Network (FCN) for single-agent coverage planning with curriculum learning. This architecture uses proven CNN components for stable, high-performance training across diverse map types.

## 🎯 Overview

This project implements a robust coverage planning agent using:

- **Fully Convolutional Network (FCN)** with spatial encoder
- **Curriculum learning** across map types (empty → random → structured)
- **Multi-scale training** (20×20 to 30×30 grids)
- **POMDP formulation** with probabilistic sensing
- **Reward shaping** with frontier detection and first-visit bonuses

**Expected Performance:** 65-75% coverage across diverse maps in 1500 episodes

## 🏗️ Architecture

```
Grid[B, 5, H, W] → Spatial Encoder (3 conv blocks)
                         ↓
                   [B, 256, H, W]
                         ↓
                  Spatial Softmax (weighted mean)
                         ↓
                    [B, 512]
                         ↓
                  Dueling Q-Head
                         ↓
                Q-values [B, 9]
```

### Key Components

1. **Spatial Encoder** (`fcn_network.py`)
   - 3 convolutional blocks with BatchNorm and ReLU
   - Channels: 5 → 64 → 128 → 256
   - Extracts spatial features from grid

2. **Spatial Softmax** 
   - Temperature-scaled softmax over spatial dimensions
   - Computes weighted (x, y) coordinates for each channel
   - Output: [B, 256*2] = [B, 512] features

3. **Dueling Q-Head**
   - Separate value V(s) and advantage A(s,a) streams
   - Combines: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
   - Final layer scaled by 0.1× for stability

**Parameters:** 548,362 total (lightweight and efficient)

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- NumPy
- Matplotlib
- TensorBoard (for logging)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python run_tests.py
```

## 🚀 Quick Start

### Training

```bash
# Full training with curriculum learning (RECOMMENDED)
python train.py --experiment-name fcn_baseline --episodes 1500 --device cuda

# Quick test (50 episodes)
python train.py --experiment-name fcn_test --episodes 50 --device cuda
```

See `TRAINING_CONFIGURATION_GUIDE.md` for advanced options.

### Evaluation

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

### Performance Targets

| Map Type | 20×20 Coverage | 30×30 Coverage |
|----------|----------------|----------------|
| Empty    | 95-99%        | 90-95%        |
| Random   | 70-80%        | 65-75%        |
| Corridor | 60-70%        | 55-65%        |
| Cave     | 50-60%        | 45-55%        |
| **Overall** | **65-75%** | **60-70%** |

## 🧪 Key Features

### Curriculum Learning
- **Phase 1**: Empty maps (easy exploration)
- **Phase 2**: Random obstacles (navigation)
- **Phase 3**: Structured maps (corridors, rooms, caves)
- **Progressive difficulty** across 1500 episodes

### POMDP Formulation
- **Probabilistic sensing**: Detection probability decreases with distance
- **Partial observability**: Agent only "sees" within sensor range
- **Realistic modeling**: Simulates real-world sensor uncertainty

### Reward Shaping
- **Coverage reward** (0.5): New cell discovery
- **First visit bonus** (0.5): Exploration incentive  
- **Frontier bonus** (0.2): Guidance to unexplored boundaries
- **Progressive penalties**: Discourage revisiting over time

## 📁 Project Structure

```
fcn/
├── Core Implementation
│   ├── fcn_network.py              # FCN architecture
│   ├── dqn_agent.py                # DQN training logic
│   ├── coverage_env.py             # POMDP environment
│   ├── replay_buffer.py            # Experience replay
│   ├── curriculum.py               # Curriculum learning
│   ├── map_generator.py            # Map generation
│   └── config.py                   # Configuration
│
├── Training & Evaluation
│   ├── train.py                    # Main training script
│   ├── test.py                     # Evaluation script
│   ├── run_tests.py                # Unit test runner
│   └── requirements.txt            # Dependencies
│
├── Utilities
│   ├── logger.py                   # Logging utilities
│   ├── metrics.py                  # Performance metrics
│   ├── visualization.py            # Plotting functions
│   └── view_logs.py                # Log viewer
│
└── Documentation
    ├── README.md                   # This file
    ├── QUICKSTART.md               # Getting started
    ├── ALL_FIXES_COMPLETE.md       # Recent fixes applied
    ├── TRAINING_CONFIGURATION_GUIDE.md
    ├── CURRICULUM_LEARNING.md
    ├── VALIDATION_SYSTEM.md
    └── ARCHITECTURE_DIAGRAM.md
```

## 🔬 Technical Details

### Training Stability

All major issues have been resolved (see `ALL_FIXES_COMPLETE.md`):
- ✅ **Target clamping** prevents Q-value explosions
- ✅ **Update frequency** (every 4 steps) for 3-4× speedup
- ✅ **Early stopping disabled** for curriculum completion
- ✅ **Reward bugs fixed** (early completion, frontier, first visit)

### Network Architecture

- **Input**: [B, 5, H, W] grid (coverage, visited, obstacles, agent, confidence)
- **Spatial encoder**: 3 conv blocks (5→64→128→256 channels)
- **Spatial softmax**: Temperature-scaled pooling → [B, 512]
- **Dueling Q-head**: Value + Advantage streams → [B, 9] Q-values
- **Total parameters**: 548,362

### Training Configuration

- **Optimizer**: Adam (lr=1e-4)
- **Replay buffer**: 50,000 transitions
- **Batch size**: 32
- **Discount (γ)**: 0.99
- **Target update**: Polyak averaging (τ=0.01)
- **Gradient clipping**: max_norm=0.2
- **Mixed precision**: FP16 with conservative scaler

## 📈 Training Tips

1. **Full curriculum**: Train for full 1500 episodes (early stopping disabled)
2. **GPU recommended**: Training takes 4-6 hours on GPU (vs 12-24 on CPU)
3. **Monitor per-map-type**: Check validation breakdown to see curriculum progress
4. **Expected timeline**:
   - Episodes 0-500: Learning empty and random maps
   - Episodes 500-1000: Mastering structured maps
   - Episodes 1000-1500: Fine-tuning and generalization
5. **Checkpoints saved**: Best model auto-saved when validation improves

## ⚠️ Known Considerations

1. **Curriculum dependent**: Performance relies on completing all curriculum phases
2. **POMDP uncertainty**: Probabilistic sensing adds stochasticity to results
3. **Scale limitations**: Trained on 20×20 and 30×30, may need retraining for larger maps
4. **Single agent**: Current implementation is single-agent only

## 🔮 Future Work

- [ ] Phase-aware validation metrics for better curriculum tracking
- [ ] Larger grid sizes (40×40, 50×50) with multi-scale curriculum
- [ ] Multi-agent coordination and communication
- [ ] Transfer learning across map distributions
- [ ] Real-world robot deployment with actual sensors

## 📚 References

1. Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
2. van Hasselt et al. "Deep Reinforcement Learning with Double Q-learning" (2016)
3. Mnih et al. "Human-level control through deep reinforcement learning" (2015)
4. Bengio et al. "Curriculum Learning" (2009)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{fcn_coverage_planning,
  title={FCN Coverage Planning with Curriculum Learning},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/fcn-coverage}}
}
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Phase-aware validation metrics
- Larger-scale experiments (40×40+ grids)
- Real-world robot deployment
- Multi-agent extensions

## 📄 License

MIT License - see LICENSE file for details

## 🙋 FAQ

**Q: How long does training take?**  
A: 4-6 hours for 1500 episodes on GPU with all optimizations enabled.

**Q: What if I see gradient explosions?**  
A: All gradient explosion fixes are already applied. If issues persist, see `ALL_FIXES_COMPLETE.md`.

**Q: Can I stop training early?**  
A: No - early stopping is disabled because it's incompatible with curriculum learning. Train full 1500 episodes.

**Q: How do I know training is working?**  
A: Watch per-map-type validation breakdown. You should see progressive improvement across map types.

**Q: Can I use this for multi-agent?**  
A: Current implementation is single-agent. Multi-agent coordination is future work.

---

**Status**: ✅ Stable and production-ready after comprehensive fixes!

For questions or issues, see documentation or open a GitHub issue.
