"""
Configuration Module

Centralized configuration for training, testing, and model architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Environment Constants
# ============================================================================

# Episode settings
MAX_EPISODE_STEPS = 500
GRID_SIZE = 20  # Default grid size

# Action space
N_ACTIONS = 9  # 8 directions + stay
ACTION_DELTAS = [
    (0, 1),    # 0: N
    (1, 1),    # 1: NE
    (1, 0),    # 2: E
    (1, -1),   # 3: SE
    (0, -1),   # 4: S
    (-1, -1),  # 5: SW
    (-1, 0),   # 6: W
    (-1, 1),   # 7: NW
    (0, 0),    # 8: STAY
]
ACTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'STAY']

# Sensing parameters
NUM_RAYS = 72  # Number of rays for raycasting
SAMPLES_PER_RAY = 20  # Samples per ray

# Probabilistic environment settings
USE_PROBABILISTIC_ENV = False  # Use probabilistic coverage model
PROBABILISTIC_COVERAGE_MIDPOINT = 2.0  # Sigmoid midpoint for coverage probability
PROBABILISTIC_COVERAGE_STEEPNESS = 1.5  # Sigmoid steepness
COVERAGE_THRESHOLD = 0.9  # Threshold for considering a cell covered

# Reward weights
COVERAGE_REWARD = 1.2  # Reward per covered cell
EXPLORATION_REWARD = 0.07  # Reward per newly sensed cell
FRONTIER_BONUS = 0.012  # Bonus per frontier cell
FRONTIER_CAP = 0.25  # Max frontier bonus
COLLISION_PENALTY = -0.25  # Penalty for collision
STEP_PENALTY = -0.0012  # Small penalty per step
STAY_PENALTY = -0.012  # Penalty for staying in place

# Rotation penalty settings
USE_ROTATION_PENALTY = True  # Enable rotation penalty
ROTATION_PENALTY_SMALL = -0.05  # ≤45°
ROTATION_PENALTY_MEDIUM = -0.10  # ≤90°
ROTATION_PENALTY_LARGE = -0.15  # >90°

# Early termination settings
ENABLE_EARLY_TERMINATION = False  # Enable early termination on coverage goal
EARLY_TERM_MIN_STEPS = 100  # Minimum steps before early termination allowed
EARLY_TERM_COVERAGE_TARGET = 0.95  # Coverage threshold for early termination
EARLY_TERM_COMPLETION_BONUS = 10.0  # Bonus for completing coverage goal
EARLY_TERM_TIME_BONUS_PER_STEP = 0.01  # Bonus per step saved


@dataclass
class ModelConfig:
    """Configuration for the Coordinate MLP model."""
    input_channels: int = 5
    num_freq_bands: int = 6
    hidden_dim: int = 256
    num_actions: int = 9
    num_attention_heads: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for DQN training."""
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_capacity: int = 50000
    target_update_tau: float = 0.01
    target_update_frequency: int = 1  # Update every N episodes
    
    # Training schedule
    num_episodes: int = 1500
    warmup_episodes: int = 50  # Episodes before training starts
    eval_frequency: int = 50  # Evaluate every N episodes
    save_frequency: int = 100  # Save checkpoint every N episodes
    
    # Multi-scale training
    multi_scale: bool = True
    grid_sizes: List[int] = field(default_factory=lambda: [15, 20, 25, 30])
    
    # Device
    device: str = 'cuda'  # or 'cpu'


@dataclass
class EnvironmentConfig:
    """Configuration for the coverage environment."""
    base_grid_size: int = 20
    sensor_range_ratio: float = 0.2  # sensor_range = grid_size * ratio
    max_steps_ratio: float = 350  # max_steps = ratio * (grid_size / 20)^2
    obstacle_density: float = 0.15  # Fraction of cells that are obstacles

    # Reward configuration (matches coverage_env.py RewardFunction)
    # SCALED DOWN 10× to prevent gradient explosion and improve training stability
    coverage_reward: float = 0.2  # Reward for new coverage (was 2.0)
    revisit_penalty: float = -0.03  # Penalty for revisiting (was -0.3)
    collision_penalty: float = -0.1  # Penalty for collision (was -1.0)
    step_penalty: float = -0.0002  # Small step penalty (was -0.002)
    frontier_bonus: float = 0.2  # Bonus for frontier exploration (was 0.4)
    coverage_confidence_weight: float = 0.01  # Weight for coverage confidence (was 0.1)

    # Progressive revisit penalty (scales from low to high as episode progresses)
    # Scaled down proportionally to maintain relative strength
    use_progressive_revisit_penalty: bool = True  # Enable progressive penalty
    revisit_penalty_min: float = -0.01  # Initial revisit penalty (was -0.1)
    revisit_penalty_max: float = -0.03  # Final revisit penalty (was -0.3)

    def get_sensor_range(self, grid_size: int) -> float:
        """Get sensor range for given grid size."""
        return grid_size * self.sensor_range_ratio

    def get_max_steps(self, grid_size: int) -> int:
        """Get max steps for given grid size."""
        return int(self.max_steps_ratio * (grid_size / self.base_grid_size) ** 2)

    def get_reward_config(self) -> dict:
        """Get reward configuration as dictionary for CoverageEnvironment."""
        return {
            'coverage_reward': self.coverage_reward,
            'revisit_penalty': self.revisit_penalty,
            'collision_penalty': self.collision_penalty,
            'step_penalty': self.step_penalty,
            'frontier_bonus': self.frontier_bonus,
            'coverage_confidence_weight': self.coverage_confidence_weight,
            'use_progressive_revisit_penalty': self.use_progressive_revisit_penalty,
            'revisit_penalty_min': self.revisit_penalty_min,
            'revisit_penalty_max': self.revisit_penalty_max
        }


@dataclass
class EvaluationConfig:
    """Configuration for testing and evaluation."""
    test_grid_sizes: List[int] = field(default_factory=lambda: [20, 25, 30, 35, 40])
    num_test_episodes: int = 20
    epsilon: float = 0.0  # Greedy evaluation
    render: bool = False
    save_attention_maps: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "coordinate_mlp_coverage"
    seed: int = 42
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_ablation_configs() -> dict:
    """
    Get configurations for ablation studies.
    
    Returns:
        Dictionary of config_name -> config
    """
    configs = {}
    
    # Baseline
    configs['baseline'] = get_default_config()
    
    # Without Fourier features
    config = get_default_config()
    config.experiment_name = "ablation_no_fourier"
    config.model.num_freq_bands = 0  # Only use raw coordinates
    configs['no_fourier'] = config
    
    # Without attention
    config = get_default_config()
    config.experiment_name = "ablation_mean_pool"
    config.model.num_attention_heads = 0  # Use mean pooling instead
    configs['mean_pool'] = config
    
    # Single scale training
    config = get_default_config()
    config.experiment_name = "single_scale"
    config.training.multi_scale = False
    config.training.grid_sizes = [20]
    configs['single_scale'] = config
    
    # Different hidden dimensions
    for hidden_dim in [128, 256, 512]:
        config = get_default_config()
        config.experiment_name = f"hidden_dim_{hidden_dim}"
        config.model.hidden_dim = hidden_dim
        configs[f'hidden_{hidden_dim}'] = config
    
    return configs


if __name__ == "__main__":
    # Test configurations
    print("="*60)
    print("Testing Configuration")
    print("="*60)
    
    # Default config
    config = get_default_config()
    print(f"\nDefault configuration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Model hidden dim: {config.model.hidden_dim}")
    print(f"  Training episodes: {config.training.num_episodes}")
    print(f"  Multi-scale: {config.training.multi_scale}")
    print(f"  Grid sizes: {config.training.grid_sizes}")
    
    # Environment scaling
    print(f"\nEnvironment scaling:")
    for grid_size in [15, 20, 25, 30, 40]:
        sensor_range = config.environment.get_sensor_range(grid_size)
        max_steps = config.environment.get_max_steps(grid_size)
        print(f"  {grid_size}×{grid_size}: "
              f"sensor={sensor_range:.1f}, max_steps={max_steps}")
    
    # Ablation configs
    print(f"\nAblation study configurations:")
    ablation_configs = get_ablation_configs()
    for name, cfg in ablation_configs.items():
        print(f"  {name:20s}: {cfg.experiment_name}")
    
    print("\n✓ Configuration test complete!")
