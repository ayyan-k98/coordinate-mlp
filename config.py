"""
Configuration Module

Centralized configuration for training, testing, and model architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
    
    # Rewards
    coverage_reward: float = 1.0
    revisit_penalty: float = -0.1
    collision_penalty: float = -0.5
    step_penalty: float = -0.01
    
    def get_sensor_range(self, grid_size: int) -> float:
        """Get sensor range for given grid size."""
        return grid_size * self.sensor_range_ratio
    
    def get_max_steps(self, grid_size: int) -> int:
        """Get max steps for given grid size."""
        return int(self.max_steps_ratio * (grid_size / self.base_grid_size) ** 2)


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
