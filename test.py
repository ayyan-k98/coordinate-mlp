"""
Testing and Evaluation Script

Test trained agents on multiple grid sizes to evaluate scale invariance.
"""

import os
import argparse
import json
import numpy as np
import torch
from typing import Dict, List

from dqn_agent import CoordinateDQNAgent
from config import get_default_config, ExperimentConfig
from metrics import CoverageMetrics, aggregate_metrics
from visualization import plot_grid_size_comparison


def create_test_environment(grid_size: int, config: ExperimentConfig, seed: int = None):
    """
    Create test environment (mock version).
    
    Replace with your actual environment.
    """
    class MockTestEnv:
        def __init__(self, grid_size, sensor_range, max_steps, seed):
            self.grid_size = grid_size
            self.sensor_range = sensor_range
            self.max_steps = max_steps
            if seed is not None:
                np.random.seed(seed)
            self.current_step = 0
            
        def reset(self):
            self.current_step = 0
            return np.random.randn(5, self.grid_size, self.grid_size)
        
        def step(self, action):
            self.current_step += 1
            next_state = np.random.randn(5, self.grid_size, self.grid_size)
            reward = np.random.randn()
            done = self.current_step >= self.max_steps
            
            # Simulate coverage increasing
            base_coverage = min(0.3 + self.current_step / self.max_steps * 0.2, 0.5)
            # Add some grid-size degradation
            size_factor = 20 / self.grid_size
            coverage = base_coverage * size_factor
            
            info = {
                'coverage_pct': coverage,
                'steps': self.current_step,
                'num_covered': int(coverage * self.grid_size ** 2),
                'total_cells': self.grid_size ** 2
            }
            
            return next_state, reward, done, info
        
        def get_valid_actions(self):
            return np.ones(9, dtype=bool)
    
    sensor_range = config.environment.get_sensor_range(grid_size)
    max_steps = config.environment.get_max_steps(grid_size)
    
    return MockTestEnv(grid_size, sensor_range, max_steps, seed)


def test_agent_on_size(
    agent: CoordinateDQNAgent,
    grid_size: int,
    config: ExperimentConfig,
    num_episodes: int = 20
) -> Dict[str, float]:
    """
    Test agent on specific grid size.
    
    Args:
        agent: Trained agent
        grid_size: Grid size to test
        config: Experiment configuration
        num_episodes: Number of test episodes
    
    Returns:
        Dictionary of aggregated metrics
    """
    coverages = []
    efficiencies = []
    steps_list = []
    
    for episode in range(num_episodes):
        # Create environment with fixed seed for reproducibility
        env = create_test_environment(grid_size, config, seed=episode)
        
        state = env.reset()
        done = False
        episode_steps = 0
        
        while not done:
            # Greedy action selection
            valid_actions = env.get_valid_actions()
            action = agent.select_action(
                state, 
                epsilon=config.evaluation.epsilon,
                valid_actions=valid_actions
            )
            
            state, reward, done, info = env.step(action)
            episode_steps += 1
        
        # Record metrics
        coverages.append(info['coverage_pct'])
        efficiencies.append(info['coverage_pct'] / episode_steps)
        steps_list.append(episode_steps)
    
    # Aggregate
    return {
        'coverage_mean': np.mean(coverages),
        'coverage_std': np.std(coverages),
        'efficiency_mean': np.mean(efficiencies),
        'efficiency_std': np.std(efficiencies),
        'steps_mean': np.mean(steps_list),
        'steps_std': np.std(steps_list),
    }


def test_grid_size_invariance(
    agent: CoordinateDQNAgent,
    config: ExperimentConfig
) -> Dict[int, Dict[str, float]]:
    """
    Test agent on multiple grid sizes.
    
    Args:
        agent: Trained agent
        config: Experiment configuration
    
    Returns:
        Dictionary mapping grid_size -> metrics
    """
    print("\n" + "="*70)
    print("Testing Grid-Size Invariance")
    print("="*70)
    
    results = {}
    
    for grid_size in config.evaluation.test_grid_sizes:
        print(f"\nTesting on {grid_size}×{grid_size} grid...")
        
        metrics = test_agent_on_size(
            agent, 
            grid_size, 
            config,
            num_episodes=config.evaluation.num_test_episodes
        )
        
        results[grid_size] = metrics
        
        print(f"  Coverage: {metrics['coverage_mean']*100:.1f}% ± "
              f"{metrics['coverage_std']*100:.1f}%")
        print(f"  Efficiency: {metrics['efficiency_mean']:.4f} ± "
              f"{metrics['efficiency_std']:.4f}")
        print(f"  Steps: {metrics['steps_mean']:.1f} ± "
              f"{metrics['steps_std']:.1f}")
    
    return results


def analyze_results(
    results: Dict[int, Dict[str, float]],
    baseline_size: int = 20
) -> None:
    """
    Analyze and print invariance results.
    
    Args:
        results: Results dictionary
        baseline_size: Baseline grid size
    """
    print("\n" + "="*70)
    print("Grid-Size Invariance Analysis")
    print("="*70)
    
    baseline_coverage = results[baseline_size]['coverage_mean']
    
    print(f"\nBaseline ({baseline_size}×{baseline_size}): "
          f"{baseline_coverage*100:.1f}%\n")
    
    print(f"{'Size':>6} {'Coverage':>12} {'Degradation':>15} {'Status':>10}")
    print("-"*50)
    
    for size in sorted(results.keys()):
        metrics = results[size]
        coverage = metrics['coverage_mean']
        std = metrics['coverage_std']
        
        # Compute degradation
        degradation = (coverage - baseline_coverage) / baseline_coverage * 100
        
        # Determine status
        if size == baseline_size:
            status = "BASELINE"
        elif degradation > -10:
            status = "✓ GOOD"
        elif degradation > -20:
            status = "⚠ OK"
        else:
            status = "✗ POOR"
        
        print(f"{size:>4}×{size:<4} "
              f"{coverage*100:>5.1f}%±{std*100:>4.1f}% "
              f"{degradation:>+6.1f}% "
              f"{status:>15}")
    
    # Summary statistics
    print("\n" + "-"*50)
    
    degradations = []
    for size, metrics in results.items():
        if size != baseline_size:
            coverage = metrics['coverage_mean']
            deg = (coverage - baseline_coverage) / baseline_coverage * 100
            degradations.append(deg)
    
    if degradations:
        avg_degradation = np.mean(degradations)
        max_degradation = np.min(degradations)  # Most negative
        
        print(f"\nAverage degradation: {avg_degradation:+.1f}%")
        print(f"Maximum degradation: {max_degradation:+.1f}%")
        
        # Final assessment
        print(f"\nFinal Assessment:")
        if avg_degradation > -15:
            print("  ✓ EXCELLENT scale invariance!")
        elif avg_degradation > -25:
            print("  ✓ GOOD scale invariance")
        elif avg_degradation > -35:
            print("  ⚠ MODERATE scale invariance")
        else:
            print("  ✗ POOR scale invariance")


def save_results(
    results: Dict[int, Dict[str, float]],
    config: ExperimentConfig,
    save_path: str
):
    """Save test results to JSON file."""
    output = {
        'experiment_name': config.experiment_name,
        'test_grid_sizes': config.evaluation.test_grid_sizes,
        'num_test_episodes': config.evaluation.num_test_episodes,
        'results': results
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Coordinate MLP Coverage Agent"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--experiment-name', type=str,
                       default='coordinate_mlp_test',
                       help='Name for test experiment')
    parser.add_argument('--test-sizes', type=int, nargs='+',
                       default=[20, 25, 30, 35, 40],
                       help='Grid sizes to test')
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of test episodes per size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_default_config()
    config.experiment_name = args.experiment_name
    config.evaluation.test_grid_sizes = args.test_sizes
    config.evaluation.num_test_episodes = args.num_episodes
    config.training.device = args.device
    
    print("="*70)
    print(f"Testing Coordinate MLP Coverage Agent")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*70)
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=config.model.input_channels,
        num_actions=config.model.num_actions,
        hidden_dim=config.model.hidden_dim,
        num_freq_bands=config.model.num_freq_bands,
        device=config.training.device
    )
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    agent.load(args.checkpoint)
    print(f"\nModel loaded successfully")
    print(f"  Parameters: {agent.policy_net.get_num_parameters():,}")
    print(f"  Training episodes: {agent.episodes}")
    
    # Test on multiple grid sizes
    results = test_grid_size_invariance(agent, config)
    
    # Analyze results
    analyze_results(results, baseline_size=20)
    
    # Save results
    os.makedirs(config.results_dir, exist_ok=True)
    save_path = os.path.join(config.results_dir, 
                            f"{config.experiment_name}_results.json")
    save_results(results, config, save_path)
    
    # Generate plots
    if args.save_plots:
        plot_path = os.path.join(config.results_dir,
                                f"{config.experiment_name}_comparison.png")
        plot_grid_size_comparison(results, baseline_size=20, save_path=plot_path)
        print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
