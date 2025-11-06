"""
Main Training Script for Coordinate MLP Coverage Agent

Supports single-scale and multi-scale curriculum training.
"""

import os
import argparse
import random
import numpy as np
import torch
from typing import Optional

from src.agent.dqn_agent import CoordinateDQNAgent
from src.config import get_default_config, ExperimentConfig
from src.utils.logger import Logger, TensorBoardLogger
from src.utils.metrics import compute_metrics, aggregate_metrics
from src.environment import CoverageEnvironment


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def create_environment(grid_size: int, config: ExperimentConfig) -> CoverageEnvironment:
    """
    Create coverage environment with appropriate parameters.
    
    Args:
        grid_size: Size of the grid
        config: Experiment configuration
    
    Returns:
        Configured CoverageEnvironment instance
    """
    sensor_range = config.environment.get_sensor_range(grid_size)
    max_steps = config.environment.get_max_steps(grid_size)
    
    # Extract reward configuration from environment config
    reward_config = {
        'coverage_reward': 10.0,
        'revisit_penalty': -0.5,
        'collision_penalty': -5.0,
        'step_penalty': -0.01,
        'frontier_bonus': 2.0,
        'coverage_confidence_weight': 0.5
    }
    
    env = CoverageEnvironment(
        grid_size=grid_size,
        num_agents=1,  # Single agent for now
        sensor_range=sensor_range,
        obstacle_density=0.15,
        max_steps=max_steps,
        seed=config.training.seed,
        reward_config=reward_config
    )
    
    return env


def train_episode(
    agent: CoordinateDQNAgent,
    env,
    config: ExperimentConfig,
    episode: int
) -> dict:
    """
    Train for one episode.
    
    Args:
        agent: DQN agent
        env: Environment
        config: Experiment configuration
        episode: Current episode number
    
    Returns:
        Dictionary of episode metrics
    """
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    losses = []
    
    while not done:
        # Select action
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions=valid_actions)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        agent.memory.push(state, action, reward, next_state, done)
        
        # Update agent (after warmup)
        if episode >= config.training.warmup_episodes:
            update_info = agent.update()
            if update_info:
                losses.append(update_info['loss'])
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
    
    # Decay epsilon
    agent.decay_epsilon()
    
    # Compile metrics
    metrics = {
        'episode': episode,
        'reward': episode_reward,
        'steps': episode_steps,
        'coverage': info['coverage_pct'],
        'epsilon': agent.epsilon,
        'memory_size': len(agent.memory),
    }
    
    if losses:
        metrics['loss'] = np.mean(losses)
    
    return metrics


def train(config: ExperimentConfig):
    """
    Main training loop.
    
    Args:
        config: Experiment configuration
    """
    print("="*70)
    print(f"Training Coordinate MLP Coverage Agent")
    print(f"Experiment: {config.experiment_name}")
    print("="*70)
    
    # Set seed
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=config.model.input_channels,
        num_actions=config.model.num_actions,
        hidden_dim=config.model.hidden_dim,
        num_freq_bands=config.model.num_freq_bands,
        learning_rate=config.training.learning_rate,
        gamma=config.training.gamma,
        epsilon_start=config.training.epsilon_start,
        epsilon_end=config.training.epsilon_end,
        epsilon_decay=config.training.epsilon_decay,
        batch_size=config.training.batch_size,
        memory_capacity=config.training.memory_capacity,
        target_update_tau=config.training.target_update_tau,
        device=config.training.device
    )
    
    print(f"\nAgent initialized:")
    print(f"  Device: {agent.device}")
    print(f"  Parameters: {agent.policy_net.get_num_parameters():,}")
    print(f"  Multi-scale: {config.training.multi_scale}")
    if config.training.multi_scale:
        print(f"  Grid sizes: {config.training.grid_sizes}")
    
    # Create loggers
    logger = Logger(config.log_dir, config.experiment_name)
    tb_logger = TensorBoardLogger(config.log_dir, config.experiment_name)
    
    # Training loop
    print(f"\nStarting training for {config.training.num_episodes} episodes...")
    print("-"*70)
    
    best_coverage = 0.0
    
    for episode in range(config.training.num_episodes):
        # Select grid size (multi-scale curriculum)
        if config.training.multi_scale:
            grid_size = random.choice(config.training.grid_sizes)
        else:
            grid_size = config.environment.base_grid_size
        
        # Create environment
        env = create_environment(grid_size, config)
        
        # Train episode
        metrics = train_episode(agent, env, config, episode)
        metrics['grid_size'] = grid_size
        
        # Update target network
        if episode % config.training.target_update_frequency == 0:
            agent.update_target_network()
        
        # Log metrics
        if episode % 10 == 0:
            logger.log_episode(episode, metrics)
        
        if tb_logger.enabled:
            tb_logger.log_scalar('train/reward', metrics['reward'], episode)
            tb_logger.log_scalar('train/coverage', metrics['coverage'], episode)
            tb_logger.log_scalar('train/epsilon', metrics['epsilon'], episode)
            if 'loss' in metrics:
                tb_logger.log_scalar('train/loss', metrics['loss'], episode)
        
        # Save best model
        if metrics['coverage'] > best_coverage:
            best_coverage = metrics['coverage']
            save_path = os.path.join(config.checkpoint_dir, 
                                    f"{config.experiment_name}_best.pt")
            agent.save(save_path)
        
        # Save checkpoint
        if episode % config.training.save_frequency == 0 and episode > 0:
            save_path = os.path.join(config.checkpoint_dir,
                                    f"{config.experiment_name}_ep{episode}.pt")
            agent.save(save_path)
            print(f"\n  Checkpoint saved: {save_path}")
        
        # Evaluation
        if episode % config.training.eval_frequency == 0 and episode > 0:
            print(f"\n  Evaluation at episode {episode}:")
            # Run quick evaluation (placeholder)
            eval_coverage = metrics['coverage']
            print(f"    Current coverage: {eval_coverage*100:.1f}%")
            print(f"    Best coverage: {best_coverage*100:.1f}%")
    
    print("-"*70)
    print("Training complete!")
    print(f"  Best coverage: {best_coverage*100:.1f}%")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    save_path = os.path.join(config.checkpoint_dir, 
                            f"{config.experiment_name}_final.pt")
    agent.save(save_path)
    print(f"  Final model saved: {save_path}")
    
    # Close loggers
    logger.save_summary()
    tb_logger.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Coordinate MLP Coverage Agent"
    )
    parser.add_argument('--experiment-name', type=str, 
                       default='coordinate_mlp_coverage',
                       help='Name of experiment')
    parser.add_argument('--episodes', type=int, default=1500,
                       help='Number of training episodes')
    parser.add_argument('--multi-scale', action='store_true',
                       help='Enable multi-scale training')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_default_config()
    config.experiment_name = args.experiment_name
    config.training.num_episodes = args.episodes
    config.training.multi_scale = args.multi_scale
    config.training.device = args.device
    config.seed = args.seed
    config.model.hidden_dim = args.hidden_dim
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
