"""
Main Training Script for Coordinate MLP Coverage Agent

Supports single-scale and multi-scale curriculum training.
"""

import os
import argparse
import random
import time
import numpy as np
import torch
from typing import Optional

from dqn_agent import CoordinateDQNAgent
from config import get_default_config, ExperimentConfig
from logger import Logger, TensorBoardLogger
from metrics import compute_metrics, aggregate_metrics
from coverage_env import CoverageEnvironment
from curriculum import CurriculumScheduler, create_default_curriculum, create_fast_curriculum, create_no_curriculum
from performance_optimizations import (
    PerformanceConfig,
    setup_performance_optimizations,
    optimize_model,
    MixedPrecisionTrainer,
    get_optimal_batch_size
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def create_environment(
    grid_size: int,
    config: ExperimentConfig,
    map_type: str = "random"  # NEW: Support curriculum map types
) -> CoverageEnvironment:
    """
    Create coverage environment with appropriate parameters.

    Args:
        grid_size: Size of the grid
        config: Experiment configuration
        map_type: Map type for curriculum learning

    Returns:
        Configured CoverageEnvironment instance
    """
    sensor_range = config.environment.get_sensor_range(grid_size)
    max_steps = config.environment.get_max_steps(grid_size)

    # FIXED: Use reward configuration from config (not hard-coded!)
    reward_config = config.environment.get_reward_config()

    env = CoverageEnvironment(
        grid_size=grid_size,
        num_agents=1,  # Single agent for now
        sensor_range=sensor_range,
        obstacle_density=config.environment.obstacle_density,
        max_steps=max_steps,
        seed=None,  # Let curriculum handle seeding per episode
        reward_config=reward_config,
        map_type=map_type  # NEW: Pass map type for curriculum
    )

    return env


def evaluate_agent(
    agent: CoordinateDQNAgent,
    config: ExperimentConfig,
    num_eval_episodes: int = 10,
    eval_grid_sizes: Optional[list] = None,
    eval_map_types: Optional[list] = None
) -> dict:
    """
    Evaluate agent performance without exploration.
    
    Args:
        agent: DQN agent to evaluate
        config: Experiment configuration
        num_eval_episodes: Number of evaluation episodes per configuration
        eval_grid_sizes: Grid sizes to evaluate (None = use training sizes)
        eval_map_types: Map types to evaluate (None = use all types)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Default evaluation configurations
    if eval_grid_sizes is None:
        eval_grid_sizes = config.training.grid_sizes if config.training.multi_scale else [config.environment.base_grid_size]
    
    if eval_map_types is None:
        eval_map_types = ["empty", "random", "corridor", "room", "cave", "lshape"]
    
    # Store results
    all_results = []
    results_by_size = {size: [] for size in eval_grid_sizes}
    results_by_type = {map_type: [] for map_type in eval_map_types}
    
    print(f"\n{'='*70}")
    print(f"Running Evaluation: {len(eval_grid_sizes)} sizes × {len(eval_map_types)} types × {num_eval_episodes} episodes")
    print(f"{'='*70}")
    
    # Evaluate on each configuration
    for grid_size in eval_grid_sizes:
        for map_type in eval_map_types:
            episode_results = []
            
            for eval_ep in range(num_eval_episodes):
                # Create evaluation environment
                env = create_environment(grid_size, config, map_type=map_type)
                
                # Run episode with greedy policy (epsilon=0)
                state = env.reset()
                done = False
                episode_reward = 0
                episode_steps = 0
                
                while not done:
                    # Greedy action selection (no exploration)
                    agent_pos = (env.state.agents[0].y, env.state.agents[0].x)
                    action = agent.select_action(state, epsilon=0.0, agent_pos=agent_pos)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                
                # Calculate cells-per-step efficiency
                total_cells = grid_size * grid_size
                cells_covered = info['coverage_pct'] * total_cells
                efficiency = cells_covered / episode_steps if episode_steps > 0 else 0

                result = {
                    'grid_size': grid_size,
                    'map_type': map_type,
                    'coverage': info['coverage_pct'],
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'efficiency': efficiency,  # Cells per step
                    'collisions': info.get('collisions', 0)
                }
                
                episode_results.append(result)
                all_results.append(result)
                results_by_size[grid_size].append(result)
                results_by_type[map_type].append(result)
            
            # Print summary for this configuration
            avg_coverage = np.mean([r['coverage'] for r in episode_results])
            avg_steps = np.mean([r['steps'] for r in episode_results])
            print(f"  {grid_size}×{grid_size} {map_type:10s}: Coverage={avg_coverage*100:5.1f}%, Steps={avg_steps:5.1f}")
    
    # Compute aggregate statistics
    eval_metrics = {
        'overall': {
            'coverage_mean': np.mean([r['coverage'] for r in all_results]),
            'coverage_std': np.std([r['coverage'] for r in all_results]),
            'reward_mean': np.mean([r['reward'] for r in all_results]),
            'steps_mean': np.mean([r['steps'] for r in all_results]),
            'efficiency_mean': np.mean([r['efficiency'] for r in all_results]),
            'collisions_mean': np.mean([r['collisions'] for r in all_results]),
        }
    }
    
    # Per grid size statistics
    eval_metrics['by_size'] = {}
    for size, results in results_by_size.items():
        if results:
            eval_metrics['by_size'][size] = {
                'coverage_mean': np.mean([r['coverage'] for r in results]),
                'coverage_std': np.std([r['coverage'] for r in results]),
                'reward_mean': np.mean([r['reward'] for r in results]),
                'steps_mean': np.mean([r['steps'] for r in results]),
            }
    
    # Per map type statistics
    eval_metrics['by_type'] = {}
    for map_type, results in results_by_type.items():
        if results:
            eval_metrics['by_type'][map_type] = {
                'coverage_mean': np.mean([r['coverage'] for r in results]),
                'coverage_std': np.std([r['coverage'] for r in results]),
                'reward_mean': np.mean([r['reward'] for r in results]),
                'steps_mean': np.mean([r['steps'] for r in results]),
            }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Evaluation Summary:")
    print(f"  Overall Coverage: {eval_metrics['overall']['coverage_mean']*100:.1f}% ± {eval_metrics['overall']['coverage_std']*100:.1f}%")
    print(f"  Overall Reward:   {eval_metrics['overall']['reward_mean']:.1f}")
    print(f"  Overall Steps:    {eval_metrics['overall']['steps_mean']:.1f}")
    print(f"  Overall Efficiency: {eval_metrics['overall']['efficiency_mean']:.4f}")
    print(f"{'='*70}\n")
    
    return eval_metrics


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
    # Start timing
    start_time = time.time()
    
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    losses = []
    q_values_list = []
    td_errors_list = []
    grad_norms_list = []
    
    while not done:
        # Select action
        valid_actions = env.get_valid_actions()
        
        # Get current agent position for POMDP
        agent_pos = (env.state.agents[0].y, env.state.agents[0].x)
        
        action = agent.select_action(state, valid_actions=valid_actions, agent_pos=agent_pos)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Get next agent position (after step)
        next_agent_pos = (env.state.agents[0].y, env.state.agents[0].x)
        
        # Store transition with visibility masks if POMDP
        # Use grid-specific replay buffer
        grid_size = env.grid_size
        if agent.use_pomdp:
            state_with_mask = agent._add_visibility_mask(state, agent_pos)
            next_state_with_mask = agent._add_visibility_mask(next_state, next_agent_pos)
            agent.store_transition(state_with_mask, action, reward, next_state_with_mask, done, grid_size)
        else:
            agent.store_transition(state, action, reward, next_state, done, grid_size)
        
        # Update agent (after warmup, every N steps)
        # Pass grid_size to sample from correct buffer
        if episode >= config.training.warmup_episodes:
            if episode_steps % config.training.update_frequency == 0:
                update_info = agent.update(grid_size=grid_size)
                if update_info:
                    losses.append(update_info['loss'])
                    q_values_list.append(update_info['q_mean'])
                    td_errors_list.append(update_info['td_error_mean'])
                    grad_norms_list.append(update_info['grad_norm'])

                    # Explosion detection (FCN-level threshold)
                    if update_info['grad_norm'] > 25.0:
                        print(f"\n⚠️  GRADIENT EXPLOSION DETECTED at step {episode_steps}")
                        print(f"   Gradient norm: {update_info['grad_norm']:.2f} (threshold: 25.0)")
                        print(f"   Q-values: mean={update_info['q_mean']:.2f}, max={update_info['q_max']:.2f}")
                        print(f"   Stopping episode early to prevent training collapse\n")
                        # Mark explosion and break episode
                        done = True

        state = next_state  # Keep raw state for next iteration
        episode_reward += reward
        episode_steps += 1
    
    # Decay epsilon
    agent.decay_epsilon()
    
    # Compute timing metrics
    episode_time = time.time() - start_time
    steps_per_second = episode_steps / episode_time if episode_time > 0 else 0.0
    
    # Compute efficiency metrics
    grid_size = env.grid_size
    total_cells = grid_size * grid_size
    cells_covered = info['coverage_pct'] * total_cells
    efficiency = cells_covered / episode_steps if episode_steps > 0 else 0.0
    # Compile enhanced metrics
    metrics = {
        'episode': episode,
        'reward': episode_reward,
        'steps': episode_steps,
        'coverage': info['coverage_pct'],
        'epsilon': agent.epsilon,
        # Report total memory across all grid sizes
        'memory_size': sum(len(mem) for mem in agent.memories.values()),
        'memory_size_15': len(agent.memories[15]),
        'memory_size_20': len(agent.memories[20]),
        'memory_size_25': len(agent.memories[25]),
        'memory_size_30': len(agent.memories[30]),
        # Performance metrics
        'episode_time': episode_time,
        'steps_per_second': steps_per_second,
        # Coverage quality metrics
        'efficiency': efficiency,
        'collisions': info.get('collisions', 0),
        'revisits': info.get('revisits', 0),
        'num_frontiers': info.get('num_frontiers', 0),
        # Reward breakdown
        'reward_coverage': info.get('reward_breakdown', {}).get('coverage', 0),
        'reward_confidence': info.get('reward_breakdown', {}).get('confidence', 0),
        'reward_revisit': info.get('reward_breakdown', {}).get('revisit', 0),
        'reward_frontier': info.get('reward_breakdown', {}).get('frontier', 0),
    }
    
    # Training diagnostics
    if losses:
        metrics['loss'] = np.mean(losses)
        metrics['loss_std'] = np.std(losses)
    if q_values_list:
        metrics['q_mean'] = np.mean(q_values_list)
        metrics['q_std'] = np.std(q_values_list)
    if td_errors_list:
        metrics['td_error'] = np.mean(td_errors_list)
    if grad_norms_list:
        metrics['grad_norm'] = np.mean(grad_norms_list)
    
    return metrics


def train(config: ExperimentConfig, curriculum_type: str = 'default'):
    """
    Main training loop.

    Args:
        config: Experiment configuration
        curriculum_type: Type of curriculum ('default', 'fast', or 'none')
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

    # Setup performance optimizations
    # Setup performance optimizations
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')

    # AMP toggle for debugging gradient explosions
    USE_AMP = True  # Set to False to test FP32-only training

    perf_config = PerformanceConfig(
        use_amp=USE_AMP,  # Can disable for debugging
        compile_model=hasattr(torch, 'compile'),
        use_cudnn_benchmark=True,
        use_tf32=True,
    )
    setup_performance_optimizations(perf_config, device)

    if not USE_AMP:
        print("⚠️  Mixed Precision (AMP) DISABLED - Using FP32 only")
    # Initialize mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(enabled=perf_config.use_amp, device=device)

    # Initialize curriculum scheduler based on type
    if curriculum_type == 'fast':
        curriculum_config = create_fast_curriculum()
    elif curriculum_type == 'none':
        curriculum_config = create_no_curriculum()
    else:  # 'default'
        curriculum_config = create_default_curriculum()

    curriculum = CurriculumScheduler(
        curriculum_config,
        grid_sizes=config.training.grid_sizes if config.training.multi_scale else [config.environment.base_grid_size]
    )

    print(f"\nCurriculum Learning: {'Enabled' if curriculum_config.enabled else 'Disabled'}")
    if curriculum_config.enabled:
        print(f"  Phases: {len(curriculum_config.phases)}")
        print(f"  Total Episodes: {curriculum.total_curriculum_episodes}")
        curriculum.print_status()

    # Create agent (with mixed precision enabled)
    # POMDP adds visibility mask as 6th channel
    use_pomdp = True
    input_channels = 6 if use_pomdp else 5
    
    agent = CoordinateDQNAgent(
        input_channels=input_channels,
        num_actions=config.model.num_actions,
        use_pomdp=use_pomdp,
        sensor_range=4.0,
        learning_rate=config.training.learning_rate,
        gamma=config.training.gamma,
        epsilon_start=config.training.epsilon_start,
        epsilon_end=config.training.epsilon_end,
        epsilon_decay=config.training.epsilon_decay,
        batch_size=config.training.batch_size,
        memory_capacity=config.training.memory_capacity,
        target_update_tau=config.training.target_update_tau,
        device=config.training.device,
        use_mixed_precision=perf_config.use_amp  # Enable AMP if available
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
    best_episode = 0
    episodes_without_improvement = 0
    patience = 200  # Early stopping: stop if no improvement for 200 episodes (disabled by default)
    
    for episode in range(config.training.num_episodes):
        # Sample from curriculum (grid size + map type)
        grid_size = curriculum.sample_grid_size()
        map_type = curriculum.sample_map_type()

        
        # Create environment with curriculum map type
        env = create_environment(grid_size, config, map_type=map_type)
        
        # Apply curriculum-based epsilon adjustment
        # Check if we should boost epsilon (at phase start)
        epsilon_boost = curriculum.should_boost_epsilon()
        if epsilon_boost is not None:
            agent.epsilon = epsilon_boost
            print(f"\n🔍 Epsilon boosted to {epsilon_boost:.2f} for new phase!")
        
        # Apply epsilon floor for current phase
        agent.epsilon = curriculum.get_epsilon_adjustment(agent.epsilon)
        
        # Train episode
        metrics = train_episode(agent, env, config, episode)
        metrics['grid_size'] = grid_size
        metrics['map_type'] = map_type

        # Advance curriculum
        phase_changed = curriculum.step()
        if phase_changed:
            print(f"\n{'='*70}")
            print(f"  Curriculum Phase Transition!")
            print(f"{'='*70}")
            curriculum.print_status()
        
        # Update target network
        if episode % config.training.target_update_frequency == 0:
            agent.update_target_network()
        
        # Log metrics (more frequently to show curriculum progress)
        if episode % 10 == 0:
            logger.log_episode(episode, metrics)
            # Log curriculum info
            progress = curriculum.get_progress()
            print(f"Episode {episode}: "
                  f"Phase={progress['phase_name']}, "
                  f"Map={map_type:10s}, "
                  f"Grid={grid_size}x{grid_size}, "
                  f"Coverage={metrics['coverage']*100:.1f}%, "
                  f"Epsilon={agent.epsilon:.3f}")
        
        # Enhanced TensorBoard logging
        if tb_logger.enabled:
            # Basic metrics
            tb_logger.log_scalar('train/reward', metrics['reward'], episode)
            tb_logger.log_scalar('train/coverage', metrics['coverage'], episode)
            tb_logger.log_scalar('train/epsilon', metrics['epsilon'], episode)
            tb_logger.log_scalar('train/steps', metrics['steps'], episode)
            
            # Performance metrics
            tb_logger.log_scalar('perf/episode_time', metrics.get('episode_time', 0), episode)
            tb_logger.log_scalar('perf/steps_per_second', metrics.get('steps_per_second', 0), episode)
            
            # Coverage quality metrics
            tb_logger.log_scalar('coverage/efficiency', metrics.get('efficiency', 0), episode)
            tb_logger.log_scalar('coverage/collisions', metrics.get('collisions', 0), episode)
            tb_logger.log_scalar('coverage/revisits', metrics.get('revisits', 0), episode)
            tb_logger.log_scalar('coverage/num_frontiers', metrics.get('num_frontiers', 0), episode)
            
            # Reward breakdown
            if 'reward_coverage' in metrics:
                reward_components = {
                    'coverage': metrics.get('reward_coverage', 0),
                    'confidence': metrics.get('reward_confidence', 0),
                    'revisit': metrics.get('reward_revisit', 0),
                    'frontier': metrics.get('reward_frontier', 0),
                }
                tb_logger.log_scalars('reward/breakdown', reward_components, episode)
            
            # Training diagnostics
            if 'loss' in metrics:
                tb_logger.log_scalar('train/loss', metrics['loss'], episode)
                if 'loss_std' in metrics:
                    tb_logger.log_scalar('train/loss_std', metrics['loss_std'], episode)
            
            if 'q_mean' in metrics:
                tb_logger.log_scalar('train/q_mean', metrics['q_mean'], episode)
                tb_logger.log_scalar('train/q_std', metrics.get('q_std', 0), episode)
            
            if 'td_error' in metrics:
                tb_logger.log_scalar('train/td_error', metrics['td_error'], episode)
            
            if 'grad_norm' in metrics:
                tb_logger.log_scalar('train/grad_norm', metrics['grad_norm'], episode)
            
            # Curriculum tracking
            if curriculum_config.enabled:
                progress = curriculum.get_progress()
                tb_logger.log_scalar('curriculum/phase_idx', progress['phase_idx'], episode)
                tb_logger.log_scalar('curriculum/phase_progress', progress['phase_progress'], episode)
                tb_logger.log_scalar('curriculum/overall_progress', progress['overall_progress'], episode)
                
                # Track phase-specific epsilon
                phase = curriculum.get_current_phase()
                if phase.epsilon_floor is not None:
                    tb_logger.log_scalar('curriculum/epsilon_floor', phase.epsilon_floor, episode)
                
                # Per-map-type performance tracking
                tb_logger.log_scalar(f'map_type/{map_type}/coverage', metrics['coverage'], episode)
                tb_logger.log_scalar(f'map_type/{map_type}/reward', metrics['reward'], episode)
                tb_logger.log_scalar(f'map_type/{map_type}/efficiency', metrics.get('efficiency', 0), episode)
            
            # Multi-scale tracking
            if config.training.multi_scale:
                tb_logger.log_scalar(f'multiscale/coverage_{grid_size}x{grid_size}', 
                                    metrics['coverage'], episode)
        
        # Note: best_coverage is now updated during validation (every eval_frequency episodes)
        # This ensures we save models based on generalization, not training performance
        
        # Save checkpoint
        if episode % config.training.save_frequency == 0 and episode > 0:
            save_path = os.path.join(config.checkpoint_dir,
                                    f"{config.experiment_name}_ep{episode}.pt")
            agent.save(save_path)
            print(f"\n  Checkpoint saved: {save_path}")
        
        # Periodic Validation
        if episode % config.training.eval_frequency == 0 and episode > 0:
            print(f"\n{'='*70}")
            print(f"VALIDATION at Episode {episode}")
            print(f"{'='*70}")
            
            # Run proper evaluation with greedy policy
            eval_metrics = evaluate_agent(
                agent=agent,
                config=config,
                num_eval_episodes=5,  # 5 episodes per configuration
                eval_grid_sizes=[20, 30] if config.training.multi_scale else [config.environment.base_grid_size],
                eval_map_types=["empty", "random", "corridor", "cave"]  # Subset for speed
            )
            
            # Log validation metrics to TensorBoard
            if tb_logger.enabled:
                # Overall validation metrics
                tb_logger.log_scalar('val/coverage', eval_metrics['overall']['coverage_mean'], episode)
                tb_logger.log_scalar('val/coverage_std', eval_metrics['overall']['coverage_std'], episode)
                tb_logger.log_scalar('val/reward', eval_metrics['overall']['reward_mean'], episode)
                tb_logger.log_scalar('val/steps', eval_metrics['overall']['steps_mean'], episode)
                tb_logger.log_scalar('val/efficiency', eval_metrics['overall']['efficiency_mean'], episode)
                
                # Per grid size validation
                for size, size_metrics in eval_metrics['by_size'].items():
                    tb_logger.log_scalar(f'val_size/coverage_{size}x{size}', 
                                        size_metrics['coverage_mean'], episode)
                
                # Per map type validation
                for map_type, type_metrics in eval_metrics['by_type'].items():
                    tb_logger.log_scalar(f'val_type/{map_type}_coverage', 
                                        type_metrics['coverage_mean'], episode)
            
            # Update best model based on validation coverage
            val_coverage = eval_metrics['overall']['coverage_mean']
            if val_coverage > best_coverage:
                improvement = (val_coverage - best_coverage) * 100 if best_coverage > 0 else val_coverage * 100
                best_coverage = val_coverage
                best_episode = episode
                episodes_without_improvement = 0
                save_path = os.path.join(config.checkpoint_dir,
                                        f"{config.experiment_name}_best.pt")
                agent.save(save_path)
                print(f"✅ New best validation coverage: {best_coverage*100:.1f}% (model saved)")
                if improvement > 0:
                    print(f"   Improvement: +{improvement:.1f}% from previous best")
            else:
                episodes_without_improvement += config.training.eval_frequency
                print(f"   Current best: {best_coverage*100:.1f}% (from episode {best_episode})")
                print(f"   Episodes without improvement: {episodes_without_improvement}/{patience}")

                # Early stopping check (only if enabled and past warmup)
                if episodes_without_improvement >= patience and episode > config.training.warmup_episodes * 2:
                    print(f"\n{'='*70}")
                    print(f"⏹️  Early stopping triggered!")
                    print(f"   No improvement for {episodes_without_improvement} episodes")
                    print(f"   Best coverage: {best_coverage*100:.1f}% at episode {best_episode}")
                    print(f"{'='*70}")
                    break  # Exit training loop
    
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
    parser.add_argument('--curriculum', type=str, default='default',
                       choices=['default', 'fast', 'none'],
                       help='Curriculum type: default (full), fast (shorter), none (disabled)')

    args = parser.parse_args()

    # Create configuration
    config = get_default_config()
    config.experiment_name = args.experiment_name
    config.training.num_episodes = args.episodes
    config.training.multi_scale = args.multi_scale
    config.training.device = args.device
    config.seed = args.seed

    # Train with curriculum
    train(config, curriculum_type=args.curriculum)


if __name__ == "__main__":
    main()
