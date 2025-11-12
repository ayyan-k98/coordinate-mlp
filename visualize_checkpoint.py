"""
Visualize Agent Behavior from Checkpoint

Automatically generates path visualizations and heatmaps for all map types
to help debug failure modes and understand agent behavior.

Usage:
    # Visualize best checkpoint on all map types
    python visualize_checkpoint.py --checkpoint checkpoints/fcn_baseline_best.pt

    # Visualize specific map types
    python visualize_checkpoint.py \\
        --checkpoint checkpoints/fcn_baseline_ep350.pt \\
        --map-types corridor cave \\
        --grid-sizes 20 30

    # Create comparison report
    python visualize_checkpoint.py \\
        --checkpoint checkpoints/fcn_baseline_ep350.pt \\
        --num-episodes 10 \\
        --create-report
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dqn_agent import CoordinateDQNAgent
from coverage_env import CoverageEnvironment
from visualization import PathVisualizer
from config import get_default_config


def visualize_single_episode(
    agent: CoordinateDQNAgent,
    grid_size: int,
    map_type: str,
    save_dir: str,
    episode_idx: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Run one episode and create visualizations.

    Args:
        agent: Trained DQN agent
        grid_size: Grid size to test
        map_type: Map type (empty, random, corridor, cave, etc.)
        save_dir: Directory to save visualizations
        episode_idx: Episode index for filename
        verbose: Print progress

    Returns:
        Dictionary with episode metrics
    """
    # Create environment
    config = get_default_config()
    sensor_range = config.environment.get_sensor_range(grid_size)
    max_steps = config.environment.get_max_steps(grid_size)

    env = CoverageEnvironment(
        grid_size=grid_size,
        num_agents=1,
        sensor_range=sensor_range,
        obstacle_density=config.environment.obstacle_density,
        max_steps=max_steps,
        seed=episode_idx,  # Different seed per episode
        reward_config=config.environment.get_reward_config(),
        map_type=map_type
    )

    # Create visualizer
    visualizer = PathVisualizer(
        grid_size=grid_size,
        save_dir=save_dir
    )
    visualizer.reset()

    # Run episode
    state = env.reset()
    done = False
    step = 0
    episode_reward = 0

    while not done:
        # Get agent position
        agent_pos = (env.state.agents[0].y, env.state.agents[0].x)

        # Select action (greedy for evaluation)
        action = agent.select_action(state, epsilon=0.0, agent_pos=agent_pos)

        # Take step
        next_state, reward, done, info = env.step(action)

        # Record for visualization
        visualizer.record_step(
            agent_pos=agent_pos,
            action=action,
            reward=reward,
            coverage_pct=info['coverage_pct'],
            coverage_map=env.state.coverage,
            obstacles=env.state.obstacles,
            frontiers=env.state.frontiers,
            visited=env.state.visited,
            step=step
        )

        state = next_state
        episode_reward += reward
        step += 1

    # Generate visualizations
    visualizer.save_summary_plot(f"episode_{episode_idx}_summary.png")
    visualizer.save_final_heatmap(f"episode_{episode_idx}_heatmap.png")

    # Collect metrics
    metrics = {
        'grid_size': grid_size,
        'map_type': map_type,
        'episode': episode_idx,
        'coverage': info['coverage_pct'],
        'reward': episode_reward,
        'steps': step,
        'unique_cells_visited': int((visualizer.visit_counts > 0).sum()),
        'max_revisits': int(visualizer.visit_counts.max()),
        'avg_revisits': float(visualizer.visit_counts[visualizer.visit_counts > 0].mean()),
        'collisions': info.get('collisions', 0),
        'save_dir': save_dir
    }

    if verbose:
        print(f"  Episode {episode_idx}: {metrics['coverage']:.1%} coverage, "
              f"{metrics['steps']} steps, {metrics['unique_cells_visited']} cells visited")

    return metrics


def create_comparison_grid(
    all_results: List[Dict],
    save_path: str
):
    """
    Create comparison grid showing performance across map types and grid sizes.

    Args:
        all_results: List of result dictionaries
        save_path: Path to save comparison figure
    """
    # Organize results by map type and grid size
    map_types = sorted(set(r['map_type'] for r in all_results))
    grid_sizes = sorted(set(r['grid_size'] for r in all_results))

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Coverage heatmap (map type × grid size)
    ax1 = fig.add_subplot(gs[0, 0])
    coverage_matrix = np.zeros((len(map_types), len(grid_sizes)))

    for i, map_type in enumerate(map_types):
        for j, grid_size in enumerate(grid_sizes):
            results = [r for r in all_results
                      if r['map_type'] == map_type and r['grid_size'] == grid_size]
            if results:
                coverage_matrix[i, j] = np.mean([r['coverage'] for r in results])

    im1 = ax1.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(grid_sizes)))
    ax1.set_yticks(range(len(map_types)))
    ax1.set_xticklabels([f"{s}×{s}" for s in grid_sizes])
    ax1.set_yticklabels(map_types)
    ax1.set_title('Coverage % by Map Type and Grid Size')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Map Type')

    # Add text annotations
    for i in range(len(map_types)):
        for j in range(len(grid_sizes)):
            text = ax1.text(j, i, f'{coverage_matrix[i, j]:.1%}',
                           ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im1, ax=ax1, label='Coverage %')

    # Plot 2: Steps distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for map_type in map_types:
        results = [r for r in all_results if r['map_type'] == map_type]
        steps = [r['steps'] for r in results]
        ax2.scatter([map_type] * len(steps), steps, alpha=0.5, s=50)

    ax2.set_xlabel('Map Type')
    ax2.set_ylabel('Steps to Completion')
    ax2.set_title('Steps Distribution by Map Type')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Coverage vs Steps efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    for map_type in map_types:
        results = [r for r in all_results if r['map_type'] == map_type]
        if results:
            coverages = [r['coverage'] for r in results]
            steps = [r['steps'] for r in results]
            ax3.scatter(steps, coverages, label=map_type, alpha=0.6, s=60)

    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Coverage %')
    ax3.set_title('Coverage Efficiency (Coverage vs Steps)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Revisit statistics
    ax4 = fig.add_subplot(gs[1, 1])
    map_type_labels = []
    avg_revisits = []
    max_revisits = []

    for map_type in map_types:
        results = [r for r in all_results if r['map_type'] == map_type]
        if results:
            map_type_labels.append(map_type)
            avg_revisits.append(np.mean([r['avg_revisits'] for r in results]))
            max_revisits.append(np.mean([r['max_revisits'] for r in results]))

    x = np.arange(len(map_type_labels))
    width = 0.35

    ax4.bar(x - width/2, avg_revisits, width, label='Avg Revisits', alpha=0.8)
    ax4.bar(x + width/2, max_revisits, width, label='Max Revisits', alpha=0.8)

    ax4.set_xlabel('Map Type')
    ax4.set_ylabel('Number of Revisits')
    ax4.set_title('Revisit Statistics (Indicator of Looping)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(map_type_labels, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Per-map-type performance summary
    ax5 = fig.add_subplot(gs[2, :])

    summary_data = []
    for map_type in map_types:
        results = [r for r in all_results if r['map_type'] == map_type]
        if results:
            summary_data.append({
                'map_type': map_type,
                'coverage_mean': np.mean([r['coverage'] for r in results]),
                'coverage_std': np.std([r['coverage'] for r in results]),
                'steps_mean': np.mean([r['steps'] for r in results]),
                'num_episodes': len(results)
            })

    # Bar plot with error bars
    x = np.arange(len(summary_data))
    coverages = [d['coverage_mean'] for d in summary_data]
    errors = [d['coverage_std'] for d in summary_data]
    labels = [d['map_type'] for d in summary_data]

    bars = ax5.bar(x, coverages, yerr=errors, capsize=5, alpha=0.7)

    # Color bars by performance
    for i, (bar, cov) in enumerate(zip(bars, coverages)):
        if cov > 0.8:
            bar.set_color('green')
        elif cov > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax5.set_xlabel('Map Type')
    ax5.set_ylabel('Coverage %')
    ax5.set_title('Overall Performance Summary (mean ± std)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_ylim(0, 1.0)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add performance annotations
    for i, d in enumerate(summary_data):
        ax5.text(i, d['coverage_mean'] + d['coverage_std'] + 0.05,
                f"{d['coverage_mean']:.1%}\n({d['num_episodes']} eps)",
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Comparison grid saved: {save_path}")


def generate_report(
    all_results: List[Dict],
    checkpoint_path: str,
    save_dir: str
) -> str:
    """
    Generate text report summarizing results.

    Args:
        all_results: List of result dictionaries
        checkpoint_path: Path to checkpoint
        save_dir: Directory to save report

    Returns:
        Path to saved report
    """
    report_path = os.path.join(save_dir, "evaluation_report.txt")

    # Organize results
    map_types = sorted(set(r['map_type'] for r in all_results))
    grid_sizes = sorted(set(r['grid_size'] for r in all_results))

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHECKPOINT EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Total Episodes: {len(all_results)}\n")
        f.write(f"Map Types: {', '.join(map_types)}\n")
        f.write(f"Grid Sizes: {', '.join(str(s) for s in grid_sizes)}\n\n")

        # Overall statistics
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")

        overall_coverage = np.mean([r['coverage'] for r in all_results])
        overall_steps = np.mean([r['steps'] for r in all_results])

        f.write(f"Average Coverage: {overall_coverage:.1%} ± {np.std([r['coverage'] for r in all_results]):.1%}\n")
        f.write(f"Average Steps: {overall_steps:.1f} ± {np.std([r['steps'] for r in all_results]):.1f}\n")
        f.write(f"Average Reward: {np.mean([r['reward'] for r in all_results]):.2f}\n\n")

        # Per-map-type breakdown
        f.write("="*80 + "\n")
        f.write("PER-MAP-TYPE BREAKDOWN\n")
        f.write("="*80 + "\n\n")

        for map_type in map_types:
            results = [r for r in all_results if r['map_type'] == map_type]

            f.write(f"\n{map_type.upper()}:\n")
            f.write("-"*40 + "\n")

            coverages = [r['coverage'] for r in results]
            steps_list = [r['steps'] for r in results]

            f.write(f"  Episodes: {len(results)}\n")
            f.write(f"  Coverage: {np.mean(coverages):.1%} ± {np.std(coverages):.1%}\n")
            f.write(f"  Steps: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}\n")
            f.write(f"  Unique Cells: {np.mean([r['unique_cells_visited'] for r in results]):.1f}\n")
            f.write(f"  Max Revisits: {np.mean([r['max_revisits'] for r in results]):.1f}\n")
            f.write(f"  Avg Revisits: {np.mean([r['avg_revisits'] for r in results]):.2f}\n")

            # Per grid size within map type
            for grid_size in grid_sizes:
                size_results = [r for r in results if r['grid_size'] == grid_size]
                if size_results:
                    size_cov = np.mean([r['coverage'] for r in size_results])
                    f.write(f"    {grid_size}×{grid_size}: {size_cov:.1%} coverage\n")

        # Best and worst episodes
        f.write("\n" + "="*80 + "\n")
        f.write("BEST EPISODES\n")
        f.write("="*80 + "\n\n")

        best_episodes = sorted(all_results, key=lambda x: x['coverage'], reverse=True)[:5]
        for i, ep in enumerate(best_episodes, 1):
            f.write(f"{i}. {ep['map_type']:10s} {ep['grid_size']}×{ep['grid_size']}: "
                   f"{ep['coverage']:.1%} coverage in {ep['steps']} steps\n")
            f.write(f"   Path: {ep['save_dir']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("WORST EPISODES\n")
        f.write("="*80 + "\n\n")

        worst_episodes = sorted(all_results, key=lambda x: x['coverage'])[:5]
        for i, ep in enumerate(worst_episodes, 1):
            f.write(f"{i}. {ep['map_type']:10s} {ep['grid_size']}×{ep['grid_size']}: "
                   f"{ep['coverage']:.1%} coverage in {ep['steps']} steps\n")
            f.write(f"   Path: {ep['save_dir']}\n")

        # Failure mode analysis
        f.write("\n" + "="*80 + "\n")
        f.write("FAILURE MODE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # High revisit episodes (looping)
        looping_episodes = [r for r in all_results if r['max_revisits'] > 20]
        if looping_episodes:
            f.write(f"⚠️  LOOPING DETECTED ({len(looping_episodes)} episodes):\n")
            for ep in sorted(looping_episodes, key=lambda x: x['max_revisits'], reverse=True)[:3]:
                f.write(f"  - {ep['map_type']} {ep['grid_size']}×{ep['grid_size']}: "
                       f"{ep['max_revisits']} max revisits\n")
        else:
            f.write("✅ No significant looping detected\n")

        f.write("\n")

        # Low coverage episodes
        low_coverage_episodes = [r for r in all_results if r['coverage'] < 0.5]
        if low_coverage_episodes:
            f.write(f"⚠️  LOW COVERAGE ({len(low_coverage_episodes)} episodes < 50%):\n")
            for ep in sorted(low_coverage_episodes, key=lambda x: x['coverage'])[:3]:
                f.write(f"  - {ep['map_type']} {ep['grid_size']}×{ep['grid_size']}: "
                       f"{ep['coverage']:.1%} coverage\n")
        else:
            f.write("✅ All episodes achieved > 50% coverage\n")

        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")

        # Generate recommendations based on results
        recommendations = []

        # Check corridor performance
        corridor_results = [r for r in all_results if r['map_type'] == 'corridor']
        if corridor_results:
            corridor_cov = np.mean([r['coverage'] for r in corridor_results])
            if corridor_cov < 0.5:
                recommendations.append(
                    f"⚠️  Corridor coverage ({corridor_cov:.1%}) is low. "
                    f"Consider:\n"
                    f"   - Increasing frontier bonus\n"
                    f"   - Adding exploration bonus for new cells\n"
                    f"   - Reducing revisit penalty initially"
                )

        # Check for looping
        avg_max_revisits = np.mean([r['max_revisits'] for r in all_results])
        if avg_max_revisits > 15:
            recommendations.append(
                f"⚠️  High revisit count ({avg_max_revisits:.1f} avg). "
                f"Consider:\n"
                f"   - Increasing revisit penalty\n"
                f"   - Implementing count-based exploration bonus"
            )

        # Check overall performance
        if overall_coverage < 0.6:
            recommendations.append(
                f"⚠️  Overall coverage ({overall_coverage:.1%}) is low. "
                f"Consider:\n"
                f"   - Reviewing reward structure\n"
                f"   - Applying fixes from TRAINING_FIXES.md\n"
                f"   - Training for more episodes"
            )
        elif overall_coverage > 0.8:
            recommendations.append(
                f"✅ Overall coverage ({overall_coverage:.1%}) is good!"
            )

        if recommendations:
            for rec in recommendations:
                f.write(rec + "\n\n")
        else:
            f.write("✅ Agent is performing well across all tested scenarios!\n")

    print(f"✅ Report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize agent behavior from checkpoint"
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--map-types', nargs='+',
                       default=['empty', 'random', 'corridor', 'cave'],
                       help='Map types to visualize')
    parser.add_argument('--grid-sizes', nargs='+', type=int,
                       default=[20, 30],
                       help='Grid sizes to test')
    parser.add_argument('--num-episodes', type=int, default=3,
                       help='Number of episodes per configuration')
    parser.add_argument('--save-dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--create-report', action='store_true',
                       help='Create comparison report')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Load checkpoint
    print("="*80)
    print("LOADING CHECKPOINT")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}\n")

    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=6,  # POMDP with visibility mask
        num_actions=9,
        use_pomdp=True,
        sensor_range=4.0,
        device=args.device
    )

    # Load weights (PyTorch 2.6+ compatible)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if 'policy_net' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
    elif 'policy_net_state_dict' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    else:
        raise KeyError("Checkpoint must contain 'policy_net' or 'policy_net_state_dict'")
    
    agent.policy_net.eval()

    print(f"✅ Checkpoint loaded successfully\n")

    # Run evaluations
    print("="*80)
    print("RUNNING EVALUATIONS")
    print("="*80)
    print(f"Map types: {', '.join(args.map_types)}")
    print(f"Grid sizes: {', '.join(str(s) for s in args.grid_sizes)}")
    print(f"Episodes per config: {args.num_episodes}\n")

    all_results = []

    for grid_size in args.grid_sizes:
        for map_type in args.map_types:
            print(f"\n{'-'*80}")
            print(f"Testing: {map_type} on {grid_size}×{grid_size} grid")
            print(f"{'-'*80}")

            for episode_idx in range(args.num_episodes):
                # Create save directory
                save_dir = os.path.join(
                    args.save_dir,
                    f"{grid_size}x{grid_size}",
                    map_type,
                    f"episode_{episode_idx}"
                )
                os.makedirs(save_dir, exist_ok=True)

                # Run episode
                metrics = visualize_single_episode(
                    agent=agent,
                    grid_size=grid_size,
                    map_type=map_type,
                    save_dir=save_dir,
                    episode_idx=episode_idx,
                    verbose=True
                )

                all_results.append(metrics)

            # Print summary for this configuration
            config_results = [r for r in all_results
                            if r['grid_size'] == grid_size and r['map_type'] == map_type]
            avg_coverage = np.mean([r['coverage'] for r in config_results])
            print(f"\nAverage coverage: {avg_coverage:.1%}")

    # Save results to JSON
    results_path = os.path.join(args.save_dir, "all_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved: {results_path}")

    # Create comparison report
    if args.create_report or len(all_results) > 5:
        print("\n" + "="*80)
        print("CREATING COMPARISON REPORT")
        print("="*80)

        # Comparison grid
        comparison_path = os.path.join(args.save_dir, "comparison_grid.png")
        create_comparison_grid(all_results, comparison_path)

        # Text report
        report_path = generate_report(
            all_results,
            args.checkpoint,
            args.save_dir
        )

        print(f"\n📊 Comparison visualizations created!")
        print(f"   Grid: {comparison_path}")
        print(f"   Report: {report_path}")

    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    overall_coverage = np.mean([r['coverage'] for r in all_results])
    print(f"\nOverall Coverage: {overall_coverage:.1%}")
    print(f"Total Episodes: {len(all_results)}")
    print(f"Visualizations: {args.save_dir}/")

    # Quick summary by map type
    print("\nPer-Map-Type Summary:")
    for map_type in args.map_types:
        results = [r for r in all_results if r['map_type'] == map_type]
        if results:
            cov = np.mean([r['coverage'] for r in results])
            print(f"  {map_type:10s}: {cov:.1%} coverage")

    print("\n✅ Done! Check visualizations in:", args.save_dir)


if __name__ == "__main__":
    main()
