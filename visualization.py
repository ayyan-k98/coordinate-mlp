"""
Visualization Utilities

Functions for visualizing attention maps, training curves, and coverage heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import torch


def visualize_attention(
    attention_weights: torch.Tensor,
    grid_size: tuple,
    save_path: Optional[str] = None,
    title: str = "Attention Map"
):
    """
    Visualize attention weights as heatmap.
    
    Args:
        attention_weights: [num_heads, 1, H*W] or [1, H*W]
        grid_size: (H, W) tuple
        save_path: Path to save figure (optional)
        title: Figure title
    """
    H, W = grid_size
    
    # Handle different input shapes
    if attention_weights.dim() == 3:
        # Average over attention heads
        attn = attention_weights.mean(dim=0).squeeze()  # [H*W]
    else:
        attn = attention_weights.squeeze()
    
    # Reshape to grid
    attn = attn.cpu().numpy().reshape(H, W)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    metrics_history: List[Dict[str, float]],
    metrics_to_plot: List[str] = ['reward', 'coverage', 'loss'],
    save_path: Optional[str] = None
):
    """
    Plot training curves.
    
    Args:
        metrics_history: List of metric dictionaries
        metrics_to_plot: List of metric names to plot
        save_path: Path to save figure (optional)
    """
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3*num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, metrics_to_plot):
        # Extract metric values
        values = [m.get(metric_name, np.nan) for m in metrics_history]
        episodes = list(range(len(values)))
        
        # Plot
        ax.plot(episodes, values, linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()} over Training')
        ax.grid(True, alpha=0.3)
        
        # Add moving average
        if len(values) > 10:
            window = min(50, len(values) // 10)
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(values)), moving_avg, 
                   color='red', linewidth=2, alpha=0.7, label=f'MA({window})')
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_coverage_heatmap(
    coverage_map: np.ndarray,
    obstacle_map: Optional[np.ndarray] = None,
    trajectory: Optional[List[tuple]] = None,
    save_path: Optional[str] = None,
    title: str = "Coverage Map"
):
    """
    Plot coverage heatmap with obstacles and trajectory.
    
    Args:
        coverage_map: Boolean array [H, W] of covered cells
        obstacle_map: Boolean array [H, W] of obstacles (optional)
        trajectory: List of (x, y) positions (optional)
        save_path: Path to save figure (optional)
        title: Figure title
    """
    H, W = coverage_map.shape
    
    # Create RGB image
    img = np.zeros((H, W, 3))
    
    # Covered cells (green)
    img[coverage_map] = [0, 1, 0]
    
    # Obstacles (black)
    if obstacle_map is not None:
        img[obstacle_map] = [0, 0, 0]
    
    # Uncovered cells (white)
    uncovered = ~coverage_map
    if obstacle_map is not None:
        uncovered = uncovered & ~obstacle_map
    img[uncovered] = [1, 1, 1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(img, origin='lower')
    
    # Plot trajectory
    if trajectory:
        x_coords = [pos[0] for pos in trajectory]
        y_coords = [pos[1] for pos in trajectory]
        plt.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.5, label='Trajectory')
        plt.plot(x_coords[0], y_coords[0], 'ro', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'r^', markersize=10, label='End')
        plt.legend()
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_grid_size_comparison(
    results: Dict[int, Dict[str, float]],
    baseline_size: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot performance comparison across grid sizes.
    
    Args:
        results: Dictionary mapping grid_size -> metrics_dict
        baseline_size: Baseline grid size for comparison
        save_path: Path to save figure (optional)
    """
    sizes = sorted(results.keys())
    
    # Extract coverage values
    coverage_means = [results[s]['coverage_mean'] for s in sizes]
    coverage_stds = [results[s]['coverage_std'] for s in sizes]
    
    # Compute degradation
    baseline_coverage = results[baseline_size]['coverage_mean']
    degradation = [(cov - baseline_coverage) / baseline_coverage * 100 
                   for cov in coverage_means]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute coverage
    ax1.errorbar(sizes, coverage_means, yerr=coverage_stds, 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.axhline(baseline_coverage, color='r', linestyle='--', 
               label=f'Baseline ({baseline_size}×{baseline_size})')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Coverage vs Grid Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Relative degradation
    ax2.bar(range(len(sizes)), degradation, tick_label=sizes)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Degradation (%)')
    ax2.set_title('Performance Degradation')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Color bars based on degradation
    colors = ['green' if d > -10 else 'orange' if d > -20 else 'red' 
             for d in degradation]
    for i, (bar, color) in enumerate(zip(ax2.patches, colors)):
        bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Testing Visualization")
    print("="*60)
    
    # Test 1: Attention visualization
    print("\nTest 1: Attention visualization")
    H, W = 20, 20
    attention = torch.randn(4, 1, H*W).softmax(dim=-1)  # 4 heads
    visualize_attention(attention, (H, W), title="Test Attention Map")
    print("  ✓ Attention map generated")
    
    # Test 2: Training curves
    print("\nTest 2: Training curves")
    metrics_history = []
    for i in range(100):
        metrics_history.append({
            'reward': 50 + 10 * np.sin(i/10) + np.random.randn() * 5,
            'coverage': 0.3 + 0.1 * (i/100) + np.random.randn() * 0.05,
            'loss': 1.0 - 0.5 * (i/100) + np.random.randn() * 0.1
        })
    plot_training_curves(metrics_history)
    print("  ✓ Training curves generated")
    
    # Test 3: Coverage heatmap
    print("\nTest 3: Coverage heatmap")
    coverage_map = np.random.rand(H, W) > 0.5
    obstacle_map = np.random.rand(H, W) > 0.85
    trajectory = [(np.random.randint(0, W), np.random.randint(0, H)) 
                  for _ in range(50)]
    plot_coverage_heatmap(coverage_map, obstacle_map, trajectory)
    print("  ✓ Coverage heatmap generated")
    
    # Test 4: Grid size comparison
    print("\nTest 4: Grid size comparison")
    results = {
        20: {'coverage_mean': 0.45, 'coverage_std': 0.05},
        25: {'coverage_mean': 0.42, 'coverage_std': 0.06},
        30: {'coverage_mean': 0.38, 'coverage_std': 0.07},
        35: {'coverage_mean': 0.35, 'coverage_std': 0.08},
        40: {'coverage_mean': 0.30, 'coverage_std': 0.09},
    }
    plot_grid_size_comparison(results, baseline_size=20)
    print("  ✓ Grid size comparison generated")
    
    print("\n✓ Visualization test complete!")
    print("\nNote: Close plot windows to continue...")
