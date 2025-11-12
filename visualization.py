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


# ============================================================================
# Path Tracking and Episode Visualization
# ============================================================================

from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import os


@dataclass
class StepRecord:
    """Record of a single step in the episode."""
    step: int
    agent_pos: tuple  # (y, x)
    action: int
    reward: float
    coverage: float
    coverage_map: np.ndarray
    obstacles: np.ndarray
    frontiers: np.ndarray
    visited: np.ndarray


class PathVisualizer:
    """
    Visualize agent paths and coverage heatmaps during episodes.

    Features:
    - Track agent trajectory
    - Generate visit count heatmaps
    - Create summary plots with multiple panels
    - Export visualizations for analysis
    """

    def __init__(
        self,
        grid_size: int = 20,
        save_dir: str = "visualizations",
        track_visit_counts: bool = True
    ):
        """
        Args:
            grid_size: Size of the grid environment
            save_dir: Directory to save visualizations
            track_visit_counts: Whether to track per-cell visit counts
        """
        self.grid_size = grid_size
        self.save_dir = save_dir
        self.track_visit_counts = track_visit_counts

        os.makedirs(save_dir, exist_ok=True)

        # Episode recording
        self.steps: List[StepRecord] = []
        self.visit_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.coverage_over_time = []
        self.reward_over_time = []

        # Action names
        self.action_names = {
            0: "N", 1: "NE", 2: "E", 3: "SE",
            4: "S", 5: "SW", 6: "W", 7: "NW",
            8: "STAY"
        }

        # Color schemes
        self.setup_colormaps()

    def setup_colormaps(self):
        """Setup custom colormaps."""
        self.coverage_cmap = LinearSegmentedColormap.from_list(
            'coverage', ['white', 'lightblue', 'blue', 'darkblue']
        )
        self.visit_cmap = LinearSegmentedColormap.from_list(
            'visits', ['white', 'yellow', 'orange', 'red', 'darkred']
        )
        self.path_cmap = plt.cm.viridis

    def reset(self):
        """Reset recording for new episode."""
        self.steps = []
        self.visit_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.coverage_over_time = []
        self.reward_over_time = []

    def record_step(
        self,
        agent_pos: tuple,
        action: int,
        reward: float,
        coverage_pct: float,
        coverage_map: np.ndarray,
        obstacles: np.ndarray,
        frontiers: np.ndarray,
        visited: np.ndarray,
        step: int
    ):
        """Record a single step during episode."""
        if self.track_visit_counts:
            self.visit_counts[agent_pos[0], agent_pos[1]] += 1

        step_record = StepRecord(
            step=step,
            agent_pos=agent_pos,
            action=action,
            reward=reward,
            coverage=coverage_pct,
            coverage_map=coverage_map.copy(),
            obstacles=obstacles.copy(),
            frontiers=frontiers.copy(),
            visited=visited.copy()
        )

        self.steps.append(step_record)
        self.coverage_over_time.append(coverage_pct)
        self.reward_over_time.append(reward)

    def get_path_coordinates(self) -> np.ndarray:
        """Get agent path as array of (y, x) coordinates."""
        return np.array([step.agent_pos for step in self.steps])

    def save_summary_plot(
        self,
        filename: str,
        steps_to_show: Optional[List[int]] = None
    ):
        """
        Save summary plot showing key moments from episode.

        Args:
            filename: Output filename
            steps_to_show: Step indices to show (default: start, 25%, 50%, 75%, end)
        """
        if len(self.steps) == 0:
            print("No steps recorded to visualize")
            return

        filepath = os.path.join(self.save_dir, filename)

        if steps_to_show is None:
            num_steps = len(self.steps)
            steps_to_show = [
                0,
                num_steps // 4,
                num_steps // 2,
                3 * num_steps // 4,
                num_steps - 1
            ]

        num_panels = len(steps_to_show)
        fig, axes = plt.subplots(1, num_panels, figsize=(5*num_panels, 5))

        if num_panels == 1:
            axes = [axes]

        for idx, step_idx in enumerate(steps_to_show):
            ax = axes[idx]
            step = self.steps[step_idx]

            # Plot environment snapshot
            base_map = np.ones((self.grid_size, self.grid_size, 3))
            base_map[step.obstacles > 0.5] = [0.3, 0.3, 0.3]

            coverage_colored = self.coverage_cmap(step.coverage_map)[:, :, :3]
            alpha = step.coverage_map[:, :, np.newaxis]
            base_map = base_map * (1 - alpha) + coverage_colored * alpha

            ax.imshow(base_map, extent=[0, self.grid_size, self.grid_size, 0])

            # Show path
            path = self.get_path_coordinates()[:step_idx+1]
            if len(path) > 1:
                colors = self.path_cmap(np.linspace(0, 1, len(path)))
                for i in range(len(path) - 1):
                    ax.plot(
                        [path[i, 1] + 0.5, path[i+1, 1] + 0.5],
                        [path[i, 0] + 0.5, path[i+1, 0] + 0.5],
                        color=colors[i],
                        alpha=0.6,
                        linewidth=1.5
                    )

            # Agent position
            ax.add_patch(Circle(
                (step.agent_pos[1] + 0.5, step.agent_pos[0] + 0.5),
                radius=0.4,
                color='red',
                zorder=10
            ))

            ax.set_xlim(0, self.grid_size)
            ax.set_ylim(self.grid_size, 0)
            ax.set_aspect('equal')
            ax.set_title(f'Step {step_idx}\nCoverage: {step.coverage:.1%}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Summary plot saved: {filepath}")

    def save_final_heatmap(self, filename: str):
        """Save final visit count heatmap."""
        filepath = os.path.join(self.save_dir, filename)

        fig, ax = plt.subplots(figsize=(10, 8))

        final_step = self.steps[-1]
        visit_display = self.visit_counts.copy().astype(float)
        visit_display[final_step.obstacles > 0.5] = np.nan

        im = ax.imshow(
            visit_display,
            cmap=self.visit_cmap,
            interpolation='nearest'
        )

        # Overlay path
        path = self.get_path_coordinates()
        if len(path) > 1:
            ax.plot(path[:, 1], path[:, 0], 'b-', alpha=0.3, linewidth=1)
            ax.plot(path[0, 1], path[0, 0], 'go', markersize=10, label='Start')
            ax.plot(path[-1, 1], path[-1, 0], 'ro', markersize=10, label='End')

        ax.set_title(f'Visit Heatmap - {len(self.steps)} steps')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Visits')

        # Statistics
        stats_text = f"""Statistics:
Total Steps: {len(self.steps)}
Unique Cells: {(self.visit_counts > 0).sum()}
Max Revisits: {self.visit_counts.max()}
Avg Revisits: {self.visit_counts[self.visit_counts > 0].mean():.1f}
Final Coverage: {self.coverage_over_time[-1]:.1%}
"""
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Heatmap saved: {filepath}")


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

    # Test 5: Path visualizer
    print("\nTest 5: Path visualizer")
    visualizer = PathVisualizer(grid_size=20, save_dir="test_visualizations")
    visualizer.reset()

    # Simulate episode
    for step in range(50):
        pos = (step // 5, step % 20)
        visualizer.record_step(
            agent_pos=pos,
            action=step % 9,
            reward=0.1,
            coverage_pct=step / 50.0,
            coverage_map=np.random.rand(20, 20),
            obstacles=np.random.rand(20, 20) > 0.9,
            frontiers=np.random.rand(20, 20) > 0.95,
            visited=np.random.rand(20, 20) > 0.5,
            step=step
        )

    visualizer.save_summary_plot("test_summary.png")
    visualizer.save_final_heatmap("test_heatmap.png")
    print("  ✓ Path visualization generated")

    print("\n✓ Visualization test complete!")
    print("\nNote: Close plot windows to continue...")
