"""
Metrics Computation

Calculate coverage and performance metrics.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class CoverageMetrics:
    """Container for coverage metrics."""
    coverage_percentage: float
    num_covered_cells: int
    total_cells: int
    num_steps: int
    collision_count: int
    revisit_count: int
    efficiency: float  # Coverage per step
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'coverage_pct': self.coverage_percentage,
            'num_covered': self.num_covered_cells,
            'total_cells': self.total_cells,
            'num_steps': self.num_steps,
            'collisions': self.collision_count,
            'revisits': self.revisit_count,
            'efficiency': self.efficiency,
        }


def compute_metrics(
    coverage_map: np.ndarray,
    obstacle_map: np.ndarray,
    trajectory: List[tuple],
    max_steps: int
) -> CoverageMetrics:
    """
    Compute comprehensive coverage metrics.
    
    Args:
        coverage_map: Boolean array [H, W] of covered cells
        obstacle_map: Boolean array [H, W] of obstacles
        trajectory: List of (x, y) positions visited
        max_steps: Maximum steps allowed
    
    Returns:
        CoverageMetrics object
    """
    # Count coverable cells (exclude obstacles)
    total_cells = (~obstacle_map).sum()
    covered_cells = (coverage_map & ~obstacle_map).sum()
    coverage_pct = covered_cells / total_cells if total_cells > 0 else 0.0
    
    # Count collisions and revisits
    collision_count = 0
    revisit_count = 0
    visited_positions = set()
    
    for pos in trajectory:
        x, y = pos
        
        # Check if valid position
        if 0 <= x < obstacle_map.shape[1] and 0 <= y < obstacle_map.shape[0]:
            # Check collision
            if obstacle_map[y, x]:
                collision_count += 1
            
            # Check revisit
            if pos in visited_positions:
                revisit_count += 1
            else:
                visited_positions.add(pos)
    
    # Efficiency
    num_steps = len(trajectory)
    efficiency = coverage_pct / num_steps if num_steps > 0 else 0.0
    
    return CoverageMetrics(
        coverage_percentage=coverage_pct,
        num_covered_cells=int(covered_cells),
        total_cells=int(total_cells),
        num_steps=num_steps,
        collision_count=collision_count,
        revisit_count=revisit_count,
        efficiency=efficiency
    )


def aggregate_metrics(metrics_list: List[CoverageMetrics]) -> Dict[str, tuple]:
    """
    Aggregate metrics from multiple episodes.
    
    Args:
        metrics_list: List of CoverageMetrics objects
    
    Returns:
        Dictionary of metric_name -> (mean, std)
    """
    if not metrics_list:
        return {}
    
    # Extract values
    coverage_pcts = [m.coverage_percentage for m in metrics_list]
    efficiencies = [m.efficiency for m in metrics_list]
    num_steps = [m.num_steps for m in metrics_list]
    collisions = [m.collision_count for m in metrics_list]
    revisits = [m.revisit_count for m in metrics_list]
    
    return {
        'coverage': (np.mean(coverage_pcts), np.std(coverage_pcts)),
        'efficiency': (np.mean(efficiencies), np.std(efficiencies)),
        'steps': (np.mean(num_steps), np.std(num_steps)),
        'collisions': (np.mean(collisions), np.std(collisions)),
        'revisits': (np.mean(revisits), np.std(revisits)),
    }


def compute_grid_size_degradation(
    baseline_coverage: float,
    test_coverages: Dict[int, float]
) -> Dict[int, float]:
    """
    Compute performance degradation across grid sizes.
    
    Args:
        baseline_coverage: Coverage on baseline grid size (e.g., 20×20)
        test_coverages: Dictionary mapping grid_size -> coverage
    
    Returns:
        Dictionary mapping grid_size -> degradation_percentage
    """
    degradation = {}
    
    for size, coverage in test_coverages.items():
        if baseline_coverage > 0:
            deg = (coverage - baseline_coverage) / baseline_coverage * 100
            degradation[size] = deg
        else:
            degradation[size] = 0.0
    
    return degradation


if __name__ == "__main__":
    print("="*60)
    print("Testing Metrics")
    print("="*60)
    
    # Create sample data
    H, W = 20, 20
    
    # Coverage map (some cells covered)
    coverage_map = np.random.rand(H, W) > 0.5
    
    # Obstacle map
    obstacle_map = np.random.rand(H, W) > 0.85
    
    # Trajectory with some revisits and collisions
    trajectory = []
    for _ in range(50):
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        trajectory.append((x, y))
    
    # Add some revisits
    trajectory.extend(trajectory[:5])
    
    # Compute metrics
    metrics = compute_metrics(coverage_map, obstacle_map, trajectory, max_steps=100)
    
    print(f"\nSingle episode metrics:")
    print(f"  Coverage: {metrics.coverage_percentage*100:.1f}%")
    print(f"  Covered cells: {metrics.num_covered_cells}/{metrics.total_cells}")
    print(f"  Steps: {metrics.num_steps}")
    print(f"  Collisions: {metrics.collision_count}")
    print(f"  Revisits: {metrics.revisit_count}")
    print(f"  Efficiency: {metrics.efficiency:.4f}")
    
    # Test aggregation
    metrics_list = []
    for _ in range(10):
        coverage_map = np.random.rand(H, W) > 0.5
        traj = [(np.random.randint(0, W), np.random.randint(0, H)) for _ in range(50)]
        m = compute_metrics(coverage_map, obstacle_map, traj, max_steps=100)
        metrics_list.append(m)
    
    aggregated = aggregate_metrics(metrics_list)
    print(f"\nAggregated metrics (10 episodes):")
    for metric_name, (mean, std) in aggregated.items():
        print(f"  {metric_name:12s}: {mean:.4f} ± {std:.4f}")
    
    # Test degradation
    baseline = 0.45
    test_coverages = {
        20: 0.45,
        25: 0.42,
        30: 0.38,
        35: 0.35,
        40: 0.30
    }
    
    degradation = compute_grid_size_degradation(baseline, test_coverages)
    print(f"\nGrid size degradation:")
    for size, deg in degradation.items():
        print(f"  {size}×{size}: {deg:+6.1f}%")
    
    print("\n✓ Metrics test complete!")
