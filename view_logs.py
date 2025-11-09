"""
Log Viewer for Training Monitoring

View training logs without TensorBoard.
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Optional
import glob


def load_log_files(log_dir: str = "./logs") -> List[Dict]:
    """
    Load all log entries from JSONL files.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        List of log entries (dicts)
    """
    entries = []
    
    # Find all .jsonl files
    log_files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return entries
    
    # Read all log files
    for log_file in sorted(log_files):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entries.append(entry)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    return entries


def format_metric(value, decimals=2):
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)


def print_summary(entries: List[Dict]):
    """Print training summary statistics."""
    if not entries:
        print("No log entries found.")
        return
    
    print("="*70)
    print("Training Summary")
    print("="*70)
    
    # Basic stats
    print(f"Total Episodes: {len(entries)}")
    
    # Get metrics with numeric values
    numeric_metrics = {}
    for entry in entries:
        for key, value in entry.items():
            if isinstance(value, (int, float)) and key not in ['step', 'episode']:
                if key not in numeric_metrics:
                    numeric_metrics[key] = []
                numeric_metrics[key].append(value)
    
    # Print key statistics
    if 'coverage' in numeric_metrics:
        coverage_values = numeric_metrics['coverage']
        print(f"\nCoverage:")
        print(f"  Current: {coverage_values[-1]*100:.1f}%")
        print(f"  Best:    {max(coverage_values)*100:.1f}%")
        print(f"  Average: {sum(coverage_values)/len(coverage_values)*100:.1f}%")
    
    if 'reward' in numeric_metrics:
        reward_values = numeric_metrics['reward']
        print(f"\nReward:")
        print(f"  Current: {reward_values[-1]:.1f}")
        print(f"  Best:    {max(reward_values):.1f}")
        print(f"  Average: {sum(reward_values)/len(reward_values):.1f}")
    
    if 'epsilon' in numeric_metrics:
        epsilon_values = numeric_metrics['epsilon']
        print(f"\nEpsilon:")
        print(f"  Current: {epsilon_values[-1]:.3f}")
        print(f"  Initial: {epsilon_values[0]:.3f}")
    
    # Training diagnostics
    print(f"\nTraining Diagnostics:")
    for metric in ['loss', 'q_mean', 'td_error', 'grad_norm']:
        if metric in numeric_metrics:
            values = numeric_metrics[metric]
            print(f"  {metric:12s}: {values[-1]:.4f} (avg: {sum(values)/len(values):.4f})")
    
    print("="*70)


def print_recent_episodes(entries: List[Dict], n: int = 20):
    """Print recent episode details."""
    if not entries:
        return
    
    recent = entries[-n:]
    
    print(f"\nRecent {len(recent)} Episodes:")
    print("-"*70)
    print(f"{'Ep':>5} {'Coverage':>8} {'Reward':>8} {'Steps':>6} {'Epsilon':>7} {'Map':>10} {'Grid':>6}")
    print("-"*70)
    
    for entry in recent:
        ep = entry.get('step', entry.get('episode', '?'))
        coverage = entry.get('coverage', 0) * 100
        reward = entry.get('reward', 0)
        steps = entry.get('steps', 0)
        epsilon = entry.get('epsilon', 0)
        map_type = entry.get('map_type', 'N/A')
        grid_size = entry.get('grid_size', 0)
        
        print(f"{ep:5d} {coverage:7.1f}% {reward:8.1f} {steps:6.0f} {epsilon:7.3f} {map_type:>10s} {grid_size:3.0f}x{grid_size:.0f}")
    
    print("-"*70)


def print_curriculum_progress(entries: List[Dict]):
    """Print curriculum learning progress."""
    if not entries:
        return
    
    # Check if curriculum info exists
    last_entry = entries[-1]
    if 'map_type' not in last_entry:
        return
    
    print("\nCurriculum Progress:")
    print("-"*70)
    
    # Aggregate by map type
    map_stats = {}
    for entry in entries:
        map_type = entry.get('map_type', 'unknown')
        if map_type not in map_stats:
            map_stats[map_type] = {'coverage': [], 'reward': [], 'count': 0}
        
        map_stats[map_type]['coverage'].append(entry.get('coverage', 0))
        map_stats[map_type]['reward'].append(entry.get('reward', 0))
        map_stats[map_type]['count'] += 1
    
    # Print stats per map type
    print(f"{'Map Type':>12} {'Episodes':>10} {'Avg Coverage':>14} {'Avg Reward':>12}")
    print("-"*70)
    for map_type, stats in sorted(map_stats.items()):
        avg_cov = sum(stats['coverage']) / len(stats['coverage']) * 100
        avg_rew = sum(stats['reward']) / len(stats['reward'])
        count = stats['count']
        print(f"{map_type:>12} {count:10d} {avg_cov:13.1f}% {avg_rew:12.1f}")
    print("-"*70)


def plot_ascii_chart(values: List[float], width: int = 60, height: int = 15, title: str = "Progress"):
    """Create a simple ASCII chart."""
    if not values or len(values) < 2:
        return
    
    print(f"\n{title}:")
    print("-"*width)
    
    # Normalize values to chart height
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val
    
    if val_range == 0:
        print("(All values are the same)")
        return
    
    # Create chart rows (top to bottom)
    chart = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for i, value in enumerate(values):
        x = int((i / len(values)) * (width - 1))
        y = height - 1 - int(((value - min_val) / val_range) * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            chart[y][x] = '█'
    
    # Print chart
    print(f"{max_val:.2f} ┤", end='')
    for row in chart:
        print(''.join(row))
    print(f"{min_val:.2f} └" + "─" * width)
    print(f"        Episodes: 0{' '*(width-20)}{len(values)}")


def watch_logs(log_dir: str = "./logs", refresh_interval: int = 5):
    """Watch logs in real-time."""
    print("Watching logs (Ctrl+C to stop)...")
    print("="*70)
    
    last_count = 0
    
    try:
        while True:
            entries = load_log_files(log_dir)
            
            # Print new entries
            if len(entries) > last_count:
                new_entries = entries[last_count:]
                print_recent_episodes(new_entries, n=len(new_entries))
                last_count = len(entries)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\nStopped watching logs.")


def main():
    parser = argparse.ArgumentParser(description="View training logs")
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory containing log files')
    parser.add_argument('--recent', type=int, default=20,
                       help='Number of recent episodes to show')
    parser.add_argument('--watch', action='store_true',
                       help='Watch logs in real-time')
    parser.add_argument('--plot', type=str, choices=['coverage', 'reward', 'epsilon', 'loss'],
                       default='coverage',
                       help='Metric to plot')
    
    args = parser.parse_args()
    
    if args.watch:
        watch_logs(args.log_dir)
        return
    
    # Load logs
    entries = load_log_files(args.log_dir)
    
    if not entries:
        print(f"No logs found in {args.log_dir}")
        print("Make sure training has started and logs are being written.")
        return
    
    # Print summary
    print_summary(entries)
    
    # Print recent episodes
    print_recent_episodes(entries, n=args.recent)
    
    # Print curriculum progress
    print_curriculum_progress(entries)
    
    # Plot metric
    metric_values = [e.get(args.plot, 0) for e in entries if args.plot in e]
    if metric_values:
        plot_ascii_chart(metric_values, title=f"{args.plot.capitalize()} Progress")


if __name__ == "__main__":
    main()
