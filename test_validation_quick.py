"""
Quick Test: Periodic Validation System

Faster version with minimal episodes for quick verification.
"""

import torch
import numpy as np
from dqn_agent import CoordinateDQNAgent
from config import get_default_config
from train import evaluate_agent


def test_validation_quick():
    """Quick test of validation system."""
    
    print("="*70)
    print("Quick Validation System Test")
    print("="*70)
    
    # Create config
    config = get_default_config()
    config.training.multi_scale = True
    config.training.grid_sizes = [15, 20]
    
    # Create agent
    print("\n1. Creating agent...")
    agent = CoordinateDQNAgent(
        input_channels=5,
        num_actions=9,
        hidden_dim=128,
        device='cpu'
    )
    print(f"   ✅ Agent created")
    
    # Run minimal evaluation
    print("\n2. Running validation (1 episode per config)...")
    eval_metrics = evaluate_agent(
        agent=agent,
        config=config,
        num_eval_episodes=1,
        eval_grid_sizes=[15, 20],
        eval_map_types=["empty", "random"]
    )
    
    print("\n3. Checking metrics...")
    
    # Check structure
    assert 'overall' in eval_metrics
    assert 'by_size' in eval_metrics
    assert 'by_type' in eval_metrics
    print("   ✅ Metrics structure correct")
    
    # Check values
    coverage = eval_metrics['overall']['coverage_mean']
    assert 0.0 <= coverage <= 1.0
    print(f"   ✅ Coverage: {coverage*100:.1f}%")
    
    # Check per-size
    assert 15 in eval_metrics['by_size']
    assert 20 in eval_metrics['by_size']
    cov_15 = eval_metrics['by_size'][15]['coverage_mean']
    cov_20 = eval_metrics['by_size'][20]['coverage_mean']
    print(f"   ✅ Coverage 15×15: {cov_15*100:.1f}%")
    print(f"   ✅ Coverage 20×20: {cov_20*100:.1f}%")
    
    # Check per-type
    assert 'empty' in eval_metrics['by_type']
    assert 'random' in eval_metrics['by_type']
    cov_empty = eval_metrics['by_type']['empty']['coverage_mean']
    cov_random = eval_metrics['by_type']['random']['coverage_mean']
    print(f"   ✅ Coverage empty:  {cov_empty*100:.1f}%")
    print(f"   ✅ Coverage random: {cov_random*100:.1f}%")
    
    print("\n" + "="*70)
    print("✅ All validation tests passed!")
    print("="*70)
    print("\nValidation Features:")
    print("  • Greedy policy (epsilon=0)")
    print("  • Per-grid-size metrics")
    print("  • Per-map-type metrics")
    print("  • Overall statistics")
    print("\nIntegration in train.py:")
    print(f"  • Runs every {config.training.eval_frequency} episodes")
    print("  • Logs to TensorBoard (val/*, val_size/*, val_type/*)")
    print("  • Best model saved based on validation coverage")
    print("="*70)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    test_validation_quick()
