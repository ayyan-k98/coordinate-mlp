"""
Quick test to verify coverage and frontier rewards are working.
"""

import numpy as np
from coverage_env import CoverageEnvironment

print("="*70)
print("Testing Coverage and Frontier Rewards")
print("="*70)

# Create environment
env = CoverageEnvironment(grid_size=20, seed=42)

# Reset
obs = env.reset()
print(f"\nInitial coverage: {env._compute_coverage_percentage():.2%}")
print(f"Initial agent position: ({env.state.agents[0].x}, {env.state.agents[0].y})")

# Take 10 steps and track rewards
print("\n" + "="*70)
print("Step-by-step diagnostics:")
print("="*70)

for i in range(10):
    action = 6 if i < 5 else np.random.randint(0, 9)  # Move right for first 5, then random
    obs, reward, done, info = env.step(action)
    
    rb = info['reward_breakdown']
    print(f"\nStep {i+1}:")
    print(f"  Action: {info.get('action_name', action)}, Moved: {info.get('actually_moved', '?')}")
    print(f"  Agent position: {info.get('agent_position', '?')}")
    print(f"  Coverage: {info['coverage_pct']:.2%} (gain: {info.get('coverage_gain', 0):.4f})")
    print(f"  On frontier: {info.get('is_on_frontier', False)}, Revisit: {info.get('revisits', 0) > (i)}")
    print(f"  Total reward: {reward:.3f}")
    print(f"  Breakdown:")
    print(f"    - coverage:    {rb.get('coverage', 0):.3f}")
    print(f"    - frontier:    {rb.get('frontier', 0):.3f}")
    print(f"    - first_visit: {rb.get('first_visit', 0):.3f}")
    print(f"    - revisit:     {rb.get('revisit', 0):.3f}")
    print(f"    - step:        {rb.get('step', 0):.3f}")
    
    if done:
        print("\n⚠️  Episode done early!")
        break

print("\n" + "="*70)
print(f"Final coverage: {env._compute_coverage_percentage():.2%}")
print("="*70)
