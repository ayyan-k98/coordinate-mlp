"""
Quick Environment Test - No Unicode
Tests basic environment functionality
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from coverage_env import CoverageEnvironment, Action

print("="*70)
print("ENVIRONMENT QUICK TEST")
print("="*70)

# Test 1: Create environment
print("\n[Test 1] Creating environment...")
env = CoverageEnvironment(grid_size=15, num_agents=1, sensor_range=3.0, seed=42)
obs = env.reset()
print(f"  OK: Observation shape = {obs.shape}")
print(f"  Initial coverage: {env.episode_stats['coverage_percentage']*100:.1f}%")
print(f"  Frontiers: {len(env.state.frontiers)}")

# Test 2: Take actions
print("\n[Test 2] Taking 20 actions...")
rewards = []
for i in range(20):
    valid_actions = env.get_valid_actions(agent_id=0)
    action = np.random.choice(np.where(valid_actions)[0])
    obs, reward, done, info = env.step(action, agent_id=0)
    rewards.append(reward)
    
print(f"  OK: Completed 20 steps")
print(f"  Average reward: {np.mean(rewards):.2f}")
print(f"  Current coverage: {info['coverage_pct']*100:.1f}%")
print(f"  Collisions: {info['collisions']}, Revisits: {info['revisits']}")

# Test 3: Full episode
print("\n[Test 3] Running full episode...")
env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions(agent_id=0)
    action = np.random.choice(np.where(valid_actions)[0])
    obs, reward, done, info = env.step(action, agent_id=0)

print(f"  OK: Episode completed in {info['steps']} steps")
print(f"  Final coverage: {info['coverage_pct']*100:.1f}%")
print(f"  Total reward: {env.episode_stats['total_reward']:.2f}")

# Test 4: Reward breakdown
print("\n[Test 4] Reward breakdown...")
env.reset()
obs, reward, done, info = env.step(1, agent_id=0)  # Move north
breakdown = info['reward_breakdown']
print(f"  Total reward: {reward:.2f}")
for component, value in breakdown.items():
    if value != 0:
        print(f"    {component}: {value:.2f}")

# Test 5: Action space
print("\n[Test 5] Action space...")
print(f"  Total actions: {len(Action)}")
for action in list(Action)[:4]:  # Show first 4
    print(f"    {action.value}: {action.name}")

# Test 6: Sensor model
print("\n[Test 6] Sensor model...")
from coverage_env import ProbabilisticSensorModel
sensor = ProbabilisticSensorModel(max_range=4.0)
for dist in [0.0, 1.0, 2.0, 3.0, 4.0]:
    prob = sensor.get_detection_probability(dist)
    print(f"  Distance {dist:.1f}: P(detect) = {prob:.3f}")

# Test 7: Scale invariance
print("\n[Test 7] Multiple grid sizes...")
for size in [15, 20, 25, 30]:
    env = CoverageEnvironment(grid_size=size, sensor_range=size*0.2, seed=42)
    obs = env.reset()
    initial_cov = env.episode_stats['coverage_percentage']
    print(f"  {size}x{size}: obs={obs.shape}, initial_coverage={initial_cov*100:.1f}%")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
