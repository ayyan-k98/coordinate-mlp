"""
Diagnostic script to check actual reward configuration and behavior.
"""

import numpy as np
from coverage_env import CoverageEnvironment, RewardFunction

print("="*60)
print("REWARD CONFIGURATION DIAGNOSTIC")
print("="*60)

# Create environment
env = CoverageEnvironment(grid_size=20, seed=42)

# Check reward function
rf = env.reward_fn

print("\nReward Function Configuration:")
print(f"  coverage_reward: {rf.coverage_reward}")
print(f"  first_visit_bonus: {rf.first_visit_bonus}")
print(f"  collision_penalty: {rf.collision_penalty}")
print(f"  step_penalty: {rf.step_penalty}")
print(f"  frontier_bonus: {rf.frontier_bonus}")
print(f"  revisit_penalty: {rf.revisit_penalty}")
print(f"  confidence_weight: {rf.confidence_weight}")

# Run one episode and track rewards
obs = env.reset()
total_reward = 0
coverage_rewards = []
first_visit_rewards = []
frontier_rewards = []
step_count = 0

print("\n" + "="*60)
print("RUNNING 100-STEP TEST EPISODE")
print("="*60)

for step in range(100):
    action = np.random.randint(0, 9)  # Random action
    obs, reward, done, info = env.step(action)
    
    total_reward += reward
    breakdown = info['reward_breakdown']
    coverage_rewards.append(breakdown.get('coverage', 0))
    first_visit_rewards.append(breakdown.get('first_visit', 0))
    frontier_rewards.append(breakdown.get('frontier', 0))
    step_count += 1
    
    # Print first 5 steps
    if step < 5:
        print(f"\nStep {step+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Breakdown: coverage={breakdown.get('coverage', 0):.3f}, "
              f"first_visit={breakdown.get('first_visit', 0):.3f}, "
              f"frontier={breakdown.get('frontier', 0):.3f}")
    
    if done:
        break

print(f"\n{'='*60}")
print(f"EPISODE SUMMARY ({step_count} steps)")
print(f"{'='*60}")
print(f"Total reward: {total_reward:.2f}")
print(f"Coverage reward sum: {sum(coverage_rewards):.2f}")
print(f"First visit reward sum: {sum(first_visit_rewards):.2f}")
print(f"Frontier reward sum: {sum(frontier_rewards):.2f}")
print(f"Average reward/step: {total_reward/step_count:.3f}")

print(f"\n{'='*60}")
print(f"EXTRAPOLATION TO FULL EPISODE")
print(f"{'='*60}")
estimated_300_step = (total_reward / step_count) * 300
print(f"Estimated 300-step total: {estimated_300_step:.1f}")

print(f"\n{'='*60}")
print(f"DIAGNOSIS")
print(f"{'='*60}")
if estimated_300_step > 500:
    print(f"⚠️  REWARDS ARE BROKEN - Episode return is TOO HIGH")
    print(f"   Expected: 50-150, Got: {estimated_300_step:.1f}")
    print(f"   Action: Find where RewardFunction is created without config")
elif estimated_300_step < 30:
    print(f"⚠️  REWARDS MAY BE TOO LOW")
    print(f"   Expected: 50-150, Got: {estimated_300_step:.1f}")
else:
    print(f"✅ REWARDS ARE CORRECT")
    print(f"   Expected: 50-150, Got: {estimated_300_step:.1f}")
    print(f"   If you still see gradient explosions, it's architectural")

print(f"\n{'='*60}")
print(f"PER-STEP REWARD BREAKDOWN")
print(f"{'='*60}")
print(f"Average coverage reward/step: {sum(coverage_rewards)/step_count:.3f}")
print(f"Average first_visit reward/step: {sum(first_visit_rewards)/step_count:.3f}")
print(f"Average frontier reward/step: {sum(frontier_rewards)/step_count:.3f}")

# Check if rewards match config
print(f"\n{'='*60}")
print(f"CONFIGURATION VERIFICATION")
print(f"{'='*60}")
print(f"Expected coverage_reward scale: 0.5")
print(f"Actual: {rf.coverage_reward}")
print(f"Match: {'✅' if abs(rf.coverage_reward - 0.5) < 0.01 else '❌'}")

print(f"\nExpected first_visit_bonus: 0.5")
print(f"Actual: {rf.first_visit_bonus}")
print(f"Match: {'✅' if abs(rf.first_visit_bonus - 0.5) < 0.01 else '❌'}")

print(f"\nExpected frontier_bonus: 0.2")
print(f"Actual: {rf.frontier_bonus}")
print(f"Match: {'✅' if abs(rf.frontier_bonus - 0.2) < 0.01 else '❌'}")

print("\n" + "="*60)
