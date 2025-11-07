"""
Test STAY Penalty Implementation

Verifies that STAY action is properly penalized to encourage movement.
"""

import numpy as np
from coverage_env import CoverageEnvironment, Action


def test_stay_penalty():
    """Test STAY penalty discourages standing still."""
    
    print("="*70)
    print("Testing STAY Penalty System")
    print("="*70)
    
    # Create environment with STAY penalty
    env = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,  # Empty map
        max_steps=100,
        seed=42,
        reward_config={
            'stay_penalty': -1.0,  # Default penalty
            'step_penalty': -0.01
        }
    )
    
    env.reset()
    print("\n1. Environment initialized")
    print(f"   STAY penalty: {env.reward_fn.stay_penalty}")
    print(f"   Step penalty: {env.reward_fn.step_penalty}")
    
    # Test 1: STAY action incurs penalty
    print("\n2. Test: STAY action")
    obs, reward, done, info = env.step(Action.STAY)
    
    stay_penalty = info['reward_breakdown']['stay']
    step_penalty = info['reward_breakdown']['step']
    total_penalty = stay_penalty + step_penalty
    
    print(f"   STAY penalty:  {stay_penalty:.3f} (expected: -1.000)")
    print(f"   Step penalty:  {step_penalty:.3f} (expected: -0.010)")
    print(f"   Total penalty: {total_penalty:.3f}")
    
    assert stay_penalty == -1.0, f"STAY penalty should be -1.0, got {stay_penalty}"
    print("   ✅ STAY action penalized")
    
    # Test 2: Movement has no STAY penalty
    print("\n3. Test: Movement action (NORTH)")
    env.reset()
    obs, reward, done, info = env.step(Action.NORTH)
    
    stay_penalty = info['reward_breakdown']['stay']
    step_penalty = info['reward_breakdown']['step']
    
    print(f"   STAY penalty: {stay_penalty:.3f} (expected: 0.000)")
    print(f"   Step penalty: {step_penalty:.3f} (expected: -0.010)")
    
    assert stay_penalty == 0.0, f"Movement should have no STAY penalty, got {stay_penalty}"
    print("   ✅ Movement has no STAY penalty")
    
    # Test 3: Compare STAY vs MOVE rewards
    print("\n4. Test: STAY vs MOVE reward comparison")
    
    # Episode with STAY
    env.reset()
    stay_rewards = []
    for _ in range(5):
        _, r, _, info = env.step(Action.STAY)
        stay_rewards.append(r)
    avg_stay_reward = np.mean(stay_rewards)
    
    # Episode with movement
    env.reset()
    move_rewards = []
    for _ in range(5):
        _, r, _, info = env.step(Action.NORTH)
        move_rewards.append(r)
    avg_move_reward = np.mean(move_rewards)
    
    print(f"   Average STAY reward: {avg_stay_reward:.3f}")
    print(f"   Average MOVE reward: {avg_move_reward:.3f}")
    print(f"   Difference:          {avg_move_reward - avg_stay_reward:.3f}")
    
    assert avg_move_reward > avg_stay_reward, "Movement should be more rewarding than staying"
    print("   ✅ Movement is more rewarding than STAY")
    
    # Test 4: Multiple consecutive STAY actions
    print("\n5. Test: Consecutive STAY actions accumulate penalties")
    env.reset()
    
    cumulative_stay_penalty = 0.0
    for i in range(3):
        _, r, _, info = env.step(Action.STAY)
        stay_p = info['reward_breakdown']['stay']
        cumulative_stay_penalty += stay_p
        print(f"   Step {i+1}: STAY penalty = {stay_p:.3f}, cumulative = {cumulative_stay_penalty:.3f}")
    
    expected_cumulative = -1.0 * 3
    print(f"   Expected cumulative: {expected_cumulative:.3f}")
    assert abs(cumulative_stay_penalty - expected_cumulative) < 0.001, \
        f"Expected {expected_cumulative}, got {cumulative_stay_penalty}"
    print("   ✅ Consecutive STAY penalties accumulate correctly")
    
    # Test 5: STAY penalty can be disabled
    print("\n6. Test: STAY penalty can be disabled")
    env_no_stay = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,
        max_steps=100,
        seed=42,
        reward_config={
            'stay_penalty': 0.0  # Disabled
        }
    )
    
    env_no_stay.reset()
    _, r, _, info = env_no_stay.step(Action.STAY)
    stay_penalty = info['reward_breakdown']['stay']
    
    print(f"   STAY penalty (disabled): {stay_penalty:.3f} (expected: 0.000)")
    assert stay_penalty == 0.0, f"Disabled should be 0.0, got {stay_penalty}"
    print("   ✅ STAY penalty can be disabled")
    
    # Test 6: Custom STAY penalty values
    print("\n7. Test: Custom STAY penalty values")
    
    custom_penalties = [-0.5, -2.0, -5.0]
    for custom_p in custom_penalties:
        env_custom = CoverageEnvironment(
            grid_size=20,
            num_agents=1,
            sensor_range=4.0,
            obstacle_density=0.0,
            max_steps=100,
            seed=42,
            reward_config={
                'stay_penalty': custom_p
            }
        )
        
        env_custom.reset()
        _, r, _, info = env_custom.step(Action.STAY)
        stay_p = info['reward_breakdown']['stay']
        
        print(f"   Custom penalty {custom_p:.1f}: got {stay_p:.3f}")
        assert abs(stay_p - custom_p) < 0.001, f"Expected {custom_p}, got {stay_p}"
    
    print("   ✅ Custom STAY penalties work correctly")
    
    # Test 7: STAY vs exploration trade-off
    print("\n8. Test: STAY penalty encourages exploration")
    
    # Agent forced to STAY (high penalty)
    env_high_stay = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,
        max_steps=20,
        seed=42,
        reward_config={
            'stay_penalty': -5.0  # Very high
        }
    )
    
    env_high_stay.reset()
    total_reward_with_stay = 0.0
    
    # Try to STAY multiple times
    for _ in range(10):
        _, r, done, _ = env_high_stay.step(Action.STAY)
        total_reward_with_stay += r
        if done:
            break
    
    # Agent that moves
    env_move = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,
        max_steps=20,
        seed=42,
        reward_config={
            'stay_penalty': -5.0
        }
    )
    
    env_move.reset()
    total_reward_with_move = 0.0
    
    # Move in one direction
    for _ in range(10):
        _, r, done, _ = env_move.step(Action.NORTH)
        total_reward_with_move += r
        if done:
            break
    
    print(f"   Total reward (STAY):  {total_reward_with_stay:.2f}")
    print(f"   Total reward (MOVE):  {total_reward_with_move:.2f}")
    print(f"   Difference:           {total_reward_with_move - total_reward_with_stay:.2f}")
    
    assert total_reward_with_move > total_reward_with_stay, \
        "Moving should yield higher reward than staying"
    print("   ✅ High STAY penalty strongly encourages exploration")
    
    print("\n" + "="*70)
    print("STAY Penalty Test Results")
    print("="*70)
    print("✅ All tests passed!")
    print("\nImplemented Features:")
    print("  ✅ STAY action incurs penalty (-1.0 default)")
    print("  ✅ Movement actions have no STAY penalty")
    print("  ✅ STAY penalties accumulate over multiple steps")
    print("  ✅ Can disable STAY penalty (set to 0.0)")
    print("  ✅ Supports custom penalty values")
    print("  ✅ Encourages exploration vs standing still")
    print("\nReward Breakdown:")
    print("  STAY action:")
    print("    • STAY penalty:  -1.00")
    print("    • Step penalty:  -0.01")
    print("    • Total penalty: -1.01 (plus other components)")
    print("\n  MOVE action:")
    print("    • STAY penalty:   0.00")
    print("    • Step penalty:  -0.01")
    print("    • Total penalty: -0.01 (plus other components)")
    print("\nEffect on Training:")
    print("  • Agent learns to keep moving")
    print("  • Prevents standing still to avoid negative rewards")
    print("  • Complements rotation penalty (smooth movement)")
    print("  • Balances with coverage rewards")
    print("="*70)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    test_stay_penalty()
    
    print("\n🎉 STAY penalty implementation verified!")
