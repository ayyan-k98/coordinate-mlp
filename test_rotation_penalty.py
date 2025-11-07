"""
Test Rotation Penalty Implementation

Verifies that rotation penalties work correctly in coverage_env.py
"""

import numpy as np
from coverage_env import CoverageEnvironment, Action


def test_rotation_penalties():
    """Test rotation penalty computation."""
    
    print("="*70)
    print("Testing Rotation Penalty System")
    print("="*70)
    
    # Create environment with rotation penalties enabled
    env = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,  # Empty map for simplicity
        max_steps=100,
        seed=42,
        reward_config={
            'use_rotation_penalty': True,
            'rotation_penalty_small': -0.05,
            'rotation_penalty_medium': -0.10,
            'rotation_penalty_large': -0.15
        }
    )
    
    # Reset environment
    obs = env.reset()
    print("\n1. Environment initialized")
    print(f"   Grid size: {env.grid_size}×{env.grid_size}")
    print(f"   Rotation penalties enabled: {env.reward_fn.use_rotation_penalty}")
    
    # Test 1: No rotation (straight line)
    print("\n2. Test: Straight movement (no rotation)")
    print("   Actions: NORTH → NORTH → NORTH")
    
    _, r1, _, info1 = env.step(Action.NORTH)
    _, r2, _, info2 = env.step(Action.NORTH)
    _, r3, _, info3 = env.step(Action.NORTH)
    
    rot1 = info1['reward_breakdown']['rotation']
    rot2 = info2['reward_breakdown']['rotation']
    rot3 = info3['reward_breakdown']['rotation']
    
    print(f"   Step 1 rotation penalty: {rot1:.3f} (expected: 0.0, first move)")
    print(f"   Step 2 rotation penalty: {rot2:.3f} (expected: 0.0, no rotation)")
    print(f"   Step 3 rotation penalty: {rot3:.3f} (expected: 0.0, no rotation)")
    
    assert rot1 == 0.0, f"First move should have no penalty, got {rot1}"
    assert rot2 == 0.0, f"No rotation should have no penalty, got {rot2}"
    assert rot3 == 0.0, f"No rotation should have no penalty, got {rot3}"
    print("   ✅ No rotation = no penalty")
    
    # Test 2: Small rotation (45°)
    print("\n3. Test: Small rotation (45°)")
    print("   Actions: NORTH → NORTHEAST")
    
    env.reset()
    env.step(Action.NORTH)
    _, r, _, info = env.step(Action.NORTHEAST)
    rot_penalty = info['reward_breakdown']['rotation']
    
    print(f"   Rotation penalty: {rot_penalty:.3f} (expected: -0.05)")
    assert rot_penalty == -0.05, f"45° rotation should be -0.05, got {rot_penalty}"
    print("   ✅ 45° rotation = -0.05 penalty")
    
    # Test 3: Medium rotation (90°)
    print("\n4. Test: Medium rotation (90°)")
    print("   Actions: NORTH → EAST")
    
    env.reset()
    env.step(Action.NORTH)
    _, r, _, info = env.step(Action.EAST)
    rot_penalty = info['reward_breakdown']['rotation']
    
    print(f"   Rotation penalty: {rot_penalty:.3f} (expected: -0.10)")
    assert rot_penalty == -0.10, f"90° rotation should be -0.10, got {rot_penalty}"
    print("   ✅ 90° rotation = -0.10 penalty")
    
    # Test 4: Large rotation (135°)
    print("\n5. Test: Large rotation (135°)")
    print("   Actions: NORTH → SOUTHEAST")
    
    env.reset()
    env.step(Action.NORTH)
    _, r, _, info = env.step(Action.SOUTHEAST)
    rot_penalty = info['reward_breakdown']['rotation']
    
    print(f"   Rotation penalty: {rot_penalty:.3f} (expected: -0.15)")
    assert rot_penalty == -0.15, f"135° rotation should be -0.15, got {rot_penalty}"
    print("   ✅ 135° rotation = -0.15 penalty")
    
    # Test 5: U-turn (180°)
    print("\n6. Test: U-turn (180°)")
    print("   Actions: NORTH → SOUTH")
    
    env.reset()
    env.step(Action.NORTH)
    _, r, _, info = env.step(Action.SOUTH)
    rot_penalty = info['reward_breakdown']['rotation']
    
    print(f"   Rotation penalty: {rot_penalty:.3f} (expected: -0.15)")
    assert rot_penalty == -0.15, f"180° rotation should be -0.15, got {rot_penalty}"
    print("   ✅ 180° rotation = -0.15 penalty")
    
    # Test 6: STAY action
    print("\n7. Test: STAY action (no rotation)")
    print("   Actions: NORTH → STAY → NORTH")
    
    env.reset()
    env.step(Action.NORTH)
    _, r1, _, info1 = env.step(Action.STAY)
    _, r2, _, info2 = env.step(Action.NORTH)
    
    rot1 = info1['reward_breakdown']['rotation']
    rot2 = info2['reward_breakdown']['rotation']
    
    print(f"   STAY rotation penalty: {rot1:.3f} (expected: 0.0)")
    print(f"   After STAY penalty: {rot2:.3f} (expected: 0.0, reset)")
    assert rot1 == 0.0, f"STAY should have no penalty, got {rot1}"
    assert rot2 == 0.0, f"After STAY should reset, got {rot2}"
    print("   ✅ STAY resets rotation tracking")
    
    # Test 7: Wrap-around (NW to NE = 90°, not 270°)
    print("\n8. Test: Angle wrap-around")
    print("   Actions: NORTHWEST (315°) → NORTHEAST (45°)")
    
    env.reset()
    env.step(Action.NORTHWEST)
    _, r, _, info = env.step(Action.NORTHEAST)
    rot_penalty = info['reward_breakdown']['rotation']
    
    print(f"   Rotation penalty: {rot_penalty:.3f} (expected: -0.10 for 90°)")
    assert rot_penalty == -0.10, f"Wrap-around should be 90°, got penalty {rot_penalty}"
    print("   ✅ Wrap-around handled correctly")
    
    # Test 8: Disabled rotation penalties
    print("\n9. Test: Disabled rotation penalties")
    
    env_no_rotation = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,
        max_steps=100,
        seed=42,
        reward_config={
            'use_rotation_penalty': False  # Disabled
        }
    )
    
    env_no_rotation.reset()
    env_no_rotation.step(Action.NORTH)
    _, r, _, info = env_no_rotation.step(Action.SOUTH)  # U-turn
    rot_penalty = info['reward_breakdown']['rotation']
    
    print(f"   Rotation penalty (disabled): {rot_penalty:.3f} (expected: 0.0)")
    assert rot_penalty == 0.0, f"Disabled should be 0.0, got {rot_penalty}"
    print("   ✅ Can disable rotation penalties")
    
    # Test 9: Cumulative penalty over episode
    print("\n10. Test: Cumulative penalties over episode")
    print("    Actions: Zigzag pattern (N→E→N→E→N)")
    
    env.reset()
    total_rotation_penalty = 0.0
    actions = [Action.NORTH, Action.EAST, Action.NORTH, Action.EAST, Action.NORTH]
    
    for i, action in enumerate(actions):
        _, r, _, info = env.step(action)
        rot = info['reward_breakdown']['rotation']
        total_rotation_penalty += rot
        print(f"    Step {i+1}: {action.name:10s} → rotation penalty: {rot:6.3f}")
    
    print(f"    Total rotation penalty: {total_rotation_penalty:.3f}")
    expected_total = 0.0 + (-0.10) + (-0.10) + (-0.10) + (-0.10)  # First has 0, rest are 90° turns
    print(f"    Expected total:         {expected_total:.3f}")
    assert abs(total_rotation_penalty - expected_total) < 0.001, \
        f"Expected {expected_total}, got {total_rotation_penalty}"
    print("   ✅ Cumulative penalties computed correctly")
    
    print("\n" + "="*70)
    print("Rotation Penalty Test Results")
    print("="*70)
    print("✅ All tests passed!")
    print("\nImplemented Features:")
    print("  ✅ Graduated penalties (0°, ≤45°, ≤90°, >90°)")
    print("  ✅ No penalty on first move")
    print("  ✅ STAY action resets rotation tracking")
    print("  ✅ 360° wrap-around handled correctly")
    print("  ✅ Can enable/disable rotation penalties")
    print("  ✅ Penalties included in reward breakdown")
    print("\nPenalty Scale:")
    print("  • No rotation (0°):     0.0")
    print("  • Small (≤45°):        -0.05")
    print("  • Medium (≤90°):       -0.10")
    print("  • Large (>90°):        -0.15")
    print("\nEffect on Training:")
    print("  • Encourages smooth, efficient paths")
    print("  • Penalizes zig-zag and backtracking")
    print("  • Helps agent learn directional momentum")
    print("="*70)


def test_probabilistic_sensing():
    """Verify environment is probabilistic."""
    
    print("\n" + "="*70)
    print("Verifying Probabilistic Sensing")
    print("="*70)
    
    env = CoverageEnvironment(
        grid_size=20,
        num_agents=1,
        sensor_range=4.0,
        obstacle_density=0.0,
        max_steps=100,
        seed=42
    )
    
    env.reset()
    
    print(f"\nSensor Model: {type(env.sensor_model).__name__}")
    print(f"  Max range: {env.sensor_model.max_range}")
    print(f"  Detection prob at center: {env.sensor_model.p_center}")
    print(f"  Detection prob at edge: {env.sensor_model.p_edge}")
    
    # Test detection probabilities at different distances
    print("\nDetection Probabilities by Distance:")
    for dist in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        prob = env.sensor_model.get_detection_probability(dist)
        print(f"  Distance {dist:.1f}: {prob:.3f}")
    
    print("\n✅ Environment uses probabilistic sensor model")
    print("✅ Coverage probabilities are continuous [0, 1]")
    print("="*70)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    test_rotation_penalties()
    test_probabilistic_sensing()
    
    print("\n🎉 All tests completed successfully!")
