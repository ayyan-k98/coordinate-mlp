"""
Quick test to verify progressive revisit penalty implementation.
"""

from coverage_env import CoverageEnvironment, RewardFunction, CoverageState, AgentState
import numpy as np

def test_progressive_penalty():
    """Test that revisit penalty scales correctly with episode progress."""

    print("="*70)
    print("Testing Progressive Revisit Penalty")
    print("="*70)

    # Configuration
    max_steps = 500
    penalty_min = -0.1
    penalty_max = -0.5

    # Create reward function with progressive penalty
    reward_fn = RewardFunction(
        use_progressive_revisit_penalty=True,
        revisit_penalty_min=penalty_min,
        revisit_penalty_max=penalty_max,
        max_steps=max_steps
    )

    print(f"\nConfiguration:")
    print(f"  Max Steps: {max_steps}")
    print(f"  Penalty Range: {penalty_min} → {penalty_max}")
    print(f"  Progressive: {reward_fn.use_progressive_revisit}")

    # Test at different episode progress points
    test_steps = [0, 125, 250, 375, 500, 600]  # 0%, 25%, 50%, 75%, 100%, >100%

    print(f"\nProgressive Penalty Test:")
    print(f"{'Step':<10} {'Progress':<12} {'Expected':<15} {'Actual':<15} {'Status':<10}")
    print("-" * 70)

    for step in test_steps:
        # Calculate expected penalty
        progress = min(step / max_steps, 1.0)
        expected = penalty_min + (penalty_max - penalty_min) * progress

        # Create mock states
        grid_size = 20
        state = CoverageState(
            height=grid_size,
            width=grid_size,
            obstacles=np.zeros((grid_size, grid_size), dtype=bool),
            visited=np.ones((grid_size, grid_size), dtype=bool),  # All visited
            coverage=np.zeros((grid_size, grid_size)),
            coverage_confidence=np.zeros((grid_size, grid_size)),
            agents=[AgentState(x=10, y=10, sensor_range=4.0)],
            frontiers=set(),
            step=step
        )

        next_state = state.copy()
        next_state.step = step  # Ensure step is set

        # Compute reward (should contain revisit penalty)
        total_reward, breakdown = reward_fn.compute_reward(
            state=state,
            next_state=next_state,
            action=0,
            agent_id=0
        )

        actual = breakdown['revisit']

        # Check if actual matches expected (within floating point tolerance)
        matches = abs(actual - expected) < 1e-6
        status = "✓ PASS" if matches else "✗ FAIL"

        print(f"{step:<10} {progress*100:>5.1f}%      {expected:>+8.3f}       {actual:>+8.3f}       {status}")

    print("\n" + "="*70)

    # Test with progressive disabled
    print("\nTesting with Progressive Penalty DISABLED:")
    print("="*70)

    reward_fn_static = RewardFunction(
        use_progressive_revisit_penalty=False,
        revisit_penalty=-0.5,
        max_steps=max_steps
    )

    print(f"\nStatic Penalty Test:")
    print(f"{'Step':<10} {'Expected':<15} {'Actual':<15} {'Status':<10}")
    print("-" * 70)

    for step in test_steps[:3]:  # Just test a few steps
        state = CoverageState(
            height=grid_size,
            width=grid_size,
            obstacles=np.zeros((grid_size, grid_size), dtype=bool),
            visited=np.ones((grid_size, grid_size), dtype=bool),
            coverage=np.zeros((grid_size, grid_size)),
            coverage_confidence=np.zeros((grid_size, grid_size)),
            agents=[AgentState(x=10, y=10, sensor_range=4.0)],
            frontiers=set(),
            step=step
        )

        next_state = state.copy()
        next_state.step = step

        total_reward, breakdown = reward_fn_static.compute_reward(
            state=state,
            next_state=next_state,
            action=0,
            agent_id=0
        )

        actual = breakdown['revisit']
        expected = -0.5  # Should always be -0.5

        matches = abs(actual - expected) < 1e-6
        status = "✓ PASS" if matches else "✗ FAIL"

        print(f"{step:<10} {expected:>+8.3f}       {actual:>+8.3f}       {status}")

    print("\n" + "="*70)
    print("✓ All tests completed!")
    print("="*70)

if __name__ == "__main__":
    test_progressive_penalty()
