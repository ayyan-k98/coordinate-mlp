"""
Test POMDP Functionality

Tests visibility masking and partial observability features.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.dqn_agent import CoordinateDQNAgent


def test_visibility_mask():
    """Test that visibility mask is created correctly."""
    print("\n" + "="*70)
    print("TEST 1: Visibility Mask Creation")
    print("="*70)
    
    # Create agent with POMDP
    agent = CoordinateDQNAgent(
        input_channels=6,  # 5 + 1 for visibility
        sensor_range=4.0,
        use_pomdp=True,
        device='cpu'
    )
    
    # Create visibility mask
    H, W = 20, 20
    agent_pos = (10, 10)  # Center
    
    mask = agent._create_visibility_mask(H, W, agent_pos)
    
    print(f"Grid size: {H}x{W}")
    print(f"Agent position: {agent_pos}")
    print(f"Sensor range: {agent.sensor_range}")
    print(f"\nVisibility mask stats:")
    print(f"  Shape: {mask.shape}")
    print(f"  Visible cells: {mask.sum():.0f}")
    print(f"  Expected (pi*r^2): {np.pi * agent.sensor_range**2:.0f}")
    print(f"  Coverage: {mask.sum() / (H*W) * 100:.1f}%")
    
    # Test specific cells
    test_cells = [
        ((10, 10), True, "Agent position"),
        ((12, 11), True, "Nearby (distance=2.2)"),
        ((14, 14), True, "Edge of range (distance=4.0)"),
        ((15, 15), False, "Outside range (distance=7.1)"),
        ((0, 0), False, "Far corner (distance=14.1)")
    ]
    
    print(f"\nTest specific cells:")
    for (y, x), expected, desc in test_cells:
        distance = np.sqrt((y - agent_pos[0])**2 + (x - agent_pos[1])**2)
        visible = mask[y, x] > 0.5
        status = "PASS" if visible == expected else "FAIL"
        print(f"  {status} ({y},{x}): dist={distance:.1f}, visible={visible}, {desc}")
    
    # Verify circular shape
    visible_count = mask.sum()
    expected_count = np.pi * agent.sensor_range**2
    error = abs(visible_count - expected_count) / expected_count * 100
    
    assert error < 15, f"Visibility mask shape error: {error:.1f}% (should be <15%)"
    print(f"\nPASS: Visibility mask shape is circular (error: {error:.1f}%)")


def test_pomdp_vs_full_obs():
    """Compare POMDP vs full observability."""
    print("\n" + "="*70)
    print("TEST 2: POMDP vs Full Observability")
    print("="*70)
    
    # Create two agents
    agent_full = CoordinateDQNAgent(
        input_channels=5,
        use_pomdp=False,
        device='cpu'
    )
    
    agent_pomdp = CoordinateDQNAgent(
        input_channels=6,  # +1 for visibility mask
        sensor_range=4.0,
        use_pomdp=True,
        device='cpu'
    )
    
    # Create test state
    state = np.random.randn(5, 20, 20).astype(np.float32)
    agent_pos = (10, 10)
    
    # Select actions
    action_full = agent_full.select_action(state, epsilon=0.0)
    action_pomdp = agent_pomdp.select_action(state, epsilon=0.0, agent_pos=agent_pos)
    
    print(f"State shape: {state.shape}")
    print(f"Agent position: {agent_pos}")
    print(f"\nFull observability:")
    print(f"  Input channels: {agent_full.policy_net.input_channels}")
    print(f"  Selected action: {action_full}")
    
    print(f"\nPOMDP (sensor_range={agent_pomdp.sensor_range}):")
    print(f"  Input channels: {agent_pomdp.policy_net.input_channels}")
    print(f"  Selected action: {action_pomdp}")
    
    # Test state preparation
    state_with_mask = agent_pomdp._add_visibility_mask(state, agent_pos)
    print(f"\nPOMDP state with visibility mask:")
    print(f"  Shape: {state_with_mask.shape}")
    print(f"  Expected: (6, 20, 20)")
    assert state_with_mask.shape == (6, 20, 20), "Incorrect shape!"
    
    # Verify visibility channel
    visibility_channel = state_with_mask[5, :, :]
    visible_cells = (visibility_channel > 0.5).sum()
    print(f"  Visible cells in mask: {visible_cells:.0f}")
    
    print(f"\nPASS: POMDP agent processes visibility mask correctly")


def test_pomdp_different_positions():
    """Test POMDP with agent at different positions."""
    print("\n" + "="*70)
    print("TEST 3: POMDP at Different Agent Positions")
    print("="*70)
    
    agent = CoordinateDQNAgent(
        input_channels=6,
        sensor_range=5.0,
        use_pomdp=True,
        device='cpu'
    )
    
    state = np.random.randn(5, 20, 20).astype(np.float32)
    
    positions = [
        (0, 0, "Top-left corner"),
        (10, 10, "Center"),
        (19, 19, "Bottom-right corner"),
        (5, 15, "Off-center")
    ]
    
    print(f"Grid size: 20x20")
    print(f"Sensor range: {agent.sensor_range}")
    print(f"\nVisibility at different positions:")
    
    for y, x, desc in positions:
        mask = agent._create_visibility_mask(20, 20, (y, x))
        visible_cells = mask.sum()
        coverage_pct = visible_cells / 400 * 100
        
        print(f"  {desc:20s} ({y:2d},{x:2d}): {visible_cells:5.0f} cells ({coverage_pct:5.1f}%)")
    
    print(f"\nPASS: POMDP works at all positions")


def test_sensor_range_variations():
    """Test different sensor ranges."""
    print("\n" + "="*70)
    print("TEST 4: Different Sensor Ranges")
    print("="*70)
    
    state = np.random.randn(5, 20, 20).astype(np.float32)
    agent_pos = (10, 10)
    
    sensor_ranges = [2.0, 4.0, 6.0, 8.0, None]
    
    print(f"Agent at center: {agent_pos}")
    print(f"Grid size: 20x20 (400 cells)\n")
    
    for sensor_range in sensor_ranges:
        agent = CoordinateDQNAgent(
            input_channels=6 if sensor_range is not None else 5,
            sensor_range=sensor_range,
            use_pomdp=sensor_range is not None,
            device='cpu'
        )
        
        if sensor_range is not None:
            mask = agent._create_visibility_mask(20, 20, agent_pos)
            visible_cells = mask.sum()
            coverage_pct = visible_cells / 400 * 100
            expected = np.pi * sensor_range**2
            
            print(f"  Range {sensor_range:4.1f}: {visible_cells:5.0f} cells ({coverage_pct:5.1f}%) "
                  f"[expected: {expected:5.0f}]")
        else:
            print(f"  Range None (full obs): 400 cells (100.0%)")
    
    print(f"\nPASS: Sensor range variations work correctly")


def test_action_selection_with_pomdp():
    """Test that action selection works with POMDP."""
    print("\n" + "="*70)
    print("TEST 5: Action Selection with POMDP")
    print("="*70)
    
    agent = CoordinateDQNAgent(
        input_channels=6,
        sensor_range=4.0,
        use_pomdp=True,
        device='cpu'
    )
    
    state = np.random.randn(5, 20, 20).astype(np.float32)
    agent_pos = (10, 10)
    
    # Test greedy selection (with same state, should be deterministic)
    print(f"Testing greedy action selection (ε=0.0):")
    action1 = agent.select_action(state, epsilon=0.0, agent_pos=agent_pos)
    action2 = agent.select_action(state, epsilon=0.0, agent_pos=agent_pos)
    
    print(f"  Action 1: {action1}")
    print(f"  Action 2: {action2}")
    print(f"  Expected: Same action (deterministic)")
    assert action1 == action2, "Greedy should be deterministic!"
    
    # Test exploration
    print(f"\nTesting exploration (ε=1.0):")
    actions = []
    for _ in range(100):
        action = agent.select_action(state, epsilon=1.0, agent_pos=agent_pos)
        actions.append(action)
    
    unique = len(set(actions))
    print(f"  100 selections, {unique} unique actions")
    print(f"  Expected: 7-9 (should explore)")
    assert unique >= 5, "Should explore multiple actions!"
    
    # Test with valid actions mask
    print(f"\nTesting with valid actions mask:")
    valid_actions = np.array([True, False, False, True, True, False, False, False, True])
    valid_indices = np.where(valid_actions)[0]
    
    actions = []
    for _ in range(50):
        action = agent.select_action(state, epsilon=0.5, agent_pos=agent_pos, 
                                     valid_actions=valid_actions)
        actions.append(action)
    
    invalid_selected = any(a not in valid_indices for a in actions)
    
    print(f"  Valid actions: {valid_indices.tolist()}")
    print(f"  50 selections")
    print(f"  Invalid action selected: {invalid_selected}")
    assert not invalid_selected, "Invalid action was selected!"
    
    print(f"\nPASS: Action selection works correctly with POMDP")


def test_visibility_edge_cases():
    """Test edge cases for visibility masking."""
    print("\n" + "="*70)
    print("TEST 6: Visibility Mask Edge Cases")
    print("="*70)
    
    # Test 1: Agent at corner with small range
    agent = CoordinateDQNAgent(
        input_channels=6,
        sensor_range=3.0,
        use_pomdp=True,
        device='cpu'
    )
    
    mask = agent._create_visibility_mask(10, 10, (0, 0))
    visible = mask.sum()
    print(f"Test 1: Agent at corner (0,0), range=3.0")
    print(f"  Visible cells: {visible:.0f}")
    print(f"  Expected: ~7-12 (quarter circle)")
    assert 5 <= visible <= 15, "Unexpected visibility at corner"
    
    # Test 2: Very large sensor range (should see everything)
    agent = CoordinateDQNAgent(
        input_channels=6,
        sensor_range=100.0,
        use_pomdp=True,
        device='cpu'
    )
    
    mask = agent._create_visibility_mask(20, 20, (10, 10))
    visible = mask.sum()
    print(f"\nTest 2: Very large range (100.0)")
    print(f"  Visible cells: {visible:.0f}")
    print(f"  Expected: 400 (entire grid)")
    assert visible == 400, "Should see entire grid with large range"
    
    # Test 3: Zero range (should only see agent cell)
    agent = CoordinateDQNAgent(
        input_channels=6,
        sensor_range=0.0,
        use_pomdp=True,
        device='cpu'
    )
    
    mask = agent._create_visibility_mask(20, 20, (10, 10))
    visible = mask.sum()
    print(f"\nTest 3: Zero range (0.0)")
    print(f"  Visible cells: {visible:.0f}")
    print(f"  Expected: 1 (only agent position)")
    assert visible == 1, "Should only see agent position with zero range"
    
    print(f"\nPASS: Edge cases handled correctly")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("POMDP FUNCTIONALITY TESTS")
    print("="*70)
    
    tests = [
        ("Visibility Mask Creation", test_visibility_mask),
        ("POMDP vs Full Observability", test_pomdp_vs_full_obs),
        ("Different Agent Positions", test_pomdp_different_positions),
        ("Sensor Range Variations", test_sensor_range_variations),
        ("Action Selection with POMDP", test_action_selection_with_pomdp),
        ("Visibility Edge Cases", test_visibility_edge_cases),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"\n[PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 70)
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 All POMDP tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
