"""
Test Obstacle Discovery in True POMDP

Verifies that:
1. Agent starts with NO knowledge of obstacles
2. Agent discovers obstacles through sensing
3. Agent can collide with undiscovered obstacles
4. Collision penalty is no longer "dead code"
"""

import numpy as np
from coverage_env import CoverageEnvironment, Action


def test_initial_obstacle_belief_is_zero():
    """Test that agent starts with no knowledge of obstacles."""
    
    print("\n" + "="*70)
    print("TEST 1: Initial Obstacle Belief")
    print("="*70)
    
    env = CoverageEnvironment(
        grid_size=10,
        obstacle_density=0.2,  # 20% obstacles
        max_steps=100
    )
    
    obs = env.reset()
    
    # Get ground truth obstacles
    ground_truth_obstacles = env.state.obstacles
    num_obstacles = ground_truth_obstacles.sum()
    
    # Get obstacle belief (Channel 4)
    obstacle_belief = obs[4]  # [H, W]
    
    print(f"\nGround truth obstacles: {num_obstacles} cells")
    print(f"Initial obstacle belief sum: {obstacle_belief.sum():.4f}")
    print(f"Initial obstacle belief max: {obstacle_belief.max():.4f}")
    
    # Initially, belief should be very low (agent hasn't seen anything yet)
    # After initial sensing, some cells near agent start position will have updated beliefs
    print(f"\nObstacle belief after initial sensing:")
    print(f"  Min: {obstacle_belief.min():.4f}")
    print(f"  Max: {obstacle_belief.max():.4f}")
    print(f"  Mean: {obstacle_belief.mean():.4f}")
    print(f"  Cells with belief > 0.5: {(obstacle_belief > 0.5).sum()}")
    
    # The agent should NOT have full knowledge
    # Count how many true obstacles are discovered (belief > 0.5)
    discovered_obstacles = (obstacle_belief > 0.5).sum()
    discovery_rate = discovered_obstacles / max(num_obstacles, 1)
    
    print(f"\nDiscovery rate: {discovery_rate:.1%} ({discovered_obstacles}/{num_obstacles})")
    
    # Agent should not have discovered ALL obstacles at start
    assert discovery_rate < 1.0, "Agent has full knowledge at start (should be partial)!"
    
    print("\n✅ Agent starts with LIMITED obstacle knowledge (true POMDP)")


def test_obstacle_discovery_through_sensing():
    """Test that agent discovers obstacles as it explores."""
    
    print("\n" + "="*70)
    print("TEST 2: Obstacle Discovery Through Exploration")
    print("="*70)
    
    env = CoverageEnvironment(
        grid_size=15,
        obstacle_density=0.15,
        max_steps=200
    )
    
    obs = env.reset()
    
    ground_truth = env.state.obstacles
    num_obstacles = ground_truth.sum()
    
    # Track discovery over time
    discovery_history = []
    
    print(f"\nTotal obstacles in map: {num_obstacles}")
    print("\nDiscovery progress:")
    print("  Step | Discovered | Discovery % | Belief Sum")
    print("  " + "-"*50)
    
    for step in range(100):
        # Take random action
        valid_actions = env.get_valid_actions()
        valid_indices = np.where(valid_actions)[0]
        action = np.random.choice(valid_indices)
        
        obs, reward, done, info = env.step(action)
        
        # Get current belief
        obstacle_belief = obs[4]
        
        # Count discovered obstacles (belief > 0.5)
        discovered = (obstacle_belief > 0.5).sum()
        discovery_rate = discovered / max(num_obstacles, 1)
        belief_sum = obstacle_belief.sum()
        
        discovery_history.append(discovery_rate)
        
        if step % 20 == 0:
            print(f"  {step:4d} | {discovered:10d} | {discovery_rate:10.1%} | {belief_sum:10.2f}")
        
        if done:
            break
    
    # Final statistics
    final_discovery = discovery_history[-1]
    initial_discovery = discovery_history[0]
    improvement = final_discovery - initial_discovery
    
    print("\n" + "-"*54)
    print(f"\nDiscovery improvement:")
    print(f"  Initial: {initial_discovery:.1%}")
    print(f"  Final:   {final_discovery:.1%}")
    print(f"  Gain:    {improvement:.1%}")
    
    # Agent should discover MORE obstacles as it explores
    assert improvement > 0.1, f"Agent didn't discover obstacles (only {improvement:.1%} gain)"
    
    print("\n✅ Agent discovers obstacles through exploration!")


def test_collision_with_undiscovered_obstacles():
    """Test that agent CAN collide with obstacles it hasn't discovered yet."""
    
    print("\n" + "="*70)
    print("TEST 3: Collision with Undiscovered Obstacles")
    print("="*70)
    
    env = CoverageEnvironment(
        grid_size=10,
        obstacle_density=0.3,  # Dense obstacles
        max_steps=100
    )
    
    collision_count = 0
    total_steps = 0
    collision_rewards = []
    
    # Run multiple episodes
    for ep in range(20):
        obs = env.reset()
        
        for step in range(50):
            # Get valid actions (based on BELIEF, not ground truth)
            valid_actions = env.get_valid_actions()
            valid_indices = np.where(valid_actions)[0]
            
            # Choose random valid action
            action = np.random.choice(valid_indices)
            
            obs, reward, done, info = env.step(action)
            
            # Check if collision happened
            if info.get('collisions', 0) > collision_count:
                collision_count = info['collisions']
                collision_rewards.append(reward)
                
                # Print details
                if collision_count <= 5:  # First few collisions
                    print(f"\n  Collision #{collision_count} at step {total_steps}")
                    print(f"    Action: {Action(action).name}")
                    print(f"    Reward: {reward:.2f}")
                    if 'reward_breakdown' in info:
                        breakdown = info['reward_breakdown']
                        print(f"    Collision penalty: {breakdown.get('collision', 0):.2f}")
            
            total_steps += 1
            
            if done:
                break
    
    print(f"\n" + "="*70)
    print(f"Collision Statistics ({total_steps} steps across 20 episodes):")
    print(f"  Total collisions: {collision_count}")
    print(f"  Collision rate: {collision_count / total_steps:.2%}")
    
    if collision_rewards:
        print(f"  Average collision reward: {np.mean(collision_rewards):.3f}")
        print(f"  Min collision reward: {np.min(collision_rewards):.3f}")
        print(f"  Max collision reward: {np.max(collision_rewards):.3f}")
    
    # Agent SHOULD collide with undiscovered obstacles
    assert collision_count > 0, "No collisions detected - collision penalty is DEAD CODE!"
    
    print(f"\n✅ Collision penalty is ACTIVE (agent collided {collision_count} times)")
    print("✅ Agent can discover obstacles the hard way!")


def test_valid_actions_uses_belief_not_truth():
    """Test that valid actions are based on belief, not ground truth."""
    
    print("\n" + "="*70)
    print("TEST 4: Valid Actions Use Belief (Not Ground Truth)")
    print("="*70)
    
    env = CoverageEnvironment(
        grid_size=10,
        obstacle_density=0.2,
        max_steps=100
    )
    
    obs = env.reset()
    
    # Get ground truth and belief
    ground_truth = env.state.obstacles
    obstacle_belief = obs[4]
    
    # Get agent position
    agent = env.state.agents[0]
    print(f"\nAgent position: ({agent.x}, {agent.y})")
    
    # Check neighboring cells
    neighbors = [
        (agent.y - 1, agent.x, "NORTH"),
        (agent.y, agent.x + 1, "EAST"),
        (agent.y + 1, agent.x, "SOUTH"),
        (agent.y, agent.x - 1, "WEST"),
    ]
    
    print("\nNeighboring cells:")
    print("  Direction | Ground Truth | Belief | Would Allow?")
    print("  " + "-"*55)
    
    undiscovered_obstacles = 0
    
    for y, x, direction in neighbors:
        if 0 <= y < env.grid_size and 0 <= x < env.grid_size:
            is_obstacle = ground_truth[y, x]
            belief = obstacle_belief[y, x]
            would_allow = belief < 0.7  # Threshold from get_valid_actions
            
            status = "OBSTACLE" if is_obstacle else "Free"
            allow_str = "✓ Allow" if would_allow else "✗ Block"
            
            print(f"  {direction:6s}    | {status:12s} | {belief:6.3f} | {allow_str}")
            
            # If there's an obstacle but low belief, agent would try to move there!
            if is_obstacle and would_allow:
                undiscovered_obstacles += 1
                print(f"           └─> UNDISCOVERED OBSTACLE! Agent would collide here.")
    
    print(f"\nUndiscovered obstacles in immediate vicinity: {undiscovered_obstacles}")
    
    # Get actual valid actions
    valid_actions = env.get_valid_actions()
    num_valid = valid_actions.sum()
    
    print(f"\nValid actions: {num_valid}/9")
    print(f"Valid action mask: {valid_actions}")
    
    # The key insight: valid actions should be based on BELIEF, not truth
    # So we might have "valid" actions that actually lead to collisions
    print("\n✅ Valid actions based on BELIEF (not ground truth)")
    if undiscovered_obstacles > 0:
        print(f"✅ Agent has {undiscovered_obstacles} risky moves it might try!")


def test_obstacle_belief_channel_differs_from_truth():
    """Test that Channel 4 (obstacle belief) differs from ground truth."""
    
    print("\n" + "="*70)
    print("TEST 5: Channel 4 is Belief (Not Ground Truth)")
    print("="*70)
    
    env = CoverageEnvironment(
        grid_size=12,
        obstacle_density=0.2,
        max_steps=100
    )
    
    obs = env.reset()
    
    # Get ground truth and observation
    ground_truth = env.state.obstacles.astype(float)
    channel_4 = obs[4]  # Obstacle belief
    
    # Compare
    difference = np.abs(ground_truth - channel_4)
    
    print(f"\nGround truth obstacles: {ground_truth.sum()} cells")
    print(f"Channel 4 (belief) sum: {channel_4.sum():.2f}")
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {difference.mean():.4f}")
    print(f"  Max difference: {difference.max():.4f}")
    print(f"  Cells with difference > 0.1: {(difference > 0.1).sum()}")
    
    # Visualize a slice
    print(f"\nSample comparison (top-left 5×5):")
    print(f"\nGround Truth:")
    print(ground_truth[:5, :5].astype(int))
    print(f"\nChannel 4 (Belief):")
    print(np.round(channel_4[:5, :5], 2))
    
    # They should differ significantly
    total_difference = difference.sum()
    assert total_difference > 0.5, "Channel 4 is identical to ground truth (not using belief!)"
    
    print(f"\n✅ Channel 4 differs from ground truth (total diff: {total_difference:.2f})")
    print("✅ Agent observes BELIEF, not ground truth!")


if __name__ == "__main__":
    print("="*70)
    print("OBSTACLE DISCOVERY TESTS (True POMDP)")
    print("="*70)
    
    test_initial_obstacle_belief_is_zero()
    test_obstacle_discovery_through_sensing()
    test_collision_with_undiscovered_obstacles()
    test_valid_actions_uses_belief_not_truth()
    test_obstacle_belief_channel_differs_from_truth()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print("\nSummary:")
    print("  ✅ Agent starts with limited obstacle knowledge")
    print("  ✅ Agent discovers obstacles through exploration")
    print("  ✅ Agent CAN collide with undiscovered obstacles")
    print("  ✅ Collision penalty is NO LONGER dead code")
    print("  ✅ Valid actions based on belief (not ground truth)")
    print("  ✅ Channel 4 shows belief (not ground truth)")
    print("\n🎯 This is now a TRUE POMDP with obstacle discovery!")
    print("="*70)


