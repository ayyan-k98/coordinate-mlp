"""
Comprehensive Environment Examples

Demonstrates all the nitty-gritty details:
- Probabilistic sensing and coverage dynamics
- Action space and state representation
- Reward computation breakdown
- Frontier detection and exploration
- Multi-agent coordination (future)
"""

import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import CoverageEnvironment


def example1_basic_episode():
    """Example 1: Basic episode with detailed logging."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Episode with Detailed State Tracking")
    print("="*80)
    
    # Create environment
    env = CoverageEnvironment(
        grid_size=15,
        num_agents=1,
        sensor_range=3.0,
        obstacle_density=0.1,
        max_steps=100,
        seed=42
    )
    
    # Reset and examine initial state
    obs = env.reset()
    print(f"\n📊 Initial State:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Channels: [visited, coverage, agents, frontiers, obstacles]")
    print(f"  Grid size: {env.grid_size}×{env.grid_size}")
    print(f"  Agent position: ({env.state.agents[0].x}, {env.state.agents[0].y})")
    print(f"  Obstacles: {env.state.obstacles.sum()} cells ({env.state.obstacles.sum()/(env.grid_size**2)*100:.1f}%)")
    print(f"  Initial coverage: {env.episode_stats['coverage_percentage']*100:.1f}%")
    print(f"  Frontiers: {len(env.state.frontiers)} cells")
    
    # Run episode
    print(f"\n🎮 Running Episode:")
    done = False
    step = 0
    
    while not done and step < 50:
        # Get valid actions
        valid_actions = env.get_valid_actions(agent_id=0)
        
        # Select random valid action
        action = np.random.choice(np.where(valid_actions)[0])
        action_name = Action(action).name
        
        # Take step
        obs, reward, done, info = env.step(action, agent_id=0)
        
        # Log interesting steps
        if step % 10 == 0 or reward > 1.0:
            print(f"\n  Step {step}:")
            print(f"    Action: {action_name}")
            print(f"    Position: {info['agent_position']}")
            print(f"    Reward: {reward:.2f} (breakdown: {info['reward_breakdown']})")
            print(f"    Coverage: {info['coverage_pct']*100:.1f}%")
            print(f"    Frontiers: {info['num_frontiers']}")
        
        step += 1
    
    # Final statistics
    print(f"\n📈 Final Statistics:")
    print(f"  Total steps: {info['steps']}")
    print(f"  Final coverage: {info['coverage_pct']*100:.1f}%")
    print(f"  Total reward: {env.episode_stats['total_reward']:.2f}")
    print(f"  Collisions: {info['collisions']}")
    print(f"  Revisits: {info['revisits']}")


def example2_probabilistic_sensing():
    """Example 2: Deep dive into probabilistic sensor model."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Probabilistic Sensor Model Details")
    print("="*80)
    
    # Create sensor model
    sensor = ProbabilisticSensorModel(
        max_range=4.0,
        detection_prob_at_center=0.95,
        detection_prob_at_edge=0.5,
        false_positive_rate=0.01,
        false_negative_rate=0.05
    )
    
    print(f"\n🔍 Sensor Configuration:")
    print(f"  Max range: {sensor.max_range}")
    print(f"  Detection @ center: {sensor.p_center}")
    print(f"  Detection @ edge: {sensor.p_edge}")
    print(f"  False positive rate: {sensor.false_positive}")
    print(f"  False negative rate: {sensor.false_negative}")
    
    # Detection probability vs distance
    print(f"\n📊 Detection Probability vs Distance:")
    distances = np.linspace(0, 5, 11)
    for dist in distances:
        prob = sensor.get_detection_probability(dist)
        bar = "█" * int(prob * 50)
        print(f"  {dist:3.1f}m: {prob:.3f} {bar}")
    
    # Simulate sensing
    print(f"\n🎲 Simulated Sensing (1000 trials at each distance):")
    for dist in [0.5, 1.5, 2.5, 3.5]:
        detections = 0
        for _ in range(1000):
            detected, _ = sensor.sense_cell(dist, is_obstacle=False)
            if detected:
                detections += 1
        
        theoretical = sensor.get_detection_probability(dist)
        empirical = detections / 1000
        print(f"  {dist:3.1f}m: theoretical={theoretical:.3f}, empirical={empirical:.3f}")
    
    # Sensor footprint visualization
    print(f"\n👁️  Sensor Footprint (20×20 grid, agent at center):")
    grid_size = 20
    agent_x, agent_y = grid_size // 2, grid_size // 2
    
    cells, probs = sensor.get_sensor_footprint(agent_x, agent_y, (grid_size, grid_size))
    
    # Create visualization grid
    vis_grid = np.zeros((grid_size, grid_size))
    for (y, x), prob in zip(cells, probs):
        vis_grid[y, x] = prob
    
    print(f"  Cells in range: {len(cells)}")
    print(f"  Average detection prob: {probs.mean():.3f}")
    print(f"  Footprint area: {len(cells)} cells ({len(cells)/(grid_size**2)*100:.1f}% of grid)")


def example3_action_space():
    """Example 3: Detailed action space analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Action Space and Movement Dynamics")
    print("="*80)
    
    print(f"\n🎯 Action Space (9 discrete actions):")
    for action in Action:
        print(f"  {action.value}: {action.name:12s} → {ACTION_TO_DELTA[action]}")
    
    # Create environment and test actions
    env = CoverageEnvironment(grid_size=10, seed=42)
    obs = env.reset()
    
    agent = env.state.agents[0]
    start_x, start_y = agent.x, agent.y
    
    print(f"\n🚶 Agent Movement Test:")
    print(f"  Starting position: ({start_x}, {start_y})")
    
    # Test each action
    for action in [Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST]:
        # Reset agent to start
        agent.x, agent.y = start_x, start_y
        
        # Take action
        obs, reward, done, info = env.step(action, agent_id=0)
        
        new_x, new_y = agent.x, agent.y
        dx, dy = new_x - start_x, new_y - start_y
        
        print(f"  {action.name:10s}: ({start_x}, {start_y}) → ({new_x}, {new_y}) "
              f"[Δx={dx:+2d}, Δy={dy:+2d}]")
    
    # Valid action masking
    print(f"\n🚫 Valid Action Masking:")
    
    # Place agent near corner
    agent.x, agent.y = 0, 0
    valid_actions = env.get_valid_actions(agent_id=0)
    
    print(f"  Agent at corner (0, 0):")
    for action in Action:
        status = "✓" if valid_actions[action] else "✗"
        print(f"    {status} {action.name:12s} {'(valid)' if valid_actions[action] else '(invalid - out of bounds)'}")


def example4_reward_breakdown():
    """Example 4: Detailed reward function analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Reward Function Breakdown")
    print("="*80)
    
    # Create reward function
    reward_fn = RewardFunction(
        coverage_reward=10.0,
        revisit_penalty=-0.5,
        collision_penalty=-5.0,
        step_penalty=-0.01,
        frontier_bonus=2.0
    )
    
    print(f"\n💰 Reward Configuration:")
    print(f"  Coverage reward: {reward_fn.coverage_reward}")
    print(f"  Revisit penalty: {reward_fn.revisit_penalty}")
    print(f"  Collision penalty: {reward_fn.collision_penalty}")
    print(f"  Step penalty: {reward_fn.step_penalty}")
    print(f"  Frontier bonus: {reward_fn.frontier_bonus}")
    
    # Run episode and track rewards
    env = CoverageEnvironment(grid_size=15, seed=42)
    obs = env.reset()
    
    print(f"\n📊 Reward Components Over Episode:")
    print(f"  {'Step':>4} {'Action':>12} {'Total':>7} {'Coverage':>9} {'Frontier':>9} {'Revisit':>8} {'Collision':>10}")
    print(f"  {'-'*4} {'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*10}")
    
    for step in range(20):
        valid_actions = env.get_valid_actions(agent_id=0)
        action = np.random.choice(np.where(valid_actions)[0])
        
        obs, reward, done, info = env.step(action, agent_id=0)
        
        breakdown = info['reward_breakdown']
        print(f"  {step:4d} {Action(action).name:>12s} {reward:7.2f} "
              f"{breakdown.get('coverage', 0):9.2f} "
              f"{breakdown.get('frontier', 0):9.2f} "
              f"{breakdown.get('revisit', 0):8.2f} "
              f"{breakdown.get('collision', 0):10.2f}")
        
        if done:
            break


def example5_frontier_detection():
    """Example 5: Frontier detection and exploration."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Frontier Detection and Exploration Dynamics")
    print("="*80)
    
    # Create environment
    env = CoverageEnvironment(grid_size=20, sensor_range=3.0, seed=42)
    obs = env.reset()
    
    detector = FrontierDetector()
    
    print(f"\n🌍 Initial Frontiers:")
    print(f"  Total frontier cells: {len(env.state.frontiers)}")
    
    # Cluster frontiers
    clusters = detector.get_frontier_clusters(env.state.frontiers, max_cluster_distance=3.0)
    print(f"  Frontier clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters[:5]):  # Show first 5
        center_y = np.mean([y for y, x in cluster])
        center_x = np.mean([x for y, x in cluster])
        print(f"    Cluster {i+1}: {len(cluster)} cells, center ≈ ({center_x:.1f}, {center_y:.1f})")
    
    # Track frontier evolution
    print(f"\n📈 Frontier Evolution Over Time:")
    print(f"  {'Step':>4} {'Frontiers':>10} {'Clusters':>9} {'Coverage':>9}")
    print(f"  {'-'*4} {'-'*10} {'-'*9} {'-'*9}")
    
    for step in range(0, 50, 5):
        # Take random actions
        for _ in range(5):
            valid_actions = env.get_valid_actions(agent_id=0)
            action = np.random.choice(np.where(valid_actions)[0])
            obs, reward, done, info = env.step(action, agent_id=0)
            
            if done:
                break
        
        clusters = detector.get_frontier_clusters(env.state.frontiers)
        print(f"  {step:4d} {len(env.state.frontiers):10d} {len(clusters):9d} {info['coverage_pct']*100:8.1f}%")


def example6_state_encoding():
    """Example 6: State encoding and observation details."""
    print("\n" + "="*80)
    print("EXAMPLE 6: State Encoding and Observation Structure")
    print("="*80)
    
    # Create environment
    env = CoverageEnvironment(grid_size=15, seed=42)
    obs = env.reset()
    
    print(f"\n📦 Observation Structure:")
    print(f"  Shape: {obs.shape} → [channels, height, width]")
    print(f"  Dtype: {obs.dtype}")
    
    # Analyze each channel
    channel_names = ['Visited', 'Coverage', 'Agents', 'Frontiers', 'Obstacles']
    
    for i, name in enumerate(channel_names):
        channel = obs[i]
        print(f"\n  Channel {i}: {name}")
        print(f"    Shape: {channel.shape}")
        print(f"    Range: [{channel.min():.3f}, {channel.max():.3f}]")
        print(f"    Mean: {channel.mean():.3f}")
        print(f"    Std: {channel.std():.3f}")
        print(f"    Sparsity: {(channel == 0).sum() / channel.size * 100:.1f}% zeros")
        
        if name == 'Visited':
            print(f"    Visited cells: {channel.sum():.0f}")
        elif name == 'Coverage':
            print(f"    Covered cells (>0.5): {(channel > 0.5).sum():.0f}")
        elif name == 'Agents':
            print(f"    Agent positions: {(channel > 0).sum():.0f}")
        elif name == 'Frontiers':
            print(f"    Frontier cells: {channel.sum():.0f}")
        elif name == 'Obstacles':
            print(f"    Obstacle cells: {channel.sum():.0f}")
    
    # Show how observation changes over time
    print(f"\n📊 Observation Evolution (first 10 steps):")
    
    for step in range(10):
        valid_actions = env.get_valid_actions(agent_id=0)
        action = np.random.choice(np.where(valid_actions)[0])
        obs, reward, done, info = env.step(action, agent_id=0)
        
        visited = obs[0].sum()
        coverage = (obs[1] > 0.5).sum()
        frontiers = obs[3].sum()
        
        print(f"  Step {step}: visited={visited:.0f}, covered={coverage:.0f}, frontiers={frontiers:.0f}")


def example7_coverage_dynamics():
    """Example 7: Coverage probability dynamics and confidence."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Coverage Probability and Confidence Dynamics")
    print("="*80)
    
    # Create environment
    env = CoverageEnvironment(grid_size=12, sensor_range=4.0, seed=42)
    obs = env.reset()
    
    agent = env.state.agents[0]
    
    print(f"\n📍 Agent starting at: ({agent.x}, {agent.y})")
    print(f"\n📊 Coverage Probability Around Agent:")
    
    # Show coverage probabilities in local neighborhood
    y_min = max(0, agent.y - 5)
    y_max = min(env.grid_size, agent.y + 6)
    x_min = max(0, agent.x - 5)
    x_max = min(env.grid_size, agent.x + 6)
    
    local_coverage = env.state.coverage[y_min:y_max, x_min:x_max]
    local_confidence = env.state.coverage_confidence[y_min:y_max, x_min:x_max]
    
    print(f"  Coverage probabilities (11×11 window):")
    for row in local_coverage:
        print(f"    " + " ".join(f"{p:.1f}" for p in row))
    
    print(f"\n  Coverage confidence (11×11 window):")
    for row in local_confidence:
        print(f"    " + " ".join(f"{c:.1f}" for c in row))
    
    # Track coverage evolution at a specific cell
    print(f"\n📈 Coverage Evolution at Fixed Cell (5, 5):")
    print(f"  {'Step':>4} {'Distance':>9} {'Coverage':>9} {'Confidence':>11}")
    print(f"  {'-'*4} {'-'*9} {'-'*9} {'-'*11}")
    
    track_y, track_x = 5, 5
    
    for step in range(30):
        # Get distance to tracked cell
        distance = np.sqrt((agent.x - track_x)**2 + (agent.y - track_y)**2)
        coverage_val = env.state.coverage[track_y, track_x]
        confidence_val = env.state.coverage_confidence[track_y, track_x]
        
        if step % 3 == 0:
            print(f"  {step:4d} {distance:9.2f} {coverage_val:9.3f} {confidence_val:11.3f}")
        
        # Take action toward tracked cell
        dx = track_x - agent.x
        dy = track_y - agent.y
        
        if abs(dx) > abs(dy):
            action = Action.EAST if dx > 0 else Action.WEST
        else:
            action = Action.SOUTH if dy > 0 else Action.NORTH
        
        # Execute
        valid_actions = env.get_valid_actions(agent_id=0)
        if not valid_actions[action]:
            action = np.random.choice(np.where(valid_actions)[0])
        
        obs, reward, done, info = env.step(action, agent_id=0)
        
        if done:
            break


def example8_performance_comparison():
    """Example 8: Performance across different grid sizes."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Performance Across Grid Sizes (Scale Invariance Test)")
    print("="*80)
    
    grid_sizes = [10, 15, 20, 25, 30]
    results = []
    
    print(f"\n🔍 Testing grid sizes: {grid_sizes}")
    print(f"\n  {'Size':>4} {'Steps':>6} {'Coverage':>9} {'Reward':>8} {'Collisions':>11} {'Revisits':>9}")
    print(f"  {'-'*4} {'-'*6} {'-'*9} {'-'*8} {'-'*11} {'-'*9}")
    
    for size in grid_sizes:
        # Create environment with scaled sensor range
        env = CoverageEnvironment(
            grid_size=size,
            sensor_range=size * 0.2,  # Scale sensor range
            max_steps=size * 10,  # Scale max steps
            seed=42
        )
        
        obs = env.reset()
        done = False
        
        # Run episode
        while not done:
            valid_actions = env.get_valid_actions(agent_id=0)
            action = np.random.choice(np.where(valid_actions)[0])
            obs, reward, done, info = env.step(action, agent_id=0)
        
        # Record results
        results.append({
            'size': size,
            'steps': info['steps'],
            'coverage': info['coverage_pct'],
            'reward': env.episode_stats['total_reward'],
            'collisions': info['collisions'],
            'revisits': info['revisits']
        })
        
        print(f"  {size:4d} {info['steps']:6d} {info['coverage_pct']*100:8.1f}% "
              f"{env.episode_stats['total_reward']:8.1f} {info['collisions']:11d} {info['revisits']:9d}")
    
    # Analyze degradation
    print(f"\n📉 Coverage Degradation Analysis:")
    base_coverage = results[2]['coverage']  # 20×20 as base
    for r in results:
        degradation = (r['coverage'] - base_coverage) / base_coverage * 100
        print(f"  {r['size']:2d}×{r['size']:2d}: {r['coverage']*100:5.1f}% "
              f"({degradation:+.1f}% vs 20×20 base)")


if __name__ == "__main__":
    """Run all examples."""
    
    # Import here to avoid issues
    from src.environment.coverage_env import ACTION_TO_DELTA
    
    print("\n" + "="*80)
    print("COMPLETE ENVIRONMENT EXAMPLES")
    print("Demonstrating all the nitty-gritty details of the coverage environment")
    print("="*80)
    
    examples = [
        ("Basic Episode", example1_basic_episode),
        ("Probabilistic Sensing", example2_probabilistic_sensing),
        ("Action Space", example3_action_space),
        ("Reward Breakdown", example4_reward_breakdown),
        ("Frontier Detection", example5_frontier_detection),
        ("State Encoding", example6_state_encoding),
        ("Coverage Dynamics", example7_coverage_dynamics),
        ("Performance Comparison", example8_performance_comparison),
    ]
    
    for name, example_fn in examples:
        try:
            example_fn()
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*80)
    
    print("\n🎉 All examples completed!")
