"""
Example: Minimal training and testing script

Demonstrates basic usage of the Coordinate MLP architecture.
"""

import numpy as np
import torch

# Import core components
from src.models.coordinate_network import CoordinateCoverageNetwork
from src.agent.dqn_agent import CoordinateDQNAgent
from src.config import get_default_config


def example_1_network_forward_pass():
    """Example 1: Basic network forward pass."""
    print("="*70)
    print("Example 1: Network Forward Pass")
    print("="*70)
    
    # Create network
    network = CoordinateCoverageNetwork(
        input_channels=5,
        num_freq_bands=6,
        hidden_dim=256,
        num_actions=9
    )
    
    print(f"\nNetwork created with {network.get_num_parameters():,} parameters")
    
    # Create sample input
    batch_size = 4
    grid_size = 20
    grid = torch.randn(batch_size, 5, grid_size, grid_size)
    
    print(f"Input shape: {grid.shape}")
    
    # Forward pass
    q_values = network(grid)
    
    print(f"Q-values shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0].tolist()}")
    
    # Test on different grid size
    grid_large = torch.randn(batch_size, 5, 30, 30)
    q_values_large = network(grid_large)
    
    print(f"\nLarger grid (30×30) Q-values shape: {q_values_large.shape}")
    print("✓ Network handles different grid sizes!")


def example_2_agent_action_selection():
    """Example 2: Agent action selection."""
    print("\n" + "="*70)
    print("Example 2: Agent Action Selection")
    print("="*70)
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=5,
        num_actions=9,
        hidden_dim=256,
        device='cpu'
    )
    
    print(f"\nAgent created on {agent.device}")
    
    # Create sample state
    state = np.random.randn(5, 20, 20)
    
    # Greedy action selection
    action_greedy = agent.select_action(state, epsilon=0.0)
    print(f"\nGreedy action: {action_greedy}")
    
    # Random action selection
    action_random = agent.select_action(state, epsilon=1.0)
    print(f"Random action: {action_random}")
    
    # With valid actions mask
    valid_actions = np.array([True, False, False, True, True, False, False, False, True])
    action_masked = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
    print(f"\nValid actions: {np.where(valid_actions)[0].tolist()}")
    print(f"Selected action: {action_masked}")
    print(f"✓ Action is valid: {valid_actions[action_masked]}")


def example_3_training_episode():
    """Example 3: Training episode (mock environment)."""
    print("\n" + "="*70)
    print("Example 3: Training Episode")
    print("="*70)
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=5,
        num_actions=9,
        hidden_dim=128,  # Smaller for faster demo
        batch_size=16,
        device='cpu'
    )
    
    # Mock environment
    class MockEnv:
        def __init__(self):
            self.step_count = 0
            self.max_steps = 100
            
        def reset(self):
            self.step_count = 0
            return np.random.randn(5, 20, 20)
        
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(5, 20, 20)
            reward = np.random.randn()
            done = self.step_count >= self.max_steps
            info = {'coverage': np.random.rand()}
            return next_state, reward, done, info
    
    env = MockEnv()
    
    # Run episode
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    print("\nRunning training episode...")
    
    while True:
        # Select action
        action = agent.select_action(state, epsilon=0.3)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store in replay buffer
        agent.memory.push(state, action, reward, next_state, done)
        
        # Update (if enough samples)
        if len(agent.memory) >= agent.batch_size:
            update_info = agent.update()
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if done:
            break
    
    print(f"\nEpisode complete:")
    print(f"  Steps: {episode_steps}")
    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Coverage: {info['coverage']*100:.1f}%")
    print(f"  Memory size: {len(agent.memory)}")
    print(f"  Epsilon: {agent.epsilon:.3f}")


def example_4_multi_scale_testing():
    """Example 4: Test on multiple grid sizes."""
    print("\n" + "="*70)
    print("Example 4: Multi-Scale Testing")
    print("="*70)
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=5,
        num_actions=9,
        hidden_dim=256,
        device='cpu'
    )
    
    print("\nTesting on different grid sizes:")
    print("-"*50)
    
    test_sizes = [15, 20, 25, 30, 35, 40]
    
    for size in test_sizes:
        # Create state for this grid size
        state = np.random.randn(5, size, size)
        
        # Select action
        action = agent.select_action(state, epsilon=0.0)
        
        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.policy_net(state_tensor)
            max_q = q_values.max().item()
        
        print(f"  {size:2d}×{size:2d}: action={action}, max_q={max_q:7.3f} ✓")
    
    print("\n✓ Agent handles all grid sizes!")


def example_5_attention_visualization():
    """Example 5: Visualize attention weights."""
    print("\n" + "="*70)
    print("Example 5: Attention Weights")
    print("="*70)
    
    # Create network
    network = CoordinateCoverageNetwork(
        input_channels=5,
        hidden_dim=256,
        num_actions=9
    )
    
    # Create sample input
    grid = torch.randn(1, 5, 20, 20)
    
    # Forward pass with attention
    q_values, attention = network(grid, return_attention=True)
    
    print(f"\nQ-values shape: {q_values.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Average attention over heads
    avg_attention = attention.mean(dim=1).squeeze()  # [H*W]
    
    # Find top attended cells
    top_k = 5
    top_values, top_indices = torch.topk(avg_attention, k=top_k)
    
    print(f"\nTop {top_k} attended cells:")
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        y = idx.item() // 20
        x = idx.item() % 20
        print(f"  {i+1}. Cell ({x:2d}, {y:2d}): weight={val.item():.4f}")
    
    print("\n✓ Network attends to different cells!")


def example_6_save_and_load():
    """Example 6: Save and load agent."""
    print("\n" + "="*70)
    print("Example 6: Save and Load Agent")
    print("="*70)
    
    import tempfile
    import os
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=5,
        num_actions=9,
        hidden_dim=128,
        device='cpu'
    )
    
    # Modify some state
    agent.epsilon = 0.123
    agent.training_steps = 999
    
    print(f"\nOriginal agent:")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Training steps: {agent.training_steps}")
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'agent.pt')
        agent.save(save_path)
        print(f"\n✓ Agent saved to: {save_path}")
        
        # Create new agent
        new_agent = CoordinateDQNAgent(
            input_channels=5,
            num_actions=9,
            hidden_dim=128,
            device='cpu'
        )
        
        # Load checkpoint
        new_agent.load(save_path)
        print(f"✓ Agent loaded from: {save_path}")
        
        print(f"\nLoaded agent:")
        print(f"  Epsilon: {new_agent.epsilon}")
        print(f"  Training steps: {new_agent.training_steps}")
        
        # Verify
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.training_steps == agent.training_steps
        print("\n✓ Save/load successful!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Coordinate MLP Coverage - Usage Examples")
    print("="*70 + "\n")
    
    examples = [
        example_1_network_forward_pass,
        example_2_agent_action_selection,
        example_3_training_episode,
        example_4_multi_scale_testing,
        example_5_attention_visualization,
        example_6_save_and_load,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Example failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
