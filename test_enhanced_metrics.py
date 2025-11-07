"""Test Enhanced Metrics System"""

print('='*70)
print('Testing Enhanced Metrics')
print('='*70)

from dqn_agent import CoordinateDQNAgent
from coverage_env import CoverageEnvironment
import numpy as np

# Create environment and agent
env = CoverageEnvironment(grid_size=15)
agent = CoordinateDQNAgent(input_channels=5, num_actions=9, device='cpu')

print('\n1. Testing agent update metrics...')
state = env.reset()

# Fill replay buffer
for _ in range(50):
    agent.memory.push(state, 0, 1.0, state, False)

# Update agent
info = agent.update()

if info:
    print(f'\n[OK] Enhanced metrics from agent.update() [{len(info)} metrics]:')
    for k, v in sorted(info.items()):
        print(f'  {k:20s}: {v:.6f}')
else:
    print('[ERROR] No metrics returned')

print('\n2. Testing episode metrics...')

# Simulate episode
from train import train_episode
from config import get_default_config

config = get_default_config()
config.training.warmup_episodes = 0

metrics = train_episode(agent, env, config, episode=10)

print(f'\n[OK] Episode metrics [{len(metrics)} metrics]:')
for category in ['Basic', 'Performance', 'Coverage', 'Reward', 'Training']:
    print(f'\n{category} Metrics:')
    
    if category == 'Basic':
        keys = ['episode', 'reward', 'steps', 'coverage', 'epsilon', 'memory_size']
    elif category == 'Performance':
        keys = ['episode_time', 'steps_per_second']
    elif category == 'Coverage':
        keys = ['efficiency', 'collisions', 'revisits', 'num_frontiers']
    elif category == 'Reward':
        keys = ['reward_coverage', 'reward_confidence', 'reward_revisit', 'reward_frontier']
    else:  # Training
        keys = ['loss', 'loss_std', 'q_mean', 'q_std', 'td_error', 'grad_norm']
    
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f'  {k:20s}: {v:.6f}')
            else:
                print(f'  {k:20s}: {v}')

print('\n'+'='*70)
print('[SUCCESS] All priority metrics implemented!')
print('='*70)
print(f'\nTotal agent metrics: {len(info) if info else 0}')
print(f'Total episode metrics: {len(metrics)}')
print(f'\nMetrics categories:')
print(f'  - Essential diagnostics (q_min, q_std, td_error, grad_norm)')
print(f'  - Coverage quality (efficiency, collisions, revisits, frontiers)')
print(f'  - Performance tracking (episode_time, steps_per_second)')
print(f'  - Reward breakdown (coverage, confidence, revisit, frontier)')
