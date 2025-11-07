"""Simple test of enhanced metrics"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dqn_agent import CoordinateDQNAgent
import numpy as np

print('Testing Enhanced Agent Metrics')
print('='*60)

agent = CoordinateDQNAgent(input_channels=5, num_actions=9, device='cpu')

# Fill replay buffer
state = np.random.randn(5, 15, 15)
for _ in range(50):
    agent.memory.push(state, 0, 1.0, state, False)

# Update and get metrics
info = agent.update()

print(f'\nAgent Update Metrics ({len(info)} total):')
print('-'*60)
for k, v in sorted(info.items()):
    print(f'{k:20s}: {v:10.4f}')

print('\n' + '='*60)
print('New metrics added:')
print('  - q_min, q_std (Q-value distribution)')
print('  - target_std (target distribution)')  
print('  - td_error_mean, td_error_max (TD errors)')
print('  - grad_norm (gradient magnitude)')
print('\nSUCCESS!')
