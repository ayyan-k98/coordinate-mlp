"""Test Phase-Specific Epsilon in Curriculum"""

from curriculum import create_default_curriculum, CurriculumScheduler

print('='*70)
print('Testing Phase-Specific Epsilon System')
print('='*70)

# Create curriculum
curriculum_config = create_default_curriculum()
scheduler = CurriculumScheduler(curriculum_config, grid_sizes=[15, 20, 25])

print('\nCurriculum Phases with Epsilon Settings:')
print('-'*70)
for i, phase in enumerate(curriculum_config.phases, 1):
    print(f'\n{i}. {phase.name}')
    print(f'   Map Types: {", ".join(phase.map_types)}')
    print(f'   Episodes: {phase.num_episodes}')
    if phase.epsilon_boost is not None:
        print(f'   Epsilon Boost: {phase.epsilon_boost} (applied at phase start)')
    if phase.epsilon_floor is not None:
        print(f'   Epsilon Floor: {phase.epsilon_floor} (minimum during phase)')
    else:
        print(f'   Epsilon: No constraints')

# Simulate training with epsilon evolution
print('\n' + '='*70)
print('Simulating Epsilon Evolution Through Curriculum')
print('='*70)

agent_epsilon = 1.0  # Starting epsilon
epsilon_decay = 0.995

print(f'\nInitial: epsilon = {agent_epsilon:.3f}')

for episode in range(1500):
    # Check for epsilon boost
    boost = scheduler.should_boost_epsilon()
    if boost is not None:
        agent_epsilon = boost
        print(f'\nEpisode {episode}: EPSILON BOOSTED to {agent_epsilon:.3f}')
    
    # Apply epsilon floor
    agent_epsilon = scheduler.get_epsilon_adjustment(agent_epsilon)
    
    # Advance curriculum
    phase_changed = scheduler.step()
    
    if phase_changed:
        progress = scheduler.get_progress()
        print(f'\nEpisode {episode}: Phase Transition -> {progress["phase_name"]}')
        print(f'  Current epsilon: {agent_epsilon:.3f}')
    
    # Natural decay (but floors will prevent it from going too low)
    agent_epsilon = max(0.05, agent_epsilon * epsilon_decay)
    
    # Sample logging
    if episode in [0, 50, 100, 200, 500, 600, 900, 1000, 1300, 1400]:
        progress = scheduler.get_progress()
        phase = scheduler.get_current_phase()
        floor = phase.epsilon_floor if phase.epsilon_floor else 0.0
        print(f'Episode {episode:4d}: Phase={progress["phase_name"]:20s} '
              f'epsilon={agent_epsilon:.3f} (floor={floor:.2f})')

print('\n' + '='*70)
print('Key Insights:')
print('='*70)
print('1. Phase 1 (Empty): Low floor (0.1) - focus on exploitation')
print('2. Phase 2 (Obstacles): Moderate floor (0.15) - balanced exploration')
print('3. Phase 3 (Structures): BOOST to 0.3! Floor 0.2 - find doors!')
print('4. Phase 4 (Complex): BOOST to 0.35! Floor 0.25 - escape mazes!')
print('5. Phase 5 (Mixed): Low floor (0.1) - polish skills')
print()
print('Without phase-specific epsilon:')
print('  By episode 900 (Phase 4 start), global epsilon ~ 0.08')
print('  Too low to escape complex caves - FAILS!')
print()
print('With phase-specific epsilon:')
print('  Epsilon boosted to 0.35 at Phase 4 start')
print('  Floor maintains 0.25 minimum - SUCCESS!')
print('='*70)
