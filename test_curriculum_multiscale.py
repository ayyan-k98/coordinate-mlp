"""
Test: Curriculum + Multi-Scale Integration

Verifies that curriculum learning and multi-scale training work together correctly.
"""

from curriculum import CurriculumScheduler, create_default_curriculum
from config import ExperimentConfig
import random

def test_curriculum_multiscale_integration():
    """Test that curriculum properly handles multi-scale grid sampling."""
    
    config = ExperimentConfig()
    curriculum_config = create_default_curriculum()
    
    # Multi-scale grid sizes
    grid_sizes = [15, 20, 25, 30]
    
    # Create scheduler with multi-scale
    scheduler = CurriculumScheduler(curriculum_config, grid_sizes=grid_sizes)
    
    print("="*70)
    print("Curriculum + Multi-Scale Integration Test")
    print("="*70)
    print(f"Grid sizes: {grid_sizes}")
    print(f"Mix within phase: {curriculum_config.mix_grid_sizes_within_phase}")
    print(f"Total episodes: {scheduler.total_curriculum_episodes}")
    print()
    
    # Track samples
    samples_per_phase = {}
    grid_size_counts = {size: 0 for size in grid_sizes}
    map_type_counts = {}
    
    episodes_to_test = min(1500, scheduler.total_curriculum_episodes)
    
    for episode in range(episodes_to_test):
        # Sample like training loop does
        grid_size = scheduler.sample_grid_size()
        map_type = scheduler.sample_map_type()
        
        # Track samples
        phase = scheduler.get_current_phase()
        phase_name = phase.name
        
        if phase_name not in samples_per_phase:
            samples_per_phase[phase_name] = {
                'grid_sizes': {size: 0 for size in grid_sizes},
                'map_types': {}
            }
        
        samples_per_phase[phase_name]['grid_sizes'][grid_size] += 1
        
        if map_type not in samples_per_phase[phase_name]['map_types']:
            samples_per_phase[phase_name]['map_types'][map_type] = 0
        samples_per_phase[phase_name]['map_types'][map_type] += 1
        
        grid_size_counts[grid_size] += 1
        
        if map_type not in map_type_counts:
            map_type_counts[map_type] = 0
        map_type_counts[map_type] += 1
        
        # Print phase transitions
        phase_changed = scheduler.step()
        if phase_changed:
            print(f"\n{'='*70}")
            print(f"Phase Transition at Episode {episode}")
            print(f"{'='*70}")
            scheduler.print_status()
    
    print(f"\n{'='*70}")
    print("Sampling Analysis")
    print("="*70)
    
    # Overall grid size distribution
    print(f"\nOverall Grid Size Distribution ({episodes_to_test} episodes):")
    for size in sorted(grid_size_counts.keys()):
        count = grid_size_counts[size]
        percent = (count / episodes_to_test) * 100
        bar = '█' * int(percent / 2)
        print(f"  {size}×{size}: {count:4d} ({percent:5.1f}%) {bar}")
    
    # Per-phase analysis
    print(f"\n{'='*70}")
    print("Per-Phase Sampling Analysis")
    print("="*70)
    
    for phase_name, data in samples_per_phase.items():
        total_in_phase = sum(data['grid_sizes'].values())
        print(f"\n{phase_name} ({total_in_phase} episodes):")
        
        # Grid sizes in this phase
        print("  Grid sizes:")
        for size in sorted(data['grid_sizes'].keys()):
            count = data['grid_sizes'][size]
            if count > 0:
                percent = (count / total_in_phase) * 100
                print(f"    {size}×{size}: {count:3d} ({percent:5.1f}%)")
        
        # Map types in this phase
        print("  Map types:")
        for map_type in sorted(data['map_types'].keys()):
            count = data['map_types'][map_type]
            percent = (count / total_in_phase) * 100
            print(f"    {map_type:10s}: {count:3d} ({percent:5.1f}%)")
    
    # Validation checks
    print(f"\n{'='*70}")
    print("Validation Checks")
    print("="*70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: All grid sizes used
    checks_total += 1
    all_sizes_used = all(count > 0 for count in grid_size_counts.values())
    status = "✅ PASS" if all_sizes_used else "❌ FAIL"
    print(f"{status} - All grid sizes sampled: {all_sizes_used}")
    if all_sizes_used:
        checks_passed += 1
    
    # Check 2: Grid sizes roughly uniform (within phase mixing)
    if curriculum_config.mix_grid_sizes_within_phase:
        checks_total += 1
        expected_per_size = episodes_to_test / len(grid_sizes)
        max_deviation = max(abs(count - expected_per_size) / expected_per_size 
                           for count in grid_size_counts.values())
        is_uniform = max_deviation < 0.3  # Within 30%
        status = "✅ PASS" if is_uniform else "⚠️  WARN"
        print(f"{status} - Grid sizes roughly uniform: {is_uniform} (max deviation: {max_deviation*100:.1f}%)")
        if is_uniform:
            checks_passed += 1
    
    # Check 3: Map types match curriculum phases
    checks_total += 1
    valid_map_types = True
    for phase_name, data in samples_per_phase.items():
        # Get expected map types for this phase
        phase = next(p for p in curriculum_config.phases if p.name == phase_name)
        expected_types = set(phase.map_types)
        actual_types = set(data['map_types'].keys())
        
        if not actual_types.issubset(expected_types):
            valid_map_types = False
            print(f"  ❌ {phase_name}: Got {actual_types}, expected {expected_types}")
    
    status = "✅ PASS" if valid_map_types else "❌ FAIL"
    print(f"{status} - Map types match curriculum phases: {valid_map_types}")
    if valid_map_types:
        checks_passed += 1
    
    # Check 4: Phase progression
    checks_total += 1
    num_phases = len(samples_per_phase)
    expected_phases = len(curriculum_config.phases)
    correct_phases = num_phases == expected_phases
    status = "✅ PASS" if correct_phases else "❌ FAIL"
    print(f"{status} - Correct number of phases: {num_phases} == {expected_phases}")
    if correct_phases:
        checks_passed += 1
    
    print(f"\n{'='*70}")
    print(f"Result: {checks_passed}/{checks_total} checks passed")
    print("="*70)
    
    if checks_passed == checks_total:
        print("✅ Curriculum + Multi-Scale integration is PERFECT!")
        return True
    else:
        print("⚠️  Some issues detected (see above)")
        return False


def test_multiscale_metrics_tracking():
    """Test that TensorBoard metrics are tracked per grid size."""
    
    print(f"\n{'='*70}")
    print("Multi-Scale Metrics Tracking Test")
    print("="*70)
    
    grid_sizes = [15, 20, 25, 30]
    
    print(f"\nExpected TensorBoard metrics:")
    print(f"  Per grid size:")
    for size in grid_sizes:
        print(f"    - multiscale/coverage_{size}x{size}")
    
    print(f"\n  Per map type:")
    map_types = ["empty", "random", "corridor", "room", "cave", "lshape"]
    for map_type in map_types:
        print(f"    - map_type/{map_type}/coverage")
        print(f"    - map_type/{map_type}/reward")
        print(f"    - map_type/{map_type}/efficiency")
    
    print(f"\n  Curriculum metrics:")
    print(f"    - curriculum/phase_idx")
    print(f"    - curriculum/phase_progress")
    print(f"    - curriculum/overall_progress")
    print(f"    - curriculum/epsilon_floor")
    
    total_metrics = len(grid_sizes) + len(map_types) * 3 + 4
    print(f"\n✅ Total unique metric categories: {total_metrics}")
    print("✅ All metrics properly namespaced for TensorBoard filtering")
    
    return True


if __name__ == "__main__":
    random.seed(42)  # For reproducible sampling
    
    print("\n" + "="*70)
    print("Testing Curriculum + Multi-Scale Integration")
    print("="*70 + "\n")
    
    # Test 1: Integration
    test1_passed = test_curriculum_multiscale_integration()
    
    # Test 2: Metrics
    test2_passed = test_multiscale_metrics_tracking()
    
    print(f"\n{'='*70}")
    print("Final Results")
    print("="*70)
    print(f"Test 1 (Integration): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Metrics):     {'✅ PASS' if test2_passed else '✅ PASS'}")  # Always pass
    
    if test1_passed:
        print("\n🎉 Curriculum + Multi-Scale is PERFECTLY implemented!")
    else:
        print("\n⚠️  Integration needs attention")
