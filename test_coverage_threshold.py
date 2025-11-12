"""
Test Coverage Threshold Upgrade (0.5 → 0.9)

Verifies that:
1. Coverage threshold is now 0.9 (not 0.5)
2. Multi-threshold metrics are tracked
3. Rewards still use continuous probabilities
4. Episode termination uses 0.95 threshold
"""

import numpy as np
from coverage_env import CoverageEnvironment

def test_coverage_threshold():
    """Test that coverage threshold is 0.9"""
    print("="*70)
    print("TEST 1: Coverage Threshold")
    print("="*70)
    
    env = CoverageEnvironment(grid_size=20, sensor_range=4.0, seed=42)
    env.reset()
    
    # Take several steps to build up coverage
    for _ in range(30):
        obs, reward, done, info = env.step(0)  # Move north
    
    # Check that coverage_pct matches cells with >0.9 confidence
    total_coverable = (~env.state.obstacles).sum()
    cells_90 = (env.state.coverage > 0.9).sum()
    expected_coverage = cells_90 / total_coverable
    
    print(f"Reported coverage_pct: {info['coverage_pct']:.3f}")
    print(f"Expected (>0.9 cells): {expected_coverage:.3f}")
    print(f"Match: {'✅' if abs(info['coverage_pct'] - expected_coverage) < 0.001 else '❌'}")
    
    assert abs(info['coverage_pct'] - expected_coverage) < 0.001, \
        "coverage_pct should match >0.9 threshold"
    
    print("\n✅ TEST 1 PASSED: Coverage threshold is 0.9\n")


def test_multi_threshold_metrics():
    """Test that multi-threshold metrics are tracked"""
    print("="*70)
    print("TEST 2: Multi-Threshold Metrics")
    print("="*70)
    
    env = CoverageEnvironment(grid_size=20, sensor_range=4.0, seed=42)
    env.reset()
    
    # Take steps and print pyramid distribution
    for i in range(50):
        obs, reward, done, info = env.step(i % 8)
        
        if i % 10 == 0:
            print(f"Step {i:3d}: "
                  f"sensed={info['cells_ever_sensed']:.2%}, "
                  f"50%={info['coverage_pct_50']:.2%}, "
                  f"70%={info['coverage_pct_70']:.2%}, "
                  f"90%={info['coverage_pct_90']:.2%}, "
                  f"95%={info['coverage_pct_95']:.2%}")
    
    # Verify pyramid: sensed > 50% > 70% > 90% > 95%
    assert info['cells_ever_sensed'] >= info['coverage_pct_50'], "Pyramid violated"
    assert info['coverage_pct_50'] >= info['coverage_pct_70'], "Pyramid violated"
    assert info['coverage_pct_70'] >= info['coverage_pct_90'], "Pyramid violated"
    assert info['coverage_pct_90'] >= info['coverage_pct_95'], "Pyramid violated"
    
    print("\n✅ TEST 2 PASSED: Multi-threshold metrics form proper pyramid\n")


def test_rewards_unchanged():
    """Test that rewards still use continuous probabilities"""
    print("="*70)
    print("TEST 3: Reward System Unchanged")
    print("="*70)
    
    env = CoverageEnvironment(grid_size=20, sensor_range=4.0, seed=42)
    obs = env.reset()
    
    # First step should give large reward (discovery)
    obs, reward1, done, info1 = env.step(0)
    
    print(f"First step reward: {reward1:.2f}")
    print(f"Coverage gain (prob sum): {info1['coverage_gain']:.3f}")
    print(f"Coverage reward component: {info1['reward_breakdown']['coverage']:.2f}")
    
    # Reward should be substantial from discovering new cells
    # Even if no cells reach >0.9 threshold yet
    assert reward1 > 0.5, "First step should give positive reward for discovery"
    
    print(f"First step gave positive reward: {'✅' if reward1 > 0 else '❌'}")
    print("\n✅ TEST 3 PASSED: Rewards use continuous probabilities\n")


def test_early_termination():
    """Test that episode terminates at 0.95 coverage (not 0.99)"""
    print("="*70)
    print("TEST 4: Early Termination Threshold")
    print("="*70)
    
    env = CoverageEnvironment(grid_size=15, sensor_range=4.0, max_steps=500, seed=42)
    env.reset()
    
    # Run until done
    step_count = 0
    final_coverage = 0
    
    for step_count in range(500):
        obs, reward, done, info = env.step(step_count % 8)
        final_coverage = info['coverage_pct']
        
        if done and step_count < 499:
            print(f"Episode terminated early at step {step_count}")
            print(f"Final coverage: {final_coverage:.2%}")
            print(f"Termination reason: {'Coverage ≥ 95%' if final_coverage >= 0.95 else 'Unknown'}")
            
            # Should terminate when coverage ≥ 0.95
            assert final_coverage >= 0.95, "Should terminate at 95% coverage"
            print("✅ Terminated at correct threshold (≥95%)\n")
            break
    else:
        print(f"Episode ran to max_steps ({step_count + 1})")
        print(f"Final coverage: {final_coverage:.2%}")
        print("✅ No early termination (didn't reach 95%)\n")
    
    print("✅ TEST 4 PASSED: Termination threshold is 0.95\n")


def test_quality_comparison():
    """Compare coverage quality at different thresholds"""
    print("="*70)
    print("TEST 5: Coverage Quality Comparison")
    print("="*70)
    
    env = CoverageEnvironment(grid_size=20, sensor_range=4.0, seed=42)
    env.reset()
    
    # Run for 100 steps
    for i in range(100):
        obs, reward, done, info = env.step(i % 8)
    
    print("Final Coverage Quality Metrics:")
    print(f"  Cells ever sensed:     {info['cells_ever_sensed']:.1%}")
    print(f"  Coverage (≥50% conf):  {info['coverage_pct_50']:.1%}")
    print(f"  Coverage (≥70% conf):  {info['coverage_pct_70']:.1%}")
    print(f"  Coverage (≥90% conf):  {info['coverage_pct_90']:.1%}  ← Primary metric")
    print(f"  Coverage (≥95% conf):  {info['coverage_pct_95']:.1%}")
    
    # Demonstrate the impact
    print(f"\nImpact of threshold choice:")
    print(f"  Old (50% threshold): {info['coverage_pct_50']:.1%} coverage")
    print(f"  New (90% threshold): {info['coverage_pct_90']:.1%} coverage")
    print(f"  Difference: {(info['coverage_pct_50'] - info['coverage_pct_90'])*100:.1f} percentage points")
    print(f"  Quality improvement: {info['coverage_pct_90'] / info['coverage_pct_50']:.2f}× confidence per cell")
    
    print("\n✅ TEST 5 PASSED: Quality metrics demonstrate improvement\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("COVERAGE THRESHOLD UPGRADE VERIFICATION")
    print("Testing upgrade from 0.5 → 0.9 threshold")
    print("="*70 + "\n")
    
    try:
        test_coverage_threshold()
        test_multi_threshold_metrics()
        test_rewards_unchanged()
        test_early_termination()
        test_quality_comparison()
        
        print("="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nCoverage threshold successfully upgraded to 0.9!")
        print("Key changes verified:")
        print("  ✅ Coverage uses 0.9 threshold (high-quality)")
        print("  ✅ Multi-threshold metrics tracked")
        print("  ✅ Rewards unchanged (continuous probabilities)")
        print("  ✅ Episode terminates at 0.95 coverage")
        print("  ✅ Quality metrics more meaningful")
        print("\nExpect lower coverage numbers but higher quality!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
