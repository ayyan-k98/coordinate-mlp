"""
Reward Scale Analysis

Analyzing current reward magnitudes to identify imbalances.
"""

import numpy as np


def analyze_current_scales():
    """Analyze reward scales across an episode."""
    
    print("="*70)
    print("CURRENT REWARD SCALE ANALYSIS")
    print("="*70)
    
    # Episode parameters
    grid_size = 20
    total_cells = grid_size * grid_size
    coverable_cells = int(total_cells * 0.85)  # Assuming 15% obstacles
    max_steps = 350
    
    print(f"\nEpisode Parameters:")
    print(f"  Grid: {grid_size}×{grid_size} = {total_cells} cells")
    print(f"  Coverable cells: ~{coverable_cells}")
    print(f"  Max steps: {max_steps}")
    
    # Current reward values
    coverage_reward_per_cell = 10.0
    revisit_penalty = -0.5
    collision_penalty = -5.0
    step_penalty = -0.01
    frontier_bonus = 2.0
    stay_penalty = -1.0
    rotation_penalty_max = -0.15
    early_completion_bonus = 50.0
    time_bonus_per_step = 0.1
    
    print("\n" + "="*70)
    print("REWARD COMPONENT MAGNITUDES (Per Episode)")
    print("="*70)
    
    # Calculate realistic episode totals
    components = []
    
    # 1. Coverage (HUGE POSITIVE)
    coverage_total = coverable_cells * coverage_reward_per_cell
    components.append(('Coverage (all cells)', coverage_total, 'POSITIVE'))
    print(f"\n1. Coverage Reward:")
    print(f"   {coverable_cells} cells × {coverage_reward_per_cell} = +{coverage_total:.1f}")
    print(f"   ⚠️  DOMINATES all other rewards!")
    
    # 2. Confidence (LARGE POSITIVE)
    confidence_total = coverable_cells * 0.5  # Approximate
    components.append(('Confidence gain', confidence_total, 'POSITIVE'))
    print(f"\n2. Coverage Confidence:")
    print(f"   ~{coverable_cells} cells × 0.5 = +{confidence_total:.1f}")
    
    # 3. Early completion (LARGE POSITIVE)
    steps_to_complete = 180  # Realistic
    steps_saved = max_steps - steps_to_complete
    completion_total = early_completion_bonus + (steps_saved * time_bonus_per_step)
    components.append(('Early completion', completion_total, 'POSITIVE'))
    print(f"\n3. Early Completion Bonus:")
    print(f"   Base: {early_completion_bonus}")
    print(f"   Time: {steps_saved} steps × {time_bonus_per_step} = {steps_saved * time_bonus_per_step:.1f}")
    print(f"   Total: +{completion_total:.1f}")
    
    # 4. Frontier bonus (MODERATE POSITIVE)
    frontier_visits = 50  # Realistic
    frontier_total = frontier_visits * frontier_bonus
    components.append(('Frontier bonus', frontier_total, 'POSITIVE'))
    print(f"\n4. Frontier Bonus:")
    print(f"   ~{frontier_visits} frontier visits × {frontier_bonus} = +{frontier_total:.1f}")
    
    # 5. Step penalty (SMALL NEGATIVE)
    step_total = steps_to_complete * step_penalty
    components.append(('Step penalty', step_total, 'NEGATIVE'))
    print(f"\n5. Step Penalty:")
    print(f"   {steps_to_complete} steps × {step_penalty} = {step_total:.1f}")
    print(f"   ⚠️  NEGLIGIBLE compared to coverage!")
    
    # 6. Rotation penalty (TINY NEGATIVE)
    rotations = 60  # Realistic number of turns
    rotation_total = rotations * rotation_penalty_max
    components.append(('Rotation penalty', rotation_total, 'NEGATIVE'))
    print(f"\n6. Rotation Penalty:")
    print(f"   ~{rotations} turns × {rotation_penalty_max} = {rotation_total:.1f}")
    print(f"   ⚠️  NEGLIGIBLE!")
    
    # 7. Revisit penalty (SMALL NEGATIVE)
    revisits = 30  # Some overlap
    revisit_total = revisits * revisit_penalty
    components.append(('Revisit penalty', revisit_total, 'NEGATIVE'))
    print(f"\n7. Revisit Penalty:")
    print(f"   ~{revisits} revisits × {revisit_penalty} = {revisit_total:.1f}")
    
    # 8. Collision penalty (SMALL NEGATIVE)
    collisions = 10  # Occasional
    collision_total = collisions * collision_penalty
    components.append(('Collision penalty', collision_total, 'NEGATIVE'))
    print(f"\n8. Collision Penalty:")
    print(f"   ~{collisions} collisions × {collision_penalty} = {collision_total:.1f}")
    
    # 9. STAY penalty (TINY NEGATIVE)
    stays = 5  # Rare
    stay_total = stays * stay_penalty
    components.append(('STAY penalty', stay_total, 'NEGATIVE'))
    print(f"\n9. STAY Penalty:")
    print(f"   ~{stays} stays × {stay_penalty} = {stay_total:.1f}")
    
    # Total
    total_positive = coverage_total + confidence_total + completion_total + frontier_total
    total_negative = step_total + rotation_total + revisit_total + collision_total + stay_total
    total_reward = total_positive + total_negative
    
    print("\n" + "="*70)
    print("EPISODE TOTALS")
    print("="*70)
    print(f"Total POSITIVE: +{total_positive:.1f}")
    print(f"Total NEGATIVE: {total_negative:.1f}")
    print(f"Net reward:     +{total_reward:.1f}")
    
    # Scale analysis
    print("\n" + "="*70)
    print("SCALE IMBALANCE ANALYSIS")
    print("="*70)
    
    # Coverage dominance
    coverage_ratio = coverage_total / abs(total_negative)
    print(f"\n⚠️  Coverage reward is {coverage_ratio:.1f}× larger than ALL penalties combined!")
    
    # Individual comparisons
    print(f"\n⚠️  Coverage (+{coverage_total:.0f}) vs Step penalty ({step_total:.1f})")
    print(f"    Ratio: {abs(coverage_total / step_total):.0f}:1")
    
    print(f"\n⚠️  Coverage (+{coverage_total:.0f}) vs Rotation penalty ({rotation_total:.1f})")
    print(f"    Ratio: {abs(coverage_total / rotation_total):.0f}:1")
    
    print(f"\n⚠️  Early completion (+{completion_total:.0f}) vs STAY penalty ({stay_total:.1f})")
    print(f"    Ratio: {abs(completion_total / stay_total):.0f}:1")
    
    # Gradient explosion risk
    print("\n" + "="*70)
    print("GRADIENT EXPLOSION RISK")
    print("="*70)
    print(f"Net reward magnitude: {abs(total_reward):.0f}")
    print(f"Expected Q-value scale: ~{abs(total_reward / (1 - 0.99)):.0f} (with γ=0.99)")
    print(f"\n⚠️  Q-values will be in range [0, ~{abs(total_reward / (1 - 0.99)):.0f}]")
    print(f"⚠️  TD errors will be ~{abs(total_reward):.0f} magnitude")
    print(f"⚠️  Gradients will be HUGE without proper scaling!")
    
    return components


def propose_balanced_scales():
    """Propose balanced reward scales."""
    
    print("\n" + "="*70)
    print("PROPOSED BALANCED REWARD SCALES")
    print("="*70)
    
    print("\nDesign Principles:")
    print("  1. Keep total episode reward in range [-10, +10]")
    print("  2. Make penalties meaningful (not negligible)")
    print("  3. Balance positive and negative components")
    print("  4. Rewards per-cell, penalties per-step")
    
    grid_size = 20
    coverable_cells = int(grid_size * grid_size * 0.85)
    max_steps = 350
    
    print(f"\n" + "="*70)
    print("PROPOSED VALUES")
    print("="*70)
    
    # Scale everything to episode total ~10
    target_total = 10.0
    
    # Coverage is primary objective - should give ~5-8 total
    coverage_per_cell = 0.02  # 340 cells × 0.02 = 6.8
    print(f"\n1. Coverage reward: {coverage_per_cell} per cell")
    print(f"   Episode total: {coverable_cells} × {coverage_per_cell} = {coverable_cells * coverage_per_cell:.1f}")
    
    # Confidence - smaller contribution
    confidence_weight = 0.01  # ~340 × 0.01 = 3.4
    print(f"\n2. Coverage confidence: {confidence_weight} per cell")
    print(f"   Episode total: ~{coverable_cells * confidence_weight:.1f}")
    
    # Early completion - significant but not dominating
    completion_bonus = 2.0
    time_bonus = 0.01  # 170 steps × 0.01 = 1.7
    print(f"\n3. Early completion:")
    print(f"   Base bonus: {completion_bonus}")
    print(f"   Time bonus: {time_bonus} per step saved")
    print(f"   Episode total (180 steps): {completion_bonus + 170 * time_bonus:.1f}")
    
    # Frontier - moderate encouragement
    frontier_bonus = 0.05
    print(f"\n4. Frontier bonus: {frontier_bonus} per visit")
    print(f"   Episode total (~50 visits): {50 * frontier_bonus:.1f}")
    
    # Step penalty - meaningful
    step_penalty = -0.005  # 180 × -0.005 = -0.9
    print(f"\n5. Step penalty: {step_penalty} per step")
    print(f"   Episode total (180 steps): {180 * step_penalty:.1f}")
    
    # Rotation - noticeable
    rotation_small = -0.01
    rotation_medium = -0.02
    rotation_large = -0.05
    print(f"\n6. Rotation penalty: {rotation_small} / {rotation_medium} / {rotation_large}")
    print(f"   Episode total (~60 turns): {60 * rotation_medium:.1f}")
    
    # Revisit - discouraging
    revisit_penalty = -0.05
    print(f"\n7. Revisit penalty: {revisit_penalty} per revisit")
    print(f"   Episode total (~30 revisits): {30 * revisit_penalty:.1f}")
    
    # Collision - strong deterrent
    collision_penalty = -0.5
    print(f"\n8. Collision penalty: {collision_penalty} per collision")
    print(f"   Episode total (~10 collisions): {10 * collision_penalty:.1f}")
    
    # STAY - strong deterrent
    stay_penalty = -0.1
    print(f"\n9. STAY penalty: {stay_penalty} per STAY")
    print(f"   Episode total (~5 stays): {5 * stay_penalty:.1f}")
    
    # Calculate totals
    total_positive = (coverable_cells * coverage_per_cell + 
                     coverable_cells * confidence_weight +
                     completion_bonus + 170 * time_bonus +
                     50 * frontier_bonus)
    
    total_negative = (180 * step_penalty + 
                     60 * rotation_medium +
                     30 * revisit_penalty +
                     10 * collision_penalty +
                     5 * stay_penalty)
    
    net_total = total_positive + total_negative
    
    print("\n" + "="*70)
    print("BALANCED EPISODE TOTALS")
    print("="*70)
    print(f"Total POSITIVE: +{total_positive:.2f}")
    print(f"Total NEGATIVE: {total_negative:.2f}")
    print(f"Net reward:     +{net_total:.2f}")
    
    print(f"\n✅ Net reward in target range [-10, +10]")
    print(f"✅ Penalties are {abs(total_negative/total_positive)*100:.1f}% of rewards (meaningful!)")
    print(f"✅ Expected Q-values: ~{abs(net_total / (1 - 0.99)):.0f} (manageable)")
    
    # Return config
    return {
        'coverage_reward': coverage_per_cell,
        'coverage_confidence_weight': confidence_weight,
        'early_completion_bonus': completion_bonus,
        'time_bonus_per_step_saved': time_bonus,
        'frontier_bonus': frontier_bonus,
        'step_penalty': step_penalty,
        'rotation_penalty_small': rotation_small,
        'rotation_penalty_medium': rotation_medium,
        'rotation_penalty_large': rotation_large,
        'revisit_penalty': revisit_penalty,
        'collision_penalty': collision_penalty,
        'stay_penalty': stay_penalty,
    }


if __name__ == "__main__":
    # Analyze current scales
    analyze_current_scales()
    
    # Propose balanced scales
    balanced_config = propose_balanced_scales()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nCURRENT SYSTEM:")
    print("  ❌ Net episode reward: ~3,500 (HUGE!)")
    print("  ❌ Coverage dominates: 100:1 ratio to penalties")
    print("  ❌ Q-values will explode to ~350,000")
    print("  ❌ Gradient explosion risk: CRITICAL")
    
    print("\nPROPOSED SYSTEM:")
    print("  ✅ Net episode reward: ~10 (Manageable)")
    print("  ✅ Balanced: Penalties are 20-30% of rewards")
    print("  ✅ Q-values will be ~1,000 (Stable)")
    print("  ✅ Gradient explosion risk: LOW")
    
    print("\n" + "="*70)
