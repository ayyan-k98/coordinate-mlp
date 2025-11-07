"""
Curriculum Learning System for Coverage Training

Implements progressive difficulty using map_generator's diverse environments.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from map_generator import MapGenerator


@dataclass
class CurriculumPhase:
    """Single phase in curriculum."""
    name: str
    map_types: List[str]  # Map types to sample from
    num_episodes: int     # Episodes in this phase
    description: str
    epsilon_boost: Optional[float] = None  # Boost epsilon when entering this phase
    epsilon_floor: Optional[float] = None  # Minimum epsilon during this phase


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    enabled: bool = True

    # Curriculum phases (progressive difficulty)
    phases: List[CurriculumPhase] = field(default_factory=lambda: [
        CurriculumPhase(
            name="Phase 1: Basics",
            map_types=["empty"],
            num_episodes=200,
            description="Learn basic coverage on empty maps",
            epsilon_floor=0.1  # Low exploration needed for empty maps
        ),
        CurriculumPhase(
            name="Phase 2: Obstacles",
            map_types=["random"],
            num_episodes=300,
            description="Learn obstacle avoidance with random obstacles",
            epsilon_floor=0.15  # Moderate exploration for obstacles
        ),
        CurriculumPhase(
            name="Phase 3: Structures",
            map_types=["corridor", "room"],
            num_episodes=400,
            description="Learn navigation in structured environments",
            epsilon_boost=0.3,  # Boost exploration to find doors!
            epsilon_floor=0.2   # Keep exploring to escape rooms
        ),
        CurriculumPhase(
            name="Phase 4: Complex",
            map_types=["cave", "lshape"],
            num_episodes=400,
            description="Master complex irregular structures",
            epsilon_boost=0.35,  # High exploration for complex mazes
            epsilon_floor=0.25   # Maintain exploration throughout
        ),
        CurriculumPhase(
            name="Phase 5: Mixed",
            map_types=["empty", "random", "corridor", "room", "cave", "lshape"],
            num_episodes=200,
            description="Generalize across all map types",
            epsilon_floor=0.1  # Final polish with lower exploration
        ),
    ])

    # Mixing strategy for multi-scale + curriculum
    mix_grid_sizes_within_phase: bool = True  # Mix grid sizes within each phase


class CurriculumScheduler:
    """
    Manages curriculum progression during training.

    Features:
    - Progressive map difficulty
    - Automatic phase transitions
    - Episode tracking
    - Map type sampling
    """

    def __init__(self, config: CurriculumConfig, grid_sizes: List[int] = None):
        """
        Args:
            config: Curriculum configuration
            grid_sizes: List of grid sizes for multi-scale training
        """
        self.config = config
        self.grid_sizes = grid_sizes or [20]

        self.current_phase_idx = 0
        self.episode_in_phase = 0
        self.total_episodes = 0

        # Precompute total episodes
        self.total_curriculum_episodes = sum(
            phase.num_episodes for phase in config.phases
        )

    def get_current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase."""
        if not self.config.enabled:
            # No curriculum - use all map types
            return CurriculumPhase(
                name="No Curriculum",
                map_types=["empty", "random", "corridor", "room", "cave", "lshape"],
                num_episodes=float('inf'),
                description="All map types"
            )

        if self.current_phase_idx >= len(self.config.phases):
            # Beyond curriculum - stay at final phase
            return self.config.phases[-1]

        return self.config.phases[self.current_phase_idx]

    def sample_map_type(self) -> str:
        """Sample a map type from current phase."""
        import random
        phase = self.get_current_phase()
        return random.choice(phase.map_types)

    def sample_grid_size(self) -> int:
        """Sample a grid size."""
        import random
        if self.config.mix_grid_sizes_within_phase:
            return random.choice(self.grid_sizes)
        else:
            # Curriculum on map types only, fixed grid size
            return self.grid_sizes[0]

    def step(self) -> bool:
        """
        Advance one episode.

        Returns:
            True if phase changed, False otherwise
        """
        self.episode_in_phase += 1
        self.total_episodes += 1

        phase = self.get_current_phase()

        # Check if phase complete
        if self.episode_in_phase >= phase.num_episodes:
            if self.current_phase_idx < len(self.config.phases) - 1:
                # Advance to next phase
                self.current_phase_idx += 1
                self.episode_in_phase = 0
                return True  # Phase changed

        return False  # Same phase
    
    def get_epsilon_adjustment(self, current_epsilon: float) -> float:
        """
        Get epsilon value adjusted for current curriculum phase.
        
        This implements phase-specific exploration:
        - Empty maps: Low epsilon (0.1) - exploitation focused
        - Structured maps: High epsilon (0.2-0.3) - must explore to find doors
        - Complex maps: Very high epsilon (0.25-0.35) - escape local optima
        
        Args:
            current_epsilon: Agent's current epsilon value
        
        Returns:
            Adjusted epsilon for current phase
        """
        phase = self.get_current_phase()
        
        # Apply epsilon floor (minimum exploration for this phase)
        if phase.epsilon_floor is not None:
            current_epsilon = max(current_epsilon, phase.epsilon_floor)
        
        return current_epsilon
    
    def should_boost_epsilon(self) -> Optional[float]:
        """
        Check if epsilon should be boosted at phase transition.
        
        Returns:
            Epsilon boost value if phase just started, None otherwise
        """
        phase = self.get_current_phase()
        
        # Only boost at the very start of a phase
        if self.episode_in_phase == 0 and phase.epsilon_boost is not None:
            return phase.epsilon_boost
        
        return None

    def get_progress(self) -> dict:
        """Get curriculum progress information."""
        phase = self.get_current_phase()

        return {
            'phase_name': phase.name,
            'phase_idx': self.current_phase_idx + 1,
            'total_phases': len(self.config.phases),
            'episode_in_phase': self.episode_in_phase,
            'phase_episodes': phase.num_episodes,
            'phase_progress': self.episode_in_phase / phase.num_episodes,
            'total_episodes': self.total_episodes,
            'curriculum_episodes': self.total_curriculum_episodes,
            'overall_progress': min(1.0, self.total_episodes / self.total_curriculum_episodes),
            'map_types': phase.map_types,
            'description': phase.description
        }

    def print_status(self):
        """Print current curriculum status."""
        progress = self.get_progress()

        print(f"\n{'='*70}")
        print(f"Curriculum Status")
        print(f"{'='*70}")
        print(f"Phase: {progress['phase_name']} ({progress['phase_idx']}/{progress['total_phases']})")
        print(f"Description: {progress['description']}")
        print(f"Map Types: {', '.join(progress['map_types'])}")
        print(f"Progress: {progress['episode_in_phase']}/{progress['phase_episodes']} "
              f"({progress['phase_progress']*100:.1f}%)")
        print(f"Overall: {progress['total_episodes']}/{progress['curriculum_episodes']} "
              f"({progress['overall_progress']*100:.1f}%)")
        print(f"{'='*70}\n")


def create_default_curriculum() -> CurriculumConfig:
    """Create default curriculum configuration."""
    return CurriculumConfig(enabled=True)


def create_fast_curriculum() -> CurriculumConfig:
    """Create faster curriculum for testing (fewer episodes)."""
    return CurriculumConfig(
        enabled=True,
        phases=[
            CurriculumPhase("Phase 1: Basics", ["empty"], 100, "Basic coverage", 
                          epsilon_floor=0.1),
            CurriculumPhase("Phase 2: Obstacles", ["random"], 150, "Obstacle avoidance",
                          epsilon_floor=0.15),
            CurriculumPhase("Phase 3: Structures", ["corridor", "room"], 200, "Structured navigation",
                          epsilon_boost=0.3, epsilon_floor=0.2),
            CurriculumPhase("Phase 4: Complex", ["cave", "lshape"], 200, "Complex structures",
                          epsilon_boost=0.35, epsilon_floor=0.25),
            CurriculumPhase("Phase 5: Mixed", ["empty", "random", "corridor", "room", "cave", "lshape"], 100, "Generalization",
                          epsilon_floor=0.1),
        ]
    )


def create_no_curriculum() -> CurriculumConfig:
    """No curriculum - all map types from start."""
    return CurriculumConfig(
        enabled=False,
        phases=[]
    )


if __name__ == "__main__":
    # Test curriculum scheduler
    print("="*70)
    print("Testing Curriculum Scheduler")
    print("="*70)

    # Create scheduler
    curriculum = create_default_curriculum()
    scheduler = CurriculumScheduler(curriculum, grid_sizes=[15, 20, 25, 30])

    print("\nDefault Curriculum Phases:")
    for i, phase in enumerate(curriculum.phases, 1):
        print(f"\n{i}. {phase.name}")
        print(f"   Map Types: {', '.join(phase.map_types)}")
        print(f"   Episodes: {phase.num_episodes}")
        print(f"   Description: {phase.description}")

    # Simulate training
    print("\n" + "="*70)
    print("Simulating Training")
    print("="*70)

    for episode in range(50):
        # Get current map type and grid size
        map_type = scheduler.sample_map_type()
        grid_size = scheduler.sample_grid_size()

        # Advance
        phase_changed = scheduler.step()

        # Print on phase change
        if phase_changed or episode == 0:
            scheduler.print_status()

        if episode < 10 or phase_changed:
            print(f"Episode {episode}: Map={map_type:10s} Grid={grid_size}×{grid_size}")

    # Test fast curriculum
    print("\n" + "="*70)
    print("Fast Curriculum (for testing)")
    print("="*70)

    fast_curriculum = create_fast_curriculum()
    print(f"\nTotal episodes: {sum(p.num_episodes for p in fast_curriculum.phases)}")
    for i, phase in enumerate(fast_curriculum.phases, 1):
        print(f"{i}. {phase.name}: {phase.num_episodes} episodes")

    print("\n✓ Curriculum system tested successfully!")
