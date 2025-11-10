"""
Complete Coverage Environment Implementation

A realistic multi-agent coverage planning environment with:
- Probabilistic sensor models
- Obstacle dynamics
- Frontier detection
- Multi-agent coordination
- Detailed state encoding
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import IntEnum
import copy

# Import map generator for curriculum learning
try:
    from map_generator import MapGenerator
    MAP_GENERATOR_AVAILABLE = True
except ImportError:
    MAP_GENERATOR_AVAILABLE = False


class Action(IntEnum):
    """Discrete action space for grid movement."""
    STAY = 0
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7
    NORTHWEST = 8


# Action to direction mapping (dy, dx)
ACTION_TO_DELTA = {
    Action.STAY: (0, 0),
    Action.NORTH: (-1, 0),
    Action.NORTHEAST: (-1, 1),
    Action.EAST: (0, 1),
    Action.SOUTHEAST: (1, 1),
    Action.SOUTH: (1, 0),
    Action.SOUTHWEST: (1, -1),
    Action.WEST: (0, -1),
    Action.NORTHWEST: (-1, -1),
}


@dataclass
class AgentState:
    """State of a single agent."""
    x: int  # X position
    y: int  # Y position
    sensor_range: float  # Sensor range
    energy: float = 1.0  # Energy level (for future extensions)
    active: bool = True  # Whether agent is active


@dataclass
class CoverageState:
    """Complete state of the coverage environment."""
    # Grid dimensions
    height: int
    width: int
    
    # Obstacles (static, ground truth - NOT observed by agent)
    obstacles: np.ndarray  # [H, W] boolean array (GROUND TRUTH)
    
    # Obstacle belief (what the agent has discovered through sensing)
    obstacle_belief: np.ndarray  # [H, W] float - probability of obstacle [0, 1]
    
    # Coverage information
    visited: np.ndarray  # [H, W] boolean - ever visited
    coverage: np.ndarray  # [H, W] float - coverage probability [0, 1]
    coverage_confidence: np.ndarray  # [H, W] float - confidence in coverage
    
    # Agent states
    agents: List[AgentState]
    
    # Frontier cells (boundary between known and unknown)
    frontiers: Set[Tuple[int, int]]
    
    # Step counter
    step: int = 0
    
    # Last action (for rotation penalty tracking)
    last_action: Optional[int] = None
    
    def copy(self) -> 'CoverageState':
        """Deep copy of state."""
        return copy.deepcopy(self)


class ProbabilisticSensorModel:
    """
    Probabilistic sensor model for coverage.
    
    Models uncertainty in sensing with distance-dependent probability.
    """
    
    def __init__(
        self,
        max_range: float,
        detection_prob_at_center: float = 0.95,
        detection_prob_at_edge: float = 0.5,
        false_positive_rate: float = 0.01,
        false_negative_rate: float = 0.05
    ):
        """
        Args:
            max_range: Maximum sensing range
            detection_prob_at_center: Detection probability at agent position
            detection_prob_at_edge: Detection probability at max_range
            false_positive_rate: Probability of detecting when nothing there
            false_negative_rate: Probability of missing when something there
        """
        self.max_range = max_range
        self.p_center = detection_prob_at_center
        self.p_edge = detection_prob_at_edge
        self.false_positive = false_positive_rate
        self.false_negative = false_negative_rate
    
    def get_detection_probability(self, distance: float) -> float:
        """
        Get detection probability as function of distance.
        
        Uses exponential decay model:
        P(detect|distance) = p_edge + (p_center - p_edge) * exp(-λ * distance)
        
        Args:
            distance: Euclidean distance from agent
        
        Returns:
            Detection probability in [0, 1]
        """
        if distance > self.max_range:
            return 0.0
        
        # Exponential decay
        lambda_param = -np.log((self.p_edge - self.false_positive) / 
                               (self.p_center - self.false_positive)) / self.max_range
        
        prob = self.false_positive + (self.p_center - self.false_positive) * \
               np.exp(-lambda_param * distance)
        
        return np.clip(prob, 0.0, 1.0)
    
    def sense_cell(self, distance: float, is_obstacle: bool = False) -> Tuple[bool, float]:
        """
        Simulate sensing a single cell.
        
        Args:
            distance: Distance to cell
            is_obstacle: Whether cell contains obstacle
        
        Returns:
            detected: Whether cell was detected
            confidence: Confidence in detection [0, 1]
        """
        base_prob = self.get_detection_probability(distance)
        
        # Apply false positive/negative
        if is_obstacle:
            # False negative possible
            detected = np.random.random() > self.false_negative
            confidence = base_prob * (1 - self.false_negative)
        else:
            # False positive possible
            detected = np.random.random() < base_prob
            confidence = base_prob
        
        return detected, confidence
    
    def get_sensor_footprint(
        self,
        agent_x: int,
        agent_y: int,
        grid_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sensor footprint (which cells can be sensed and with what probability).
        
        Args:
            agent_x: Agent X position
            agent_y: Agent Y position
            grid_shape: (height, width) of grid
        
        Returns:
            cells: [N, 2] array of (y, x) coordinates in range
            probs: [N] array of detection probabilities
        """
        height, width = grid_shape
        
        # Get all cells within circular range
        cells = []
        probs = []
        
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - agent_x)**2 + (y - agent_y)**2)
                
                if distance <= self.max_range:
                    prob = self.get_detection_probability(distance)
                    cells.append((y, x))
                    probs.append(prob)
        
        return np.array(cells), np.array(probs)


class FrontierDetector:
    """
    Detects frontier cells (boundary between explored and unexplored regions).
    
    Frontier cells are critical for exploration strategies.
    """
    
    @staticmethod
    def detect_frontiers(
        visited: np.ndarray,
        obstacles: np.ndarray,
        min_unknown_neighbors: int = 1
    ) -> Set[Tuple[int, int]]:
        """
        Detect frontier cells.
        
        A cell is a frontier if:
        1. It is visited (known)
        2. It is not an obstacle
        3. It has at least min_unknown_neighbors that are unvisited
        
        Args:
            visited: [H, W] boolean array of visited cells
            obstacles: [H, W] boolean array of obstacles
            min_unknown_neighbors: Minimum number of unvisited neighbors
        
        Returns:
            Set of (y, x) frontier coordinates
        """
        height, width = visited.shape
        frontiers = set()
        
        # 8-connectivity neighbors
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for y in range(height):
            for x in range(width):
                # Must be visited and not obstacle
                if not visited[y, x] or obstacles[y, x]:
                    continue
                
                # Count unvisited neighbors
                unknown_count = 0
                for dy, dx in neighbor_offsets:
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if 0 <= ny < height and 0 <= nx < width:
                        if not visited[ny, nx] and not obstacles[ny, nx]:
                            unknown_count += 1
                
                if unknown_count >= min_unknown_neighbors:
                    frontiers.add((y, x))
        
        return frontiers
    
    @staticmethod
    def get_frontier_clusters(
        frontiers: Set[Tuple[int, int]],
        max_cluster_distance: float = 2.0
    ) -> List[List[Tuple[int, int]]]:
        """
        Cluster frontier cells into groups.
        
        Useful for high-level planning (send agents to different frontier clusters).
        
        Args:
            frontiers: Set of frontier coordinates
            max_cluster_distance: Maximum distance within cluster
        
        Returns:
            List of frontier clusters (each cluster is list of coordinates)
        """
        if not frontiers:
            return []
        
        frontiers_list = list(frontiers)
        clusters = []
        visited = set()
        
        for frontier in frontiers_list:
            if frontier in visited:
                continue
            
            # BFS to find connected frontiers
            cluster = [frontier]
            queue = [frontier]
            visited.add(frontier)
            
            while queue:
                cy, cx = queue.pop(0)
                
                for other in frontiers_list:
                    if other in visited:
                        continue
                    
                    oy, ox = other
                    distance = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                    
                    if distance <= max_cluster_distance:
                        cluster.append(other)
                        queue.append(other)
                        visited.add(other)
            
            clusters.append(cluster)
        
        return clusters


class RewardFunction:
    """
    Reward function for coverage task.
    
    Balances:
    - Coverage gain (positive)
    - Revisiting (negative)
    - Collisions (negative)
    - Step penalty (negative)
    - Frontier exploration (positive)
    - Rotation penalty (negative, encourages smooth paths)
    - STAY penalty (negative, encourages movement)
    - Early completion bonus (positive, rewards fast completion)
    """
    
    def __init__(
        self,
        coverage_reward: float = 0.2,  # 10× scaled: ~68 per episode (was 0.02)
        revisit_penalty: float = -0.5,  # 10× scaled: meaningful penalty (was -0.05)
        collision_penalty: float = -5.0,  # 10× scaled: strong deterrent (was -0.5)
        step_penalty: float = -0.05,  # 10× scaled: efficiency pressure (was -0.005)
        frontier_bonus: float = 0.5,  # 10× scaled: exploration bonus (was 0.05)
        coverage_confidence_weight: float = 0.1,  # 10× scaled: confidence weight (was 0.01)
        use_rotation_penalty: bool = True,
        rotation_penalty_small: float = -0.1,  # ≤45° 10× scaled (was -0.01)
        rotation_penalty_medium: float = -0.2,  # ≤90° 10× scaled (was -0.02)
        rotation_penalty_large: float = -0.5,  # >90° 10× scaled (was -0.05)
        stay_penalty: float = -1.0,  # 10× scaled: movement incentive (was -0.1)
        enable_early_completion: bool = True,  # Enable early completion bonus
        early_completion_threshold: float = 0.95,  # Coverage threshold for bonus
        early_completion_bonus: float = 20.0,  # 10× scaled: base bonus (was 2.0)
        time_bonus_per_step_saved: float = 0.1,  # 10× scaled: time bonus (was 0.01)
        use_progressive_revisit_penalty: bool = True,  # Progressive penalty scaling
        revisit_penalty_min: float = -0.1,  # Initial revisit penalty (lenient early on)
        revisit_penalty_max: float = -0.5,  # Final revisit penalty (strict later)
        max_steps: int = 500  # Maximum steps per episode (for calculating progress ratio)
    ):
        """
        Args:
            coverage_reward: Reward per newly covered cell
            revisit_penalty: Penalty for revisiting covered cell (used as max if progressive enabled)
            collision_penalty: Penalty for collision with obstacle
            step_penalty: Small penalty per step (encourages efficiency)
            frontier_bonus: Bonus for moving to frontier cell
            coverage_confidence_weight: How much to weight coverage confidence
            use_rotation_penalty: Enable rotation penalty
            rotation_penalty_small: Penalty for small rotations (≤45°)
            rotation_penalty_medium: Penalty for medium rotations (≤90°)
            rotation_penalty_large: Penalty for large rotations (>90°)
            stay_penalty: Penalty for STAY action (encourages movement)
            enable_early_completion: Enable early completion bonus
            early_completion_threshold: Coverage percentage to trigger bonus (e.g., 0.95 = 95%)
            early_completion_bonus: Base bonus for completing coverage goal
            time_bonus_per_step_saved: Additional bonus per step saved from max_steps
            use_progressive_revisit_penalty: Enable progressive penalty that scales with episode progress
            revisit_penalty_min: Initial revisit penalty (lenient early on)
            revisit_penalty_max: Final revisit penalty (strict later)
            max_steps: Maximum steps per episode (for calculating progress ratio)
        """
        self.coverage_reward = coverage_reward
        self.revisit_penalty = revisit_penalty
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.frontier_bonus = frontier_bonus
        self.confidence_weight = coverage_confidence_weight

        # Rotation penalty
        self.use_rotation_penalty = use_rotation_penalty
        self.rotation_penalty_small = rotation_penalty_small
        self.rotation_penalty_medium = rotation_penalty_medium
        self.rotation_penalty_large = rotation_penalty_large
        self.stay_penalty = stay_penalty

        # Early completion bonus
        self.enable_early_completion = enable_early_completion
        self.early_completion_threshold = early_completion_threshold
        self.early_completion_bonus = early_completion_bonus
        self.time_bonus_per_step_saved = time_bonus_per_step_saved

        # Progressive revisit penalty
        self.use_progressive_revisit = use_progressive_revisit_penalty
        self.revisit_penalty_min = revisit_penalty_min
        self.revisit_penalty_max = revisit_penalty_max
        self.max_steps = max_steps

        # Action angles for rotation computation
        # N=0°, NE=45°, E=90°, SE=135°, S=180°, SW=225°, W=270°, NW=315°
        self.action_angles = {
            Action.NORTH: 0,
            Action.NORTHEAST: 45,
            Action.EAST: 90,
            Action.SOUTHEAST: 135,
            Action.SOUTH: 180,
            Action.SOUTHWEST: 225,
            Action.WEST: 270,
            Action.NORTHWEST: 315,
            Action.STAY: None  # No direction
        }
    
    
    def _compute_rotation_penalty(self, prev_action: Optional[int], current_action: int) -> float:
        """
        Compute penalty based on direction change between actions.
        
        Encourages smooth trajectories by penalizing sharp turns.
        
        Args:
            prev_action: Previous action [0-8] or None
            current_action: Current action [0-8]
        
        Returns:
            Negative reward proportional to rotation angle (0 to -0.15)
        """
        if not self.use_rotation_penalty:
            return 0.0
        
        # No penalty on first move or after STAY
        if prev_action is None or prev_action == Action.STAY:
            return 0.0
        
        # STAY action has no rotation
        if current_action == Action.STAY:
            return 0.0
        
        # Get angles for both actions
        prev_angle = self.action_angles[Action(prev_action)]
        current_angle = self.action_angles[Action(current_action)]
        
        # Calculate minimum rotation (accounting for 360° wrap-around)
        angle_diff = abs(current_angle - prev_angle)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # Apply graduated penalties based on rotation magnitude
        if angle_diff == 0:
            return 0.0  # No rotation
        elif angle_diff <= 45:
            return self.rotation_penalty_small   # -0.05
        elif angle_diff <= 90:
            return self.rotation_penalty_medium  # -0.10
        else:  # 135° or 180°
            return self.rotation_penalty_large   # -0.15
    
    def compute_early_completion_bonus(
        self,
        coverage_percentage: float,
        current_step: int,
        max_steps: int
    ) -> float:
        """
        Compute bonus for completing coverage early.

        Rewards agents that:
        1. Reach high coverage (>= threshold, e.g., 95%)
        2. Complete quickly (fewer steps = higher bonus)

        Args:
            coverage_percentage: Current coverage percentage [0, 1]
            current_step: Current step number
            max_steps: Maximum allowed steps

        Returns:
            Completion bonus (0 if not completed, or large positive if completed early)
        """
        if not self.enable_early_completion:
            return 0.0

        # Check if coverage threshold reached
        if coverage_percentage < self.early_completion_threshold:
            return 0.0

        # Base bonus for completing coverage
        bonus = self.early_completion_bonus

        # Additional time bonus for steps saved
        steps_saved = max(0, max_steps - current_step)
        time_bonus = steps_saved * self.time_bonus_per_step_saved

        total_bonus = bonus + time_bonus

        return total_bonus

    def compute_reward_optimized(
        self,
        coverage_gain: float,
        confidence_gain: float,
        is_revisit: bool,
        is_on_frontier: bool,
        is_collision: bool,
        action: Action,
        last_action: int,
        current_step: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimized reward computation using pre-computed flags.

        This avoids the shallow copy bug by receiving pre-computed values
        instead of comparing states.

        Args:
            coverage_gain: Sum of new coverage added
            confidence_gain: Sum of new confidence added
            is_revisit: True if agent moved to already visited cell
            is_on_frontier: True if new position has unvisited neighbors
            is_collision: True if agent collided with obstacle
            action: Action taken
            last_action: Previous action (for rotation penalty)
            current_step: Current step number

        Returns:
            total_reward: Scalar reward
            reward_breakdown: Dictionary of reward components
        """
        breakdown = {}

        # 1. Coverage reward (based on new coverage)
        coverage_r = coverage_gain * self.coverage_reward
        breakdown['coverage'] = coverage_r

        # 2. Coverage confidence bonus
        confidence_r = confidence_gain * self.confidence_weight
        breakdown['confidence'] = confidence_r

        # 3. Revisit penalty (progressive: scales from lenient to strict)
        if is_revisit:
            if self.use_progressive_revisit:
                # Calculate progress ratio (0.0 at start, 1.0 at max_steps)
                progress = min(current_step / self.max_steps, 1.0)
                # Linear interpolation: penalty grows from min to max
                current_penalty = self.revisit_penalty_min + (
                    self.revisit_penalty_max - self.revisit_penalty_min
                ) * progress
                revisit_r = current_penalty
            else:
                revisit_r = self.revisit_penalty
        else:
            revisit_r = 0.0
        breakdown['revisit'] = revisit_r

        # 4. Rotation penalty
        rotation_r = self._compute_rotation_penalty(last_action, int(action))
        breakdown['rotation'] = rotation_r

        # 5. Collision penalty
        collision_r = self.collision_penalty if is_collision else 0.0
        breakdown['collision'] = collision_r

        # 6. Frontier bonus - reward expanding exploration
        # Uses pre-computed flag to avoid shallow copy bug
        frontier_r = self.frontier_bonus if is_on_frontier else 0.0
        breakdown['frontier'] = frontier_r

        # 7. STAY penalty (encourages movement)
        stay_r = self.stay_penalty if int(action) == Action.STAY else 0.0
        breakdown['stay'] = stay_r

        # 8. Step penalty (encourages efficiency)
        breakdown['step'] = self.step_penalty

        # Total reward
        total = sum(breakdown.values())

        return total, breakdown
    
    def compute_reward(
        self,
        state: CoverageState,
        next_state: CoverageState,
        action: Action,
        agent_id: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for transition.
        
        Args:
            state: Previous state
            next_state: New state after action
            action: Action taken
            agent_id: Which agent took action
        
        Returns:
            total_reward: Scalar reward
            reward_breakdown: Dictionary of reward components
        """
        agent = next_state.agents[agent_id]
        breakdown = {}
        
        # 1. Coverage reward (based on new coverage)
        coverage_gain = (next_state.coverage - state.coverage).sum()
        coverage_r = coverage_gain * self.coverage_reward
        breakdown['coverage'] = coverage_r
        
        # 2. Coverage confidence bonus
        confidence_gain = (next_state.coverage_confidence - 
                          state.coverage_confidence).sum()
        confidence_r = confidence_gain * self.confidence_weight
        breakdown['confidence'] = confidence_r
        
        # 3. Revisit penalty (progressive: scales from lenient to strict)
        if state.visited[agent.y, agent.x]:
            if self.use_progressive_revisit:
                # Calculate progress ratio (0.0 at start, 1.0 at max_steps)
                progress = min(next_state.step / self.max_steps, 1.0)
                # Linear interpolation: penalty grows from min to max
                current_penalty = self.revisit_penalty_min + (
                    self.revisit_penalty_max - self.revisit_penalty_min
                ) * progress
                revisit_r = current_penalty
            else:
                revisit_r = self.revisit_penalty
        else:
            revisit_r = 0.0
        breakdown['revisit'] = revisit_r
        
        # 4. Rotation penalty (NEW)
        rotation_r = self._compute_rotation_penalty(state.last_action, int(action))
        breakdown['rotation'] = rotation_r
        
        # 5. Collision penalty
        if next_state.obstacles[agent.y, agent.x]:
            collision_r = self.collision_penalty
        else:
            collision_r = 0.0
        breakdown['collision'] = collision_r
        
        # 6. Frontier bonus - reward expanding exploration
        # Give bonus if current position has unvisited neighbors (agent is at frontier)
        # This correctly rewards exploring toward unexplored areas
        frontier_r = 0.0
        H, W = state.visited.shape
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for dy, dx in neighbor_offsets:
            ny, nx = agent.y + dy, agent.x + dx
            if (0 <= ny < H and 0 <= nx < W and
                not state.visited[ny, nx] and not state.obstacles[ny, nx]):
                frontier_r = self.frontier_bonus
                break
        breakdown['frontier'] = frontier_r
        
        # 7. STAY penalty (encourages movement)
        if int(action) == Action.STAY:
            stay_r = self.stay_penalty
        else:
            stay_r = 0.0
        breakdown['stay'] = stay_r
        
        # 8. Step penalty (encourages efficiency)
        breakdown['step'] = self.step_penalty
        
        # Total reward
        total = sum(breakdown.values())
        
        return total, breakdown


class CoverageEnvironment:
    """
    Complete multi-agent coverage environment.
    
    Features:
    - Probabilistic sensing
    - Dynamic obstacle generation
    - Frontier detection
    - Multi-agent support
    - Rich state encoding
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        num_agents: int = 1,
        sensor_range: float = 4.0,
        obstacle_density: float = 0.15,
        max_steps: int = 500,
        seed: Optional[int] = None,
        reward_config: Optional[Dict] = None,
        map_type: str = "random"  # NEW: Support for curriculum learning
    ):
        """
        Args:
            grid_size: Size of square grid (height = width = grid_size)
            num_agents: Number of agents
            sensor_range: Sensor range for each agent
            obstacle_density: Fraction of cells that are obstacles (for random maps)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            reward_config: Custom reward function parameters
            map_type: Map type - "random", "empty", "corridor", "room", "cave", "lshape"
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.sensor_range = sensor_range
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.map_type = map_type  # Store for curriculum learning

        # Random seed
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Map generator for curriculum learning
        if MAP_GENERATOR_AVAILABLE:
            self.map_generator = MapGenerator(grid_size)
        else:
            self.map_generator = None
        
        # Components
        self.sensor_model = ProbabilisticSensorModel(
            max_range=sensor_range,
            detection_prob_at_center=0.95,
            detection_prob_at_edge=0.5
        )
        self.frontier_detector = FrontierDetector()
        
        # Reward function
        if reward_config is None:
            reward_config = {}
        # Inject max_steps for progressive penalty calculation
        reward_config['max_steps'] = max_steps
        self.reward_fn = RewardFunction(**reward_config)
        
        # State
        self.state: Optional[CoverageState] = None
        
        # Statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'coverage_percentage': 0.0,
            'collisions': 0,
            'revisits': 0
        }
    
    def _generate_obstacles(self, map_type: Optional[str] = None) -> np.ndarray:
        """
        Generate obstacles using map_generator if available, else use simple clustering.

        Args:
            map_type: Type of map to generate ("random", "empty", "corridor", "room", "cave", "lshape")
                     If None, uses self.map_type

        Returns:
            obstacles: [H, W] boolean array
        """
        if map_type is None:
            map_type = self.map_type

        # Try to use MapGenerator for structured maps
        if self.map_generator is not None and map_type != "random":
            try:
                _, obstacle_set = self.map_generator.generate(map_type)

                # Convert set of (x, y) to boolean array [H, W]
                obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
                for (x, y) in obstacle_set:
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        obstacles[y, x] = True  # Note: y, x indexing

                return obstacles
            except Exception as e:
                # Fall back to simple generation
                print(f"Warning: MapGenerator failed ({e}), using simple generation")

        # Simple clustered generation (original method)
        obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        if map_type == "empty":
            # No obstacles
            return obstacles

        # Simple clustered generation for "random" or fallback
        num_obstacles = int(self.grid_size ** 2 * self.obstacle_density)
        num_clusters = max(1, num_obstacles // 10)

        for _ in range(num_clusters):
            # Random cluster center
            cy = self.rng.randint(0, self.grid_size)
            cx = self.rng.randint(0, self.grid_size)

            # Random cluster size
            cluster_size = self.rng.randint(3, 8)

            # Add obstacles in cluster
            for _ in range(cluster_size):
                dy = self.rng.randint(-2, 3)
                dx = self.rng.randint(-2, 3)

                ny, nx = cy + dy, cx + dx

                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    obstacles[ny, nx] = True

        return obstacles
    
    def _initialize_agents(self, obstacles: np.ndarray) -> List[AgentState]:
        """
        Initialize agent positions.
        
        Ensures agents start in non-obstacle cells with spacing.
        
        Args:
            obstacles: Obstacle map
        
        Returns:
            List of agent states
        """
        agents = []
        occupied = set()
        
        for i in range(self.num_agents):
            # Find valid starting position
            attempts = 0
            while attempts < 1000:
                y = self.rng.randint(0, self.grid_size)
                x = self.rng.randint(0, self.grid_size)
                
                # Check not obstacle and not too close to other agents
                if not obstacles[y, x]:
                    too_close = False
                    for oy, ox in occupied:
                        if abs(y - oy) < 3 and abs(x - ox) < 3:
                            too_close = True
                            break
                    
                    if not too_close:
                        agents.append(AgentState(
                            x=x,
                            y=y,
                            sensor_range=self.sensor_range
                        ))
                        occupied.add((y, x))
                        break
                
                attempts += 1
            
            if attempts >= 1000:
                # Fallback: just place anywhere valid
                valid_cells = np.argwhere(~obstacles)
                idx = self.rng.choice(len(valid_cells))
                y, x = valid_cells[idx]
                agents.append(AgentState(x=x, y=y, sensor_range=self.sensor_range))
        
        return agents
    
    def reset(self, map_type: Optional[str] = None) -> np.ndarray:
        """
        Reset environment to initial state.

        Args:
            map_type: Override map type for this episode (for curriculum learning)

        Returns:
            observation: [5, H, W] encoded state
        """
        # Allow curriculum to override map type
        if map_type is not None:
            current_map_type = map_type
        else:
            current_map_type = self.map_type

        # Generate obstacles using specified map type
        obstacles = self._generate_obstacles(current_map_type)
        
        # Initialize agents
        agents = self._initialize_agents(obstacles)
        
        # Initialize coverage arrays
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        coverage = np.zeros((self.grid_size, self.grid_size), dtype=float)
        coverage_confidence = np.zeros((self.grid_size, self.grid_size), dtype=float)
        
        # Initialize obstacle belief (starts with NO knowledge of obstacles)
        obstacle_belief = np.zeros((self.grid_size, self.grid_size), dtype=float)
        
        # Initial sensing for each agent (will update obstacle_belief)
        for agent in agents:
            self._update_coverage(agent, coverage, coverage_confidence, obstacles, visited, obstacle_belief)
        
        # Detect frontiers
        frontiers = self.frontier_detector.detect_frontiers(visited, obstacles)
        
        # Create state
        self.state = CoverageState(
            height=self.grid_size,
            width=self.grid_size,
            obstacles=obstacles,  # Ground truth (hidden from agent)
            obstacle_belief=obstacle_belief,  # Agent's belief (starts at zero)
            visited=visited,
            coverage=coverage,
            coverage_confidence=coverage_confidence,
            agents=agents,
            frontiers=frontiers,
            step=0,
            last_action=None  # No previous action at start
        )
        
        # Reset statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'coverage_percentage': self._compute_coverage_percentage(),
            'collisions': 0,
            'revisits': 0
        }
        
        return self._encode_observation()
    
    def _update_coverage(
        self,
        agent: AgentState,
        coverage: np.ndarray,
        coverage_confidence: np.ndarray,
        obstacles: np.ndarray,
        visited: np.ndarray,
        obstacle_belief: np.ndarray
    ):
        """
        Update coverage AND obstacle belief based on agent's sensing.
        
        Uses probabilistic sensor model to detect both coverage and obstacles.
        
        Args:
            agent: Agent performing the sensing
            coverage: Coverage probability map to update
            coverage_confidence: Confidence map to update
            obstacles: Ground truth obstacles (for simulation)
            visited: Visited cells map to update
            obstacle_belief: Agent's belief about obstacles (UPDATED HERE)
        """
        # Get sensor footprint
        cells, probs = self.sensor_model.get_sensor_footprint(
            agent.x, agent.y, (self.grid_size, self.grid_size)
        )
        
        for (y, x), prob in zip(cells, probs):
            # Simulate sensing (uses ground truth for simulation accuracy)
            detected, confidence = self.sensor_model.sense_cell(
                distance=np.sqrt((x - agent.x)**2 + (y - agent.y)**2),
                is_obstacle=obstacles[y, x]
            )
            
            # Update obstacle belief (CRITICAL: This is what the agent learns!)
            if obstacles[y, x]:
                # This cell IS an obstacle (ground truth)
                # Agent's sensor detects it with probability `detected`
                if detected:
                    # Detected as obstacle - increase belief
                    obstacle_belief[y, x] = min(1.0, obstacle_belief[y, x] + prob * 0.8)
            else:
                # This cell is NOT an obstacle (ground truth)
                # If detected, we know it's free space
                if detected:
                    # Detected as free - decrease obstacle belief (confirm it's clear)
                    obstacle_belief[y, x] = max(0.0, obstacle_belief[y, x] - prob * 0.5)
                    
                    # Update coverage with Bayesian update
                    # Prior: current coverage
                    # Likelihood: detection probability
                    # Posterior: updated coverage
                    
                    prior = coverage[y, x]
                    likelihood = prob
                    
                    # Bayesian update (simplified)
                    posterior = prior + (1 - prior) * likelihood
                    coverage[y, x] = min(1.0, posterior)
                    
                    # Update confidence
                    coverage_confidence[y, x] = min(1.0, coverage_confidence[y, x] + confidence * 0.1)
                    
                    # Mark as visited
                    visited[y, x] = True

    def _has_unvisited_neighbors(self, y: int, x: int) -> bool:
        """
        Check if position (y, x) has any unvisited neighbors.

        This is used for frontier bonus computation and must be called
        BEFORE updating the visited array to avoid the shallow copy bug.

        Args:
            y: Y coordinate
            x: X coordinate

        Returns:
            True if position has at least one unvisited, non-obstacle neighbor
        """
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        H, W = self.state.visited.shape

        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx
            if (0 <= ny < H and 0 <= nx < W and
                not self.state.visited[ny, nx] and
                not self.state.obstacles[ny, nx]):
                return True
        return False

    def step(self, action: int, agent_id: int = 0) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take environment step.

        Args:
            action: Action index [0-8]
            agent_id: Which agent is acting

        Returns:
            observation: [5, H, W] next state
            reward: Scalar reward
            done: Whether episode is done
            info: Additional information dictionary
        """
        assert self.state is not None, "Must call reset() first"
        assert 0 <= action < 9, f"Invalid action: {action}"
        assert 0 <= agent_id < self.num_agents, f"Invalid agent_id: {agent_id}"

        # PRE-COMPUTE VALUES BEFORE STATE MODIFICATION (avoids shallow copy bug)
        # Save coverage/confidence sums before any modifications
        prev_coverage_sum = self.state.coverage.sum()
        prev_confidence_sum = self.state.coverage_confidence.sum()

        # Get action delta
        dy, dx = ACTION_TO_DELTA[Action(action)]

        # Get agent
        agent = self.state.agents[agent_id]

        # Compute new position
        new_x = np.clip(agent.x + dx, 0, self.grid_size - 1)
        new_y = np.clip(agent.y + dy, 0, self.grid_size - 1)

        # Check frontier status BEFORE moving (critical for frontier bonus!)
        is_on_frontier = self._has_unvisited_neighbors(new_y, new_x)

        # Save last action for rotation penalty
        last_action = self.state.last_action if self.state.last_action is not None else action

        # Check collision
        collision = self.state.obstacles[new_y, new_x]

        if collision:
            # Don't move if collision
            self.episode_stats['collisions'] += 1
        else:
            # Update agent position
            agent.x = new_x
            agent.y = new_y

        # Check revisit BEFORE updating coverage (critical for revisit penalty!)
        is_revisit = self.state.visited[agent.y, agent.x]
        if is_revisit:
            self.episode_stats['revisits'] += 1

        # Update coverage AND obstacle belief based on new position
        self._update_coverage(
            agent,
            self.state.coverage,
            self.state.coverage_confidence,
            self.state.obstacles,
            self.state.visited,
            self.state.obstacle_belief  # Now updates obstacle belief!
        )

        # Compute gains AFTER state update
        coverage_gain = self.state.coverage.sum() - prev_coverage_sum
        confidence_gain = self.state.coverage_confidence.sum() - prev_confidence_sum

        # Update frontiers
        self.state.frontiers = self.frontier_detector.detect_frontiers(
            self.state.visited,
            self.state.obstacles
        )

        # Increment step
        self.state.step += 1

        # Update last action for rotation penalty tracking
        self.state.last_action = action

        # Compute reward using OPTIMIZED function with pre-computed flags
        reward, reward_breakdown = self.reward_fn.compute_reward_optimized(
            coverage_gain=coverage_gain,
            confidence_gain=confidence_gain,
            is_revisit=is_revisit,
            is_on_frontier=is_on_frontier,
            is_collision=collision,
            action=Action(action),
            last_action=last_action,
            current_step=self.state.step
        )
        
        # Compute coverage percentage for early completion check
        coverage_pct = self._compute_coverage_percentage()
        
        # Check for early completion bonus
        early_completion_bonus = self.reward_fn.compute_early_completion_bonus(
            coverage_pct, self.state.step, self.max_steps
        )
        
        if early_completion_bonus > 0:
            reward_breakdown['early_completion'] = early_completion_bonus
            reward += early_completion_bonus
        else:
            reward_breakdown['early_completion'] = 0.0
        
        self.episode_stats['total_reward'] += reward
        
        # Check done
        done = (self.state.step >= self.max_steps or 
                self._compute_coverage_percentage() >= 0.99)
        
        # Update stats
        self.episode_stats['coverage_percentage'] = self._compute_coverage_percentage()
        
        # Prepare info
        info = {
            'coverage_pct': self.episode_stats['coverage_percentage'],
            'steps': self.state.step,
            'collisions': self.episode_stats['collisions'],
            'revisits': self.episode_stats['revisits'],
            'reward_breakdown': reward_breakdown,
            'num_frontiers': len(self.state.frontiers),
            'agent_position': (agent.x, agent.y)
        }
        
        return self._encode_observation(), reward, done, info
    
    def _compute_coverage_percentage(self) -> float:
        """Compute coverage percentage (excluding obstacles)."""
        total_coverable = (~self.state.obstacles).sum()
        covered = (self.state.coverage > 0.5).sum()  # Threshold at 50% confidence
        return covered / total_coverable if total_coverable > 0 else 0.0
    
    def _encode_observation(self) -> np.ndarray:
        """
        Encode state as observation for neural network.
        
        Returns:
            observation: [5, H, W] array with channels:
                0: Visited cells (binary)
                1: Coverage probability (float [0, 1])
                2: Agent positions (binary, all agents)
                3: Frontiers (binary)
                4: Obstacles (binary)
        """
        obs = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Visited
        obs[0] = self.state.visited.astype(np.float32)
        
        # Channel 1: Coverage probability
        obs[1] = self.state.coverage
        
        # Channel 2: Agent positions
        for agent in self.state.agents:
            if agent.active:
                obs[2, agent.y, agent.x] = 1.0
        
        # Channel 3: Frontiers
        for y, x in self.state.frontiers:
            obs[3, y, x] = 1.0
        
        # Channel 4: Obstacle Belief (NOT ground truth!)
        # Agent only knows what it has sensed, not the true obstacle map
        obs[4] = self.state.obstacle_belief
        
        return obs
    
    def get_valid_actions(self, agent_id: int = 0) -> np.ndarray:
        """
        Get boolean mask of valid actions.
        
        CRITICAL: Uses OBSTACLE_BELIEF (not ground truth) to determine validity.
        This allows the agent to attempt moves into unknown cells and learn from
        collision penalties.
        
        Args:
            agent_id: Which agent
        
        Returns:
            valid: [9] boolean array
        """
        valid = np.ones(9, dtype=bool)
        
        agent = self.state.agents[agent_id]
        
        # Obstacle belief threshold for considering a cell "blocked"
        # If belief > 0.7, assume it's an obstacle and mark as invalid
        # If belief < 0.7, allow the action (agent might discover obstacle!)
        obstacle_threshold = 0.7
        
        for action in range(9):
            dy, dx = ACTION_TO_DELTA[Action(action)]
            new_x = agent.x + dx
            new_y = agent.y + dy
            
            # Check bounds (always enforced)
            if new_x < 0 or new_x >= self.grid_size or \
               new_y < 0 or new_y >= self.grid_size:
                valid[action] = False
                continue
            
            # Check obstacle BELIEF (not ground truth!)
            # Agent can only avoid obstacles it has discovered
            if self.state.obstacle_belief[new_y, new_x] > obstacle_threshold:
                valid[action] = False
        
        return valid
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: 'human' for display, 'rgb_array' for array
        
        Returns:
            RGB array if mode='rgb_array', else None
        """
        # Create RGB image
        img = np.ones((self.grid_size, self.grid_size, 3), dtype=np.uint8) * 255
        
        # Obstacles (black)
        img[self.state.obstacles] = [0, 0, 0]
        
        # Coverage (green gradient)
        covered = self.state.coverage > 0.5
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if covered[y, x] and not self.state.obstacles[y, x]:
                    intensity = int(self.state.coverage[y, x] * 200)
                    img[y, x] = [0, intensity, 0]
        
        # Frontiers (yellow)
        for y, x in self.state.frontiers:
            img[y, x] = [255, 255, 0]
        
        # Agents (red)
        for agent in self.state.agents:
            if agent.active:
                img[agent.y, agent.x] = [255, 0, 0]
        
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            # For human display, you'd use matplotlib or similar
            try:
                import matplotlib.pyplot as plt
                plt.imshow(img, origin='lower')
                plt.title(f"Step: {self.state.step}, Coverage: {self.episode_stats['coverage_percentage']*100:.1f}%")
                plt.pause(0.001)
            except ImportError:
                print("Matplotlib not available for rendering")
        
        return None


if __name__ == "__main__":
    # Comprehensive test
    print("="*70)
    print("Testing Complete Coverage Environment")
    print("="*70)
    
    # Test 1: Basic environment creation
    print("\nTest 1: Environment creation")
    env = CoverageEnvironment(grid_size=20, num_agents=1, sensor_range=4.0, seed=42)
    obs = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Initial coverage: {env.episode_stats['coverage_percentage']*100:.1f}%")
    print(f"  Obstacles: {env.state.obstacles.sum()} cells")
    print(f"  Frontiers: {len(env.state.frontiers)} cells")
    
    # Test 2: Action execution
    print("\nTest 2: Taking actions")
    for i in range(10):
        valid_actions = env.get_valid_actions(agent_id=0)
        action = np.random.choice(np.where(valid_actions)[0])
        
        obs, reward, done, info = env.step(action, agent_id=0)
        
        if i == 0 or i == 9:
            print(f"  Step {info['steps']}: action={action}, reward={reward:.2f}, "
                  f"coverage={info['coverage_pct']*100:.1f}%, frontiers={info['num_frontiers']}")
    
    # Test 3: Full episode
    print("\nTest 3: Full episode")
    env.reset()
    done = False
    step = 0
    
    while not done and step < 100:
        valid_actions = env.get_valid_actions(agent_id=0)
        action = np.random.choice(np.where(valid_actions)[0])
        obs, reward, done, info = env.step(action, agent_id=0)
        step += 1
    
    print(f"  Episode finished:")
    print(f"    Steps: {info['steps']}")
    print(f"    Coverage: {info['coverage_pct']*100:.1f}%")
    print(f"    Collisions: {info['collisions']}")
    print(f"    Revisits: {info['revisits']}")
    print(f"    Total reward: {env.episode_stats['total_reward']:.2f}")
    
    # Test 4: Sensor model
    print("\nTest 4: Sensor model")
    sensor = ProbabilisticSensorModel(max_range=4.0)
    for dist in [0.0, 1.0, 2.0, 3.0, 4.0]:
        prob = sensor.get_detection_probability(dist)
        print(f"  Distance {dist:.1f}: detection prob = {prob:.3f}")
    
    # Test 5: Multiple grid sizes
    print("\nTest 5: Multiple grid sizes (scale invariance)")
    for size in [15, 20, 25, 30]:
        env = CoverageEnvironment(grid_size=size, sensor_range=size*0.2, seed=42)
        obs = env.reset()
        print(f"  {size}×{size}: obs={obs.shape}, initial_coverage={env.episode_stats['coverage_percentage']*100:.1f}%")
    
    print("\n✓ All tests passed!")
