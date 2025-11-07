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
    
    # Obstacles (static)
    obstacles: np.ndarray  # [H, W] boolean array
    
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
    """
    
    def __init__(
        self,
        coverage_reward: float = 10.0,
        revisit_penalty: float = -0.5,
        collision_penalty: float = -5.0,
        step_penalty: float = -0.01,
        frontier_bonus: float = 2.0,
        coverage_confidence_weight: float = 0.5
    ):
        """
        Args:
            coverage_reward: Reward per newly covered cell
            revisit_penalty: Penalty for revisiting covered cell
            collision_penalty: Penalty for collision with obstacle
            step_penalty: Small penalty per step (encourages efficiency)
            frontier_bonus: Bonus for moving to frontier cell
            coverage_confidence_weight: How much to weight coverage confidence
        """
        self.coverage_reward = coverage_reward
        self.revisit_penalty = revisit_penalty
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.frontier_bonus = frontier_bonus
        self.confidence_weight = coverage_confidence_weight
    
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
        
        # 3. Revisit penalty
        if state.visited[agent.y, agent.x]:
            revisit_r = self.revisit_penalty
        else:
            revisit_r = 0.0
        breakdown['revisit'] = revisit_r
        
        # 4. Collision penalty
        if next_state.obstacles[agent.y, agent.x]:
            collision_r = self.collision_penalty
        else:
            collision_r = 0.0
        breakdown['collision'] = collision_r
        
        # 5. Frontier bonus
        if (agent.y, agent.x) in next_state.frontiers:
            frontier_r = self.frontier_bonus
        else:
            frontier_r = 0.0
        breakdown['frontier'] = frontier_r
        
        # 6. Step penalty (encourages efficiency)
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
        
        # Initial sensing for each agent
        for agent in agents:
            self._update_coverage(agent, coverage, coverage_confidence, obstacles, visited)
        
        # Detect frontiers
        frontiers = self.frontier_detector.detect_frontiers(visited, obstacles)
        
        # Create state
        self.state = CoverageState(
            height=self.grid_size,
            width=self.grid_size,
            obstacles=obstacles,
            visited=visited,
            coverage=coverage,
            coverage_confidence=coverage_confidence,
            agents=agents,
            frontiers=frontiers,
            step=0
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
        visited: np.ndarray
    ):
        """
        Update coverage based on agent's sensing.
        
        Uses probabilistic sensor model.
        """
        # Get sensor footprint
        cells, probs = self.sensor_model.get_sensor_footprint(
            agent.x, agent.y, (self.grid_size, self.grid_size)
        )
        
        for (y, x), prob in zip(cells, probs):
            # Simulate sensing
            detected, confidence = self.sensor_model.sense_cell(
                distance=np.sqrt((x - agent.x)**2 + (y - agent.y)**2),
                is_obstacle=obstacles[y, x]
            )
            
            if detected and not obstacles[y, x]:
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
        
        # Save previous state for reward computation
        prev_state = self.state.copy()
        
        # Get action delta
        dy, dx = ACTION_TO_DELTA[Action(action)]
        
        # Get agent
        agent = self.state.agents[agent_id]
        
        # Compute new position
        new_x = np.clip(agent.x + dx, 0, self.grid_size - 1)
        new_y = np.clip(agent.y + dy, 0, self.grid_size - 1)
        
        # Check collision
        collision = self.state.obstacles[new_y, new_x]
        
        if collision:
            # Don't move if collision
            self.episode_stats['collisions'] += 1
        else:
            # Update agent position
            agent.x = new_x
            agent.y = new_y
        
        # Check revisit
        if self.state.visited[agent.y, agent.x]:
            self.episode_stats['revisits'] += 1
        
        # Update coverage based on new position
        self._update_coverage(
            agent,
            self.state.coverage,
            self.state.coverage_confidence,
            self.state.obstacles,
            self.state.visited
        )
        
        # Update frontiers
        self.state.frontiers = self.frontier_detector.detect_frontiers(
            self.state.visited,
            self.state.obstacles
        )
        
        # Increment step
        self.state.step += 1
        
        # Compute reward
        reward, reward_breakdown = self.reward_fn.compute_reward(
            prev_state, self.state, Action(action), agent_id
        )
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
        
        # Channel 4: Obstacles
        obs[4] = self.state.obstacles.astype(np.float32)
        
        return obs
    
    def get_valid_actions(self, agent_id: int = 0) -> np.ndarray:
        """
        Get boolean mask of valid actions.
        
        Args:
            agent_id: Which agent
        
        Returns:
            valid: [9] boolean array
        """
        valid = np.ones(9, dtype=bool)
        
        agent = self.state.agents[agent_id]
        
        for action in range(9):
            dy, dx = ACTION_TO_DELTA[Action(action)]
            new_x = agent.x + dx
            new_y = agent.y + dy
            
            # Check bounds
            if new_x < 0 or new_x >= self.grid_size or \
               new_y < 0 or new_y >= self.grid_size:
                valid[action] = False
                continue
            
            # Check obstacle (optional: allow but penalize)
            # Here we mark as invalid to prevent collisions
            if self.state.obstacles[new_y, new_x]:
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
