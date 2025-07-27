import pygame
import random
from collections import deque
import logging
import matplotlib.pyplot as plt
import numpy as np
import uuid
import time
import os, json

random.seed(52)

# Constants
GRID_SIZE = 29
CELL_SIZE = 29
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
TOTAL_CELLS = GRID_SIZE * GRID_SIZE
NUM_AGENTS = 15
COST_MOVE = 1
BASE_COST_SAND = 5
BASE_COST_PIT_FILL = 2
MAX_STEPS = 500
TERRAIN_DIFFICULTY = {0: 1, -3: 2}
COLLISION_DELAY = 2
MAX_PATH_STEPS = 1000
OBSTACLE_DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # Range of obstacle densities
# Weights for effectiveness score (adjusted for removed success rate)
W1, W2, W4 = 0.33, 0.33, 0.34  # Adjusted weights to sum to 1

# Colors
COLORS = {
    0: (245, 245, 220),     # Empty Space: Beige
    1: (210, 180, 140),     # Sandbags: Tan
    2: (70, 130, 180),      # Agents: Steel Blue
    -1: (178, 34, 34),      # Walls: Firebrick Red
    -2: (25, 25, 25),       # Pits: Dark Gray
    -3: (169, 169, 169),    # Filled Pits: Light Gray
    'START': (255, 215, 0), # Start: Yellow
    'GOAL': (60, 179, 113)  # Goal: Medium Sea Green
}

# Directions
DIRECTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, obstacle_density):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.agents = []
        self.sandbags = []
        self.pits = []
        self.pit_depth = {}
        self.step_count = 0
        self.pits_filled = 0
        self.total_distance = 0
        self.obstacle_density = obstacle_density
        self._initialize_environment()

    def _initialize_environment(self):

        
        # Calculate number of obstacles based on density
        total_obstacles = int(TOTAL_CELLS * self.obstacle_density)
        num_walls = int(total_obstacles * 0.4)  # 40% walls
        num_pits = int(total_obstacles * 0.3)   # 30% pits
        num_sands = int(total_obstacles * 0.3)  # 30% sandbags

        # Initialize walls
        for _ in range(num_walls):
            while True:
                x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if self.grid[x][y] == 0:
                    self.grid[x][y] = -1
                    break

        # Initialize pits
        for _ in range(num_pits):
            while True:
                x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if self.grid[x][y] == 0:
                    self.grid[x][y] = -2
                    self.pits.append((x, y))
                    self.pit_depth[(x, y)] = random.randint(1, 3)
                    break

        # Initialize sandbags
        for _ in range(num_sands):
            while True:
                x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if self.grid[x][y] == 0:
                    self.grid[x][y] = 1
                    self.sandbags.append((x, y))
                    break

        # Initialize agents, ensuring start and goal are not on obstacles
        for _ in range(NUM_AGENTS):
            while True:
                start_x, start_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                goal_x, goal_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if (self.grid[start_x][start_y] == 0 and 
                    self.grid[goal_x][goal_y] == 0 and 
                    (start_x, start_y) != (goal_x, goal_y) and
                    self.grid[start_x][start_y] != -1 and 
                    self.grid[goal_x][goal_y] != -1):
                    self.grid[start_x][start_y] = 2
                    self.agents.append(Agent((start_x, start_y), (goal_x, goal_y)))
                    break
    def coordinate_pit_filling(self):
        """Coordinate pit filling using the research paper's approach"""
        if not hasattr(self, 'pit_coordinator'):
            self.pit_coordinator = PitFillingCoordinator(self)
        
        self.pit_coordinator.calculate_traffic_index()
        
        # Check each pit for potential filling
        pits_to_fill = []
        for pit_pos in self.pits[:]:  # Copy list to avoid modification during iteration
            should_fill, assignment, cost = self.pit_coordinator.should_fill_pit(pit_pos)
            if should_fill and assignment:
                robot, sandbag_pos = assignment
                pits_to_fill.append((pit_pos, robot, sandbag_pos, cost))
        
        # Sort by cost efficiency (lowest cost first)
        pits_to_fill.sort(key=lambda x: x[3])
        
        # Execute pit filling assignments
        filled_count = 0
        for pit_pos, robot, sandbag_pos, cost in pits_to_fill[:3]:  # Limit to 3 fills per step
            if self._execute_pit_filling(pit_pos, robot, sandbag_pos):
                filled_count += 1
                logger.info(f'Coordinated pit filling: Robot at {robot.position} using sandbag at {sandbag_pos} to fill pit at {pit_pos}')
        
        return filled_count

    def _execute_pit_filling(self, pit_pos, robot, sandbag_pos):
        """Execute the actual pit filling if still valid"""
        # Check if pit and sandbag still exist
        if pit_pos not in self.pits or sandbag_pos not in self.sandbags:
            return False
        
        # Check if robot is still available and nearby
        robot_distance = abs(robot.position[0] - sandbag_pos[0]) + abs(robot.position[1] - sandbag_pos[1])
        if robot_distance > 5 or robot.reached_goal():  # Robot too far or done
            return False
        
        # Execute the filling
        depth = self.pit_depth[pit_pos]
        
        # Remove sandbag
        self.grid[sandbag_pos[0]][sandbag_pos[1]] = 0
        self.sandbags.remove(sandbag_pos)
        
        # Fill pit
        self.grid[pit_pos[0]][pit_pos[1]] = -3  # Filled pit
        self.pits.remove(pit_pos)
        self.pits_filled += 1
        
        # Add cost to robot
        fill_cost = BASE_COST_PIT_FILL * depth + BASE_COST_SAND
        robot.energy_cost += fill_cost
        
        # Invalidate cached paths for all agents since environment changed
        for agent in self.agents:
            agent.cached_path = None
            agent.grid_state = None
        
        return True
    
    

    def step(self, screen):
        # First, coordinate pit filling using the research approach
        if self.step_count % 5 == 0:  # Every 5 steps, check for pit filling opportunities
            self.coordinate_pit_filling()
        
        moves = {}
        for agent in self.agents:
            if not agent.reached_goal() and agent.energy_cost < agent.energy_limit:
                if agent.cached_path is None or any(self.grid[x][y] != agent.grid_state[x][y] for x in range(GRID_SIZE) for y in range(GRID_SIZE)):
                    agent.cached_path = agent.find_path(self)
                    agent.grid_state = [row[:] for row in self.grid]
                if agent.cached_path and len(agent.cached_path) > 1:
                    moves[agent] = agent.cached_path[1]

        position_counts = {}
        for agent, next_pos in moves.items():
            position_counts[next_pos] = position_counts.get(next_pos, 0) + 1

        for agent, next_pos in moves.items():
            if position_counts.get(next_pos, 0) > 1:
                pygame.time.delay(int(COLLISION_DELAY * 1000))
                agent.cached_path = agent.find_path(self)
                if agent.cached_path and len(agent.cached_path) > 1:
                    agent.move(agent.cached_path[1], self, screen)
            else:
                agent.move(next_pos, self, screen)

        self.step_count += 1

    def is_done(self):
        return all(agent.reached_goal() for agent in self.agents) or self.step_count >= MAX_STEPS

    def draw(self, screen):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                color = COLORS[self.grid[x][y]]
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)

        for agent in self.agents:
            if len(agent.path_history) > 1:
                for i in range(len(agent.path_history) - 1):
                    alpha = max(50, 255 - (len(agent.path_history) - i) * 20)
                    start_pos = (agent.path_history[i][1] * CELL_SIZE + CELL_SIZE // 2,
                                 agent.path_history[i][0] * CELL_SIZE + CELL_SIZE // 2)
                    end_pos = (agent.path_history[i + 1][1] * CELL_SIZE + CELL_SIZE // 2,
                               agent.path_history[i + 1][0] * CELL_SIZE + CELL_SIZE // 2)
                    surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(surface, (70, 130, 180, alpha), start_pos, end_pos, 3)
                    screen.blit(surface, (0, 0))

        pulse = (pygame.time.get_ticks() // 500) % 2
        for agent in self.agents:
            if pulse:
                start_rect = pygame.Rect(agent.start[1] * CELL_SIZE, agent.start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                goal_rect = pygame.Rect(agent.final_goal[1] * CELL_SIZE, agent.final_goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLORS['START'], start_rect)
                pygame.draw.rect(screen, COLORS['GOAL'], goal_rect)

        for agent in self.agents:
            center = (agent.position[1] * CELL_SIZE + CELL_SIZE // 2, agent.position[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, COLORS[2], center, CELL_SIZE // 3)
            pygame.draw.circle(screen, (0, 0, 0), center, CELL_SIZE // 3, 2)

        font = pygame.font.Font(None, 36)
        stats = f"Steps: {self.step_count}/{MAX_STEPS}  Total Energy: {sum(a.energy_cost for a in self.agents)}"
        text = font.render(stats, True, (255, 255, 255))
        screen.blit(text, (10, 10))

class Agent:
    def __init__(self, start_pos, final_goal):
        self.start = start_pos
        self.position = start_pos
        self.final_goal = final_goal
        self.energy_cost = 0
        self.energy_limit = 100
        self.path_history = [start_pos]
        self.cached_path = None
        self.grid_state = None
        self.goal_reported = False
        self.distance_traveled = 0
        self.shortest_path_length = abs(final_goal[0] - start_pos[0]) + abs(final_goal[1] - start_pos[1])  # Manhattan distance

    def move(self, new_pos, env, screen):
        if self.energy_cost >= self.energy_limit or self.reached_goal():
            return

        cost = COST_MOVE
        if env.grid[new_pos[0]][new_pos[1]] == 1:
            dx, dy = new_pos[0] - self.position[0], new_pos[1] - self.position[1]
            sandbag_new_pos = (new_pos[0] + dx, new_pos[1] + dy)
            if self._is_valid_move(sandbag_new_pos, env):
                collaborating_agents = sum(1 for a in env.agents if a.position in [(new_pos[0] + dx2, new_pos[1] + dy2) for dx2, dy2 in DIRECTIONS.values()])
                base_cost = BASE_COST_SAND / max(1, collaborating_agents)
                terrain_diff = TERRAIN_DIFFICULTY.get(env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]], 1)
                cost = base_cost * terrain_diff
                logger.info(f'Sandbag moved from {new_pos} to {sandbag_new_pos} for better agent movement')
                env.grid[new_pos[0]][new_pos[1]] = 0
                env.sandbags.remove(new_pos)
                if env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]] == -2:
                    depth = env.pit_depth[(sandbag_new_pos[0], sandbag_new_pos[1])]
                    env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]] = -3
                    env.pits.remove(sandbag_new_pos)
                    env.pits_filled += 1
                    cost = BASE_COST_PIT_FILL * depth
                    logger.info(f'Pit at {sandbag_new_pos} filled for better agent movement')
                else:
                    env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]] = 1
                    env.sandbags.append(sandbag_new_pos)
        else:
            cost = COST_MOVE

        self.distance_traveled += abs(new_pos[0] - self.position[0]) + abs(new_pos[1] - self.position[1])
        self.energy_cost += cost
        env.grid[self.position[0]][self.position[1]] = 0
        env.grid[new_pos[0]][new_pos[1]] = 2
        env.total_distance += abs(new_pos[0] - self.position[0]) + abs(new_pos[1] - self.position[1])
        self.position = new_pos
        self.path_history.append(new_pos)
        if self.reached_goal() and not self.goal_reported:
            logger.info(f'Agent (start: {self.start}) has reached goal state ({self.final_goal})')
            self.goal_reported = True

    def reached_goal(self):
        return self.position == self.final_goal

    def find_path(self, env):
        wavefront = [[float('inf')] * GRID_SIZE for _ in range(GRID_SIZE)]
        wavefront[self.final_goal[0]][self.final_goal[1]] = 0
        queue = deque([self.final_goal])
        visited = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            curr_cost = wavefront[current[0]][current[1]]
            for dx, dy in DIRECTIONS.values():
                next_pos = (current[0] + dx, current[1] + dy)
                if not self._is_valid_move(next_pos, env) or next_pos in visited:
                    continue
                additional_cost = COST_MOVE
                if env.grid[next_pos[0]][next_pos[1]] == 1:
                    beyond_x, beyond_y = next_pos[0] + dx, next_pos[1] + dy
                    if 0 <= beyond_x < GRID_SIZE and 0 <= beyond_y < GRID_SIZE:
                        additional_cost = BASE_COST_SAND * TERRAIN_DIFFICULTY.get(env.grid[beyond_x][beyond_y], 1)
                elif env.grid[next_pos[0]][next_pos[1]] == -2:
                    additional_cost = BASE_COST_PIT_FILL * env.pit_depth.get(next_pos, 1)
                new_cost = curr_cost + additional_cost
                if new_cost < wavefront[next_pos[0]][next_pos[1]]:
                    wavefront[next_pos[0]][next_pos[1]] = new_cost
                    queue.append(next_pos)

        path = []
        current = self.position
        path.append(current)
        steps = 0
        while current != self.final_goal and steps < MAX_PATH_STEPS:
            min_cost = float('inf')
            next_step = None
            for dx, dy in DIRECTIONS.values():
                neighbor = (current[0] + dx, current[1] + dy)
                if self._is_valid_move(neighbor, env) and neighbor not in path and wavefront[neighbor[0]][neighbor[1]] < min_cost:
                    min_cost = wavefront[neighbor[0]][neighbor[1]]
                    next_step = neighbor
            if next_step is None:
                break
            current = next_step
            path.append(current)
            steps += 1
        return path if path and path[-1] == self.final_goal else []

    def _is_valid_move(self, pos, env):
        x, y = pos
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and env.grid[x][y] != -1
    
class DirectionalWavefront:
    def __init__(self, env):
        self.env = env
        self.wavefront_cost = [[float('inf')] * GRID_SIZE for _ in range(GRID_SIZE)]
    
    def find_nearest_robot_path(self, sandbag_pos, pit_pos, agents):
        """
        Find the nearest robot path to a sandbag using directional wavefront propagation
        Prioritizes robots moving towards the pit direction
        """

        # Reset wavefront costs
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.wavefront_cost[i][j] = float('inf')

        # Calculate direction from sandbag to pit
        pit_direction = (pit_pos[0] - sandbag_pos[0], pit_pos[1] - sandbag_pos[1])
        
        # Normalize direction
        if pit_direction[0] != 0:
            pit_direction = (1 if pit_direction[0] > 0 else -1, pit_direction[1])
        if pit_direction[1] != 0:
            pit_direction = (pit_direction[0], 1 if pit_direction[1] > 0 else -1)
        
        
        
        # Initialize wavefront from sandbag position
        self.wavefront_cost[sandbag_pos[0]][sandbag_pos[1]] = 0
        queue = deque([sandbag_pos])
        
        while queue:
            current = queue.popleft()
            current_cost = self.wavefront_cost[current[0]][current[1]]
            
            for dx, dy in DIRECTIONS.values():
                next_pos = (current[0] + dx, current[1] + dy)
            
            # Build & sort neighbors so those pointing toward the pit go first
            neighbors = []
            for dx, dy in DIRECTIONS.values():
                nx, ny = current[0] + dx, current[1] + dy
                neighbors.append(( (dx, dy), (nx, ny) ))
            # Sort by dot(move_vector, pit_direction) descending
            neighbors.sort(key=lambda m: -(m[0][0]*pit_direction[0] + m[0][1]*pit_direction[1]))

            for (dx, dy), next_pos in neighbors:    
                if not self._is_valid_position(next_pos):
                    continue
                
                # Calculate directional cost based on movement towards pit
                directional_weight = self._calculate_directional_weight(
                    (dx, dy), pit_direction, current, pit_pos
                )
                
                new_cost = current_cost + directional_weight
                
                if new_cost < self.wavefront_cost[next_pos[0]][next_pos[1]]:
                    self.wavefront_cost[next_pos[0]][next_pos[1]] = new_cost
                    queue.append(next_pos)
                
        
        # Find the best robot based on wavefront costs and robot paths
        best_robot = None
        min_total_cost = float('inf')
        best_robot_point = None
        
        for agent in agents:
            if agent.reached_goal() or not agent.cached_path:
                continue
            
            # Check each point in the robot's path
            for i, path_point in enumerate(agent.cached_path):
                if self.wavefront_cost[path_point[0]][path_point[1]] < float('inf'):
                    # Calculate total cost: wavefront cost + deviation cost + return cost
                    wavefront_cost = self.wavefront_cost[path_point[0]][path_point[1]]
                    
                    # Distance from path point to sandbag
                    path_to_sandbag = abs(path_point[0] - sandbag_pos[0]) + abs(path_point[1] - sandbag_pos[1])
                    
                    # Distance from sandbag to pit (pushing cost)
                    sandbag_to_pit = abs(sandbag_pos[0] - pit_pos[0]) + abs(sandbag_pos[1] - pit_pos[1])
                    
                    # Distance to return to path (simplified)
                    return_cost = abs(pit_pos[0] - path_point[0]) + abs(pit_pos[1] - path_point[1])
                    
                    # Calculate robot's direction alignment with pit direction
                    robot_direction = self._get_robot_direction(agent, i)
                    direction_bonus = self._calculate_direction_alignment(robot_direction, pit_direction)
                    
                    total_cost = (wavefront_cost + 
                                path_to_sandbag + 
                                sandbag_to_pit * BASE_COST_SAND + 
                                return_cost - direction_bonus)
                    
                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_robot = agent
                        best_robot_point = path_point
        
        return best_robot, best_robot_point, min_total_cost
    
    def _calculate_directional_weight(self, move_direction, pit_direction, current_pos, pit_pos):
        """Calculate directional weight based on movement towards pit"""
        base_cost = 1.0
        
        # If moving away from pit, increase cost
        if (move_direction[0] != 0 and 
            ((move_direction[0] > 0 and pit_direction[0] < 0) or 
             (move_direction[0] < 0 and pit_direction[0] > 0))):
            base_cost *= 2.0
        
        if (move_direction[1] != 0 and 
            ((move_direction[1] > 0 and pit_direction[1] < 0) or 
             (move_direction[1] < 0 and pit_direction[1] > 0))):
            base_cost *= 2.0
        
        # If moving towards pit, reduce cost
        if (move_direction[0] != 0 and move_direction[0] == pit_direction[0]) or \
           (move_direction[1] != 0 and move_direction[1] == pit_direction[1]):
            base_cost *= 0.5
        
        return base_cost
    
    def _get_robot_direction(self, agent, path_index):
        """Get robot's movement direction at a specific path point"""
        if path_index >= len(agent.cached_path) - 1:
            # Use direction towards goal
            return (agent.final_goal[0] - agent.position[0], 
                   agent.final_goal[1] - agent.position[1])
        
        current = agent.cached_path[path_index]
        next_pos = agent.cached_path[path_index + 1]
        return (next_pos[0] - current[0], next_pos[1] - current[1])
    
    def _calculate_direction_alignment(self, robot_dir, pit_dir):
        """Calculate bonus for robots moving towards the pit"""
        alignment = 0
        if robot_dir[0] != 0 and robot_dir[0] == pit_dir[0]:
            alignment += 5
        if robot_dir[1] != 0 and robot_dir[1] == pit_dir[1]:
            alignment += 5
        return alignment
    
    def _is_valid_position(self, pos):
        """Check if position is valid for wavefront propagation"""
        x, y = pos
        return (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and 
                self.env.grid[x][y] != -1)  # Not a wall
    
class PitFillingCoordinator:
    def __init__(self, env):
        self.env = env
        self.wavefront = DirectionalWavefront(env)
        self.traffic_index = {}
        self.pit_fill_assignments = {}
    
    def calculate_traffic_index(self):
        """Calculate traffic index for each cell based on all agent paths"""
        self.traffic_index = {}
        
        for agent in self.env.agents:
            if agent.cached_path:
                for pos in agent.cached_path:
                    self.traffic_index[pos] = self.traffic_index.get(pos, 0) + 1
    
    def should_fill_pit(self, pit_pos):
        """Determine if a pit should be filled based on traffic and cost analysis"""
        if pit_pos not in self.env.pits:
            return False, None, float('inf')
        
        # Calculate traffic through this pit
        traffic = self.traffic_index.get(pit_pos, 0)
        if traffic == 0:
            return False, None, float('inf')
        
        # Find nearby sandbags using simplified kd-tree approach
        nearby_sandbags = self._find_nearby_sandbags(pit_pos, max_distance=10)
        
        if not nearby_sandbags:
            return False, None, float('inf')
        
        # Find best robot-sandbag combination using directional wavefront
        best_cost = float('inf')
        best_robot = None
        best_sandbag = None
        
        for sandbag_pos in nearby_sandbags:
            robot, robot_point, cost = self.wavefront.find_nearest_robot_path(
                sandbag_pos, pit_pos, self.env.agents
            )
            
            if robot and cost < best_cost:
                best_cost = cost
                best_robot = robot
                best_sandbag = sandbag_pos
        
        # Calculate average detour cost for all agents passing through this pit
        detour_costs = []
        for agent in self.env.agents:
            if agent.cached_path and pit_pos in agent.cached_path:
                detour_cost = self._calculate_detour_cost(agent, pit_pos)
                detour_costs.append(detour_cost)
        
        average_detour = sum(detour_costs) / max(1, len(detour_costs)) if detour_costs else 0
        
        # Decision: fill if removal cost < average detour cost * traffic weight
        traffic_weight = min(traffic, 3)  # Cap the traffic influence
        should_fill = best_cost < average_detour * traffic_weight
        
        return should_fill, (best_robot, best_sandbag), best_cost
    
    def _find_nearby_sandbags(self, pit_pos, max_distance=10):
        """Find nearby sandbags using simplified spatial search"""
        nearby = []
        for sandbag_pos in self.env.sandbags:
            distance = abs(pit_pos[0] - sandbag_pos[0]) + abs(pit_pos[1] - sandbag_pos[1])
            if distance <= max_distance:
                nearby.append(sandbag_pos)
        
        # Sort by distance (closest first)
        nearby.sort(key=lambda s: abs(pit_pos[0] - s[0]) + abs(pit_pos[1] - s[1]))
        
        # Return up to 5 closest sandbags
        return nearby[:5]
    
    def _calculate_detour_cost(self, agent, pit_pos):
        """Calculate the cost of detouring around a pit"""
        if not agent.cached_path or pit_pos not in agent.cached_path:
            return 0
        
        # Simplified detour calculation - assume agent needs to go around
        # This is a heuristic approximation
        return BASE_COST_PIT_FILL * self.env.pit_depth.get(pit_pos, 1) + 2

def simulate(obstacle_density, headless=True):
    """Run one simulation; if headless, skip all rendering calls."""
    
    if headless:
        # off-screen surface; no window pops up
        screen = pygame.Surface((WIDTH, HEIGHT))
        clock = None
    else:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Enhanced Sandbag Environment")
        clock = pygame.time.Clock()

    env = Environment(obstacle_density)

    import json

    # once, after you create env
    fixed = {
        'grid':        env.grid,
        'pits':        env.pits,
        'pit_depth':   { f"{x},{y}":depth for (x,y),depth in env.pit_depth.items() },
        'sandbags':    env.sandbags,
        'agents':      [ (agent.start, agent.final_goal) for agent in env.agents ]
    }
    with open('fixed_env.json', 'w') as f:
        json.dump(fixed, f, indent=2)
    print("[MAIN] fixed_env.json written")

    running = True
    while running and not env.is_done():
        if not headless:
            pygame.init()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        # always advance the simulation
        env.step(screen)

        if not headless:
            screen.fill((0, 0, 0))
            env.draw(screen)
            pygame.display.flip()
            clock.tick(100)

    # compute metrics (unchanged)
    total_energy   = sum(a.energy_cost for a in env.agents)
    success_count  = sum(1 for a in env.agents if a.reached_goal())
    total_steps    = sum(len(a.path_history) - 1 for a in env.agents)
    total_shortest = sum(a.shortest_path_length for a in env.agents)
    min_energy     = sum(COST_MOVE * a.shortest_path_length for a in env.agents)
    energy_eff     = min_energy / max(1, total_energy)
    path_opt       = total_shortest / max(1, total_steps)
    sandbag_util   = env.pits_filled / max(1, len(env.pits))
    eff_score      = W1 * energy_eff + W2 * path_opt + W4 * sandbag_util

    metrics = {
        'obstacle_density': obstacle_density,
        'energy_efficiency':    energy_eff,
        'path_optimization':    path_opt,
        'sandbag_utilization':  sandbag_util,
        'pits_filled':          env.pits_filled,
        'effectiveness_score':  eff_score,
        'total_pits':           len(env.pits),
        'total_energy':         total_energy,
        'total_distance':       env.total_distance,
        'success_count':        success_count
    }

    if not headless:
        pygame.quit()
    return metrics

def plot_graphs(metrics_list):
    obstacle_densities = [m['obstacle_density'] for m in metrics_list]
    energy_efficiencies = [m['energy_efficiency'] for m in metrics_list]
    path_optimizations = [m['path_optimization'] for m in metrics_list]
    sandbag_utilizations = [m['sandbag_utilization'] for m in metrics_list]
    pits_filled = [m['pits_filled'] for m in metrics_list]
    effectiveness_scores = [m['effectiveness_score'] for m in metrics_list]

    # Plot 1: Energy Efficiency vs Obstacle Density
    plt.figure(figsize=(8, 6))
    plt.plot([d * 100 for d in obstacle_densities], energy_efficiencies, 'o-', color='blue')
    plt.title('Energy Efficiency vs Obstacle Density')
    plt.xlabel('Obstacle Density (%)')
    plt.ylabel('Energy Efficiency (Min Energy / Total Energy)')
    plt.grid(True)
    plt.savefig('energy_efficiency_vs_obstacle_density.png')
    plt.close()

    # Plot 2: Path Optimization vs Obstacle Density
    plt.figure(figsize=(8, 6))
    plt.plot([d * 100 for d in obstacle_densities], path_optimizations, 'o-', color='green')
    plt.title('Path Optimization vs Obstacle Density')
    plt.xlabel('Obstacle Density (%)')
    plt.ylabel('Path Optimization (Shortest Path / Total Steps)')
    plt.grid(True)
    plt.savefig('path_optimization_vs_obstacle_density.png')
    plt.close()

    # Plot 3: Sandbag Utilization vs Obstacle Density
    plt.figure(figsize=(8, 6))
    plt.plot([d * 100 for d in obstacle_densities], sandbag_utilizations, 'o-', color='purple')
    plt.title('Sandbag Utilization vs Obstacle Density')
    plt.xlabel('Obstacle Density (%)')
    plt.ylabel('Sandbag Utilization (Filled Pits / Total Pits)')
    plt.grid(True)
    plt.savefig('sandbag_utilization_vs_obstacle_density.png')
    plt.close()

    # Plot 4: Pits Filled vs Obstacle Density
    plt.figure(figsize=(8, 6))
    plt.plot([d * 100 for d in obstacle_densities], pits_filled, 'o-', color='orange')
    plt.title('Number of Pits Filled vs Obstacle Density')
    plt.xlabel('Obstacle Density (%)')
    plt.ylabel('Pits Filled')
    plt.grid(True)
    plt.savefig('pits_filled_vs_obstacle_density.png')
    plt.close()

    # Plot 5: Effectiveness Score vs Obstacle Density
    plt.figure(figsize=(8, 6))
    plt.plot([d * 100 for d in obstacle_densities], effectiveness_scores, 'o-', color='cyan')
    plt.title('Effectiveness Score vs Obstacle Density')
    plt.xlabel('Obstacle Density (%)')
    plt.ylabel('Effectiveness Score')
    plt.grid(True)
    plt.savefig('effectiveness_score_vs_obstacle_density.png')
    plt.close()

if __name__ == "__main__":
    
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Sandbag Simulation')
    parser.add_argument('--mode',
                        choices=['single', 'batch', 'visual'],
                        default='visual',
                        help='single: one headless run; visual: one Pygame run; batch: sweep')
    parser.add_argument('--density', type=float, default=0.2,
                        help='Obstacle density for single/visual mode (0.0–1.0)')
    parser.add_argument('--runs',    type=int,   default=10,
                        help='Repetitions per density for batch mode')
    args = parser.parse_args()

    if args.mode in ('single', 'visual'):
        headless = (args.mode == 'single')
        logger.info(f"Running {args.mode} simulation at density {args.density:.2f}")
        t0 = time.time()
        metrics = simulate(args.density, headless=headless)
        dt = time.time() - t0
        print(f"[MAIN] simulate() took {dt:.2f}s")
        print("Results:", metrics)

    elif args.mode == 'batch':
        logger.info(f"Running batch simulations ({args.runs} runs per density)...")
        from math import isclose
        all_results = []
        for d in OBSTACLE_DENSITIES:
            logger.info(f" Density {d:.2f}")
            runs = []
            for i in range(args.runs):
                runs.append(simulate(d, headless=True))
            # compute averages here (as you did before)...
            # append to all_results
        # print summary table, plot, and save JSON exactly as before
