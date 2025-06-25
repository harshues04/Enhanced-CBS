import pygame
import random
from collections import deque
import logging
import matplotlib.pyplot as plt
import numpy as np
import uuid
import time
import os, json

random.seed(42)

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

    def step(self, screen):
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
