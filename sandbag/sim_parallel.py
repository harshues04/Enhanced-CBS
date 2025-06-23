import pygame
import random
from collections import deque
import logging
import matplotlib.pyplot as plt
import numpy as np
import heapq
import multiprocessing
from functools import partial

# ─── Insert below your existing imports ────────────────────────────────────────

def process_node(args, my_map, starts, goals, heuristics, splitter, k=5, horizon=100):
    """
    Process a CT node: check for solution or generate children (for parallel Pool).
    """
    node, conflict_freq = args
    # if no collisions, it's a solution
    if not node['collisions']:
        return True, node

    collision = node['collisions'][0]
    constraints = splitter(collision)
    children = []

    for constraint in constraints:
        child = {
            'cost': 0,
            'constraints': node['constraints'] + [constraint],
            'paths': node['paths'].copy(),
            'conflict_freq': dict(conflict_freq),
            'collisions': []
        }
        ai = constraint['agent']

        # low‐level A*
        astar = A_Star(my_map, starts, goals, heuristics, ai, child['constraints'])
        path = astar.find_paths()
        if path is None:
            continue
        child['paths'][ai] = path[0]

        # if positive constraint, replan others that might violate it
        if constraint.get('positive', False):
            violators = paths_violate_constraint(constraint, child['paths'])
            for v in violators:
                astar_v = A_Star(my_map, starts, goals, heuristics, v, child['constraints'])
                subpath = astar_v.find_paths()
                if subpath is None:
                    break
                child['paths'][v] = subpath[0]
            else:
                child['collisions'] = detect_collisions(child['paths'], goals, child['conflict_freq'], k, horizon)
                child['cost'] = get_sum_of_cost(child['paths'])
                children.append(child)
                continue

        # standard replanning
        child['collisions'] = detect_collisions(child['paths'], goals, child['conflict_freq'], k, horizon)
        child['cost'] = get_sum_of_cost(child['paths'])
        children.append(child)

    return False, children


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
OBSTACLE_DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
W1, W2, W4 = 0.33, 0.33, 0.34

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
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ============================================================================
# CBS Implementation (Adapted from cbs.py)
# ============================================================================

def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]

def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
        if len(path) > 1:
            assert path[-1] != path[-2]
    return rst

def detect_collision(path1, path2):
    t_range = max(len(path1), len(path2))
    for t in range(t_range):
        loc_c1 = get_location(path1, t)
        loc_c2 = get_location(path2, t)
        loc1 = get_location(path1, t + 1)
        loc2 = get_location(path2, t + 1)
        # Vertex collision
        if loc1 == loc2:
            return [loc1], t
        # Edge collision
        if [loc_c1, loc1] == [loc2, loc_c2]:
            return [loc2, loc_c2], t
    return None

def compute_criticality_score(collision, goals, conflict_freq, w1=1.0, w2=1.0, w3=1.0, T_freq=2, horizon=100):
    a1 = collision['a1']
    a2 = collision['a2']
    timestep = collision['timestep']
    loc = collision['loc'][0] if len(collision['loc']) == 1 else collision['loc'][0]
    loc = tuple(loc)
    
    dist_i = abs(loc[0] - goals[a1][0]) + abs(loc[1] - goals[a1][1])
    dist_j = abs(loc[0] - goals[a2][0]) + abs(loc[1] - goals[a2][1])
    total_dist = dist_i + dist_j
    
    if total_dist == 0:
        dist_score = w2 * 1.0
    else:
        dist_score = w2 * (1.0 / total_dist)
    
    freq_key = tuple(sorted([a1, a2]))
    freq_score = w1 * (1 if conflict_freq.get(freq_key, 0) >= T_freq else 0)
    time_score = w3 * (horizon - timestep)
    
    return freq_score + dist_score + time_score

def detect_collisions(paths, goals, conflict_freq, k=5, horizon=100):
    collisions = []
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            collision = detect_collision(paths[i], paths[j])
            if collision is not None:
                position, t = collision
                collisions.append({
                    'a1': i,
                    'a2': j,
                    'loc': position,
                    'timestep': t + 1
                })
    
    if not collisions:
        return []
    
    collisions.sort(key=lambda x: x['timestep'])
    collisions = collisions[:min(k, len(collisions))]
    
    scored_collisions = []
    for collision in collisions:
        score = compute_criticality_score(collision, goals, conflict_freq, horizon=horizon)
        scored_collisions.append((score, collision))
    
    selected_collision = max(scored_collisions, key=lambda x: x[0])[1]
    
    freq_key = tuple(sorted([selected_collision['a1'], selected_collision['a2']]))
    conflict_freq[freq_key] = conflict_freq.get(freq_key, 0) + 1
    
    return [selected_collision]

def standard_splitting(collision):
    constraints = []
    if len(collision['loc']) == 1:
        constraints.append({'agent': collision['a1'],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
        constraints.append({'agent': collision['a2'],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
    else:
        constraints.append({'agent': collision['a1'],
                            'loc': [collision['loc'][0], collision['loc'][1]],
                            'timestep': collision['timestep'],
                            'positive': False})
        constraints.append({'agent': collision['a2'],
                            'loc': [collision['loc'][1], collision['loc'][0]],
                            'timestep': collision['timestep'],
                            'positive': False})
    return constraints

def paths_violate_constraint(constraint, paths):
    """
    Check which other agents’ paths violate a positive constraint.
    """
    assert constraint['positive'] is True
    violating = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        # vertex constraint
        if len(constraint['loc']) == 1:
            if constraint['loc'][0] == curr:
                violating.append(i)
        else:  # edge constraint
            # either moving into or out of the forbidden edge
            if (constraint['loc'][0] == prev or
                constraint['loc'][1] == curr or
                constraint['loc'] == [curr, prev]):
                violating.append(i)
    return violating


def compute_heuristics(my_map, goal):
    # Open set for Dijkstra's algorithm
    open_set = []
    heapq.heappush(open_set, (0, goal))
    
    # Initialize distances
    distances = {}
    for i in range(len(my_map)):
        for j in range(len(my_map[i])):
            distances[(i, j)] = float('inf')
    distances[goal] = 0
    
    while open_set:
        current_dist, current = heapq.heappop(open_set)
        
        if current_dist > distances[current]:
            continue
        
        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < len(my_map) and 
                0 <= neighbor[1] < len(my_map[0]) and 
                my_map[neighbor[0]][neighbor[1]] != -1):  # Not a wall
                
                cost = 1  # Basic movement cost
                # Add terrain-specific costs
                if my_map[neighbor[0]][neighbor[1]] == 1:  # Sandbag
                    cost = BASE_COST_SAND
                elif my_map[neighbor[0]][neighbor[1]] == -2:  # Pit
                    cost = BASE_COST_PIT_FILL
                
                new_dist = distances[current] + cost
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(open_set, (new_dist, neighbor))
    
    return distances

class A_Star:
    def __init__(self, my_map, starts, goals, heuristics, agent_id, constraints):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.heuristics = heuristics
        self.agent_id = agent_id
        self.constraints = constraints
        self.open_set = []
        self.closed_set = set()
        
    def get_neighbors(self, curr_loc):
        neighbors = []
        for dx, dy in DIRECTIONS:
            new_loc = (curr_loc[0] + dx, curr_loc[1] + dy)
            if (0 <= new_loc[0] < len(self.my_map) and 
                0 <= new_loc[1] < len(self.my_map[0]) and 
                self.my_map[new_loc[0]][new_loc[1]] != -1):  # Not a wall
                neighbors.append(new_loc)
        return neighbors
    
    def is_constrained(self, curr_loc, next_loc, next_time):
        for constraint in self.constraints:
            if constraint['agent'] != self.agent_id:
                continue
            
            if constraint['timestep'] != next_time:
                continue
            
            if len(constraint['loc']) == 1:  # Vertex constraint
                if constraint['loc'][0] == next_loc:
                    return True
            else:  # Edge constraint
                if constraint['loc'] == [curr_loc, next_loc]:
                    return True
        return False
    
    def find_paths(self):
        start = self.starts[self.agent_id]
        goal = self.goals[self.agent_id]
        
        # Priority queue: (f_score, g_score, location, time, path)
        heapq.heappush(self.open_set, (0, 0, start, 0, [start]))
        
        visited = set()
        
        while self.open_set:
            f_score, g_score, curr_loc, curr_time, path = heapq.heappop(self.open_set)
            
            state = (curr_loc, curr_time)
            if state in visited:
                continue
            visited.add(state)
            
            if curr_loc == goal:
                return [path]
            
            # Wait action
            if not self.is_constrained(curr_loc, curr_loc, curr_time + 1):
                new_g = g_score + 1
                new_f = new_g + self.heuristics[self.agent_id][curr_loc]
                new_path = path + [curr_loc]
                if (curr_loc, curr_time + 1) not in visited:
                    heapq.heappush(self.open_set, (new_f, new_g, curr_loc, curr_time + 1, new_path))
            
            # Move actions
            for next_loc in self.get_neighbors(curr_loc):
                if self.is_constrained(curr_loc, next_loc, curr_time + 1):
                    continue
                
                cost = 1  # Basic movement cost
                # Add terrain-specific costs
                if self.my_map[next_loc[0]][next_loc[1]] == 1:  # Sandbag
                    cost = BASE_COST_SAND
                elif self.my_map[next_loc[0]][next_loc[1]] == -2:  # Pit
                    cost = BASE_COST_PIT_FILL
                
                new_g = g_score + cost
                new_f = new_g + self.heuristics[self.agent_id][next_loc]
                new_path = path + [next_loc]
                
                if (next_loc, curr_time + 1) not in visited:
                    heapq.heappush(self.open_set, (new_f, new_g, next_loc, curr_time + 1, new_path))
        
        return None

class CBSSolver:
    def __init__(self, my_map, starts, goals, max_time=30, max_nodes=1000):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []
        self.conflict_freq = {}
        self.max_time = max_time
        self.max_nodes = max_nodes
        self.start_time = None
        
        # Compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def get_adaptive_batch_size(self, base_batch_size):
        """
        Calculate adaptive batch size based on number of agents,
        depth of search, and open‐list size.
        """
        batch = base_batch_size
        agent_factor = max(1, self.num_of_agents // 5)
        depth_factor = max(1, self.num_of_expanded // 1000)
        batch = max(1, batch // (agent_factor * depth_factor))
        if len(self.open_list) > 10000:
            batch = max(1, batch // 2)
        return min(batch, len(self.open_list))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def should_terminate(self):
        import time as timer
        if self.start_time is None:
            return False
        if timer.time() - self.start_time > self.max_time:
            return True
        if self.num_of_expanded > self.max_nodes:
            return True
        return False

    def find_solution(self,batch_size=None, k=5, horizon=100):
            import time as timer
            self.start_time = timer.time()

            splitter = standard_splitting
            base_bs = multiprocessing.cpu_count() if batch_size is None else batch_size

            # build root
            root = {'cost':0, 'constraints':[], 'paths':[], 'collisions':[], 'conflict_freq':{}}
            for i in range(self.num_of_agents):
                astar = A_Star(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
                p = astar.find_paths()
                if p is None:
                    raise Exception("No solution for agent %d" % i)
                root['paths'].append(p[0])
            root['cost'] = get_sum_of_cost(root['paths'])
            root['collisions'] = detect_collisions(root['paths'], self.goals, root['conflict_freq'], k, horizon)
            self.push_node(root)

            best_cost = float('inf')
            best_sol  = None

            with multiprocessing.Pool(processes=base_bs) as pool:
                while self.open_list and not self.should_terminate():
                    # how many to expand this round?
                    cur_bs = self.get_adaptive_batch_size(base_bs)
                    batch = []
                    for _ in range(cur_bs):
                        if not self.open_list: break
                        node = self.pop_node()
                        if not node['collisions']:
                            return node['paths'], self.num_of_generated, self.num_of_expanded
                        batch.append(node)

                    # track best‐so‐far
                    for nd in batch:
                        if nd['cost'] < best_cost:
                            best_cost = nd['cost']
                            if not nd['collisions']:
                                best_sol = nd

                    # parallel expand
                    proc = partial(process_node,
                                my_map=self.my_map,
                                starts=self.starts,
                                goals=self.goals,
                                heuristics=self.heuristics,
                                splitter=splitter,
                                k=k,
                                horizon=horizon)
                    args = [(n, n['conflict_freq']) for n in batch]
                    results = pool.map(proc, args)

                    for is_sol, out in results:
                        if is_sol:
                            return out['paths'], self.num_of_generated, self.num_of_expanded
                        for child in out:
                            self.push_node(child)

            # fallback to best found
            if best_sol:
                return best_sol['paths'], self.num_of_generated, self.num_of_expanded
            return None


# ============================================================================
# Environment Classes (Adapted from sim1.py)
# ============================================================================

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
        self.cbs_paths = None
        self.path_index = 0
        self._initialize_environment()
        self._plan_paths()

    def _initialize_environment(self):
        # Calculate number of obstacles based on density
        total_obstacles = int(TOTAL_CELLS * self.obstacle_density)
        num_walls = int(total_obstacles * 0.4)
        num_pits = int(total_obstacles * 0.3)
        num_sands = int(total_obstacles * 0.3)

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

        # Initialize agents
        for _ in range(NUM_AGENTS):
            while True:
                start_x, start_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                goal_x, goal_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if (self.grid[start_x][start_y] == 0 and 
                    self.grid[goal_x][goal_y] == 0 and 
                    (start_x, start_y) != (goal_x, goal_y)):
                    self.grid[start_x][start_y] = 2
                    self.agents.append(Agent((start_x, start_y), (goal_x, goal_y)))
                    break

    def _plan_paths(self):
        """Use CBS to plan paths for all agents"""
        starts = [agent.start for agent in self.agents]
        goals  = [agent.final_goal for agent in self.agents]

        # Create a clean map for CBS (remove agents from grid)
        clean_map = [row[:] for row in self.grid]
        for agent in self.agents:
            clean_map[agent.start[0]][agent.start[1]] = 0

        cbs_solver = CBSSolver(clean_map, starts, goals)
        # CBS now returns (paths, num_generated, num_expanded)
        result = cbs_solver.find_solution()
        if result is None:
            logger.warning("CBS failed to find any solution")
            for agent in self.agents:
                agent.cbs_path = None
            return

        # unpack: first element is the list of per-agent paths
        self.cbs_paths, self.num_generated, self.num_expanded = result
        logger.info(f"CBS found paths for all {len(self.agents)} agents")

        for i, agent in enumerate(self.agents):
            agent.cbs_path = self.cbs_paths[i]


    def step(self, screen):
        """Execute one simulation step using CBS paths"""
        if self.cbs_paths and self.path_index < max(len(path) for path in self.cbs_paths):
            # Move agents according to CBS paths
            for i, agent in enumerate(self.agents):
                if (not agent.reached_goal() and 
                    agent.energy_cost < agent.energy_limit and
                    self.path_index < len(self.cbs_paths[i])):
                    
                    next_pos = self.cbs_paths[i][self.path_index]
                    if next_pos != agent.position:
                        agent.move(next_pos, self, screen)
            
            self.path_index += 1
        else:
            # Fallback to individual pathfinding if CBS failed or paths completed
            for agent in self.agents:
                if not agent.reached_goal() and agent.energy_cost < agent.energy_limit:
                    if agent.cached_path is None:
                        agent.cached_path = agent.find_path(self)
                    if agent.cached_path and len(agent.cached_path) > 1:
                        agent.move(agent.cached_path[1], self, screen)
        
        self.step_count += 1

    def is_done(self):
        return all(agent.reached_goal() for agent in self.agents) or self.step_count >= MAX_STEPS

    def draw(self, screen):
        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                color = COLORS[self.grid[x][y]]
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)

        # Draw path histories
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

        # Draw start/goal positions with pulsing effect
        pulse = (pygame.time.get_ticks() // 500) % 2
        for agent in self.agents:
            if pulse:
                start_rect = pygame.Rect(agent.start[1] * CELL_SIZE, agent.start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                goal_rect = pygame.Rect(agent.final_goal[1] * CELL_SIZE, agent.final_goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLORS['START'], start_rect)
                pygame.draw.rect(screen, COLORS['GOAL'], goal_rect)

        # Draw agents
        for agent in self.agents:
            center = (agent.position[1] * CELL_SIZE + CELL_SIZE // 2, agent.position[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, COLORS[2], center, CELL_SIZE // 3)
            pygame.draw.circle(screen, (0, 0, 0), center, CELL_SIZE // 3, 2)

        # Draw statistics
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
        self.cbs_path = None
        self.goal_reported = False
        self.distance_traveled = 0
        self.shortest_path_length = abs(final_goal[0] - start_pos[0]) + abs(final_goal[1] - start_pos[1])

    def move(self, new_pos, env, screen):
        if self.energy_cost >= self.energy_limit or self.reached_goal():
            return

        cost = COST_MOVE
        if env.grid[new_pos[0]][new_pos[1]] == 1:  # Sandbag
            # Handle sandbag pushing logic
            dx, dy = new_pos[0] - self.position[0], new_pos[1] - self.position[1]
            sandbag_new_pos = (new_pos[0] + dx, new_pos[1] + dy)
            if self._is_valid_move(sandbag_new_pos, env):
                cost = BASE_COST_SAND
                env.grid[new_pos[0]][new_pos[1]] = 0
                env.sandbags.remove(new_pos)
                if env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]] == -2:  # Pit
                    depth = env.pit_depth[(sandbag_new_pos[0], sandbag_new_pos[1])]
                    env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]] = -3
                    env.pits.remove(sandbag_new_pos)
                    env.pits_filled += 1
                    cost = BASE_COST_PIT_FILL * depth
                    logger.info(f'Pit at {sandbag_new_pos} filled')
                else:
                    env.grid[sandbag_new_pos[0]][sandbag_new_pos[1]] = 1
                    env.sandbags.append(sandbag_new_pos)

        self.distance_traveled += abs(new_pos[0] - self.position[0]) + abs(new_pos[1] - self.position[1])
        self.energy_cost += cost
        env.grid[self.position[0]][self.position[1]] = 0
        env.grid[new_pos[0]][new_pos[1]] = 2
        env.total_distance += abs(new_pos[0] - self.position[0]) + abs(new_pos[1] - self.position[1])
        self.position = new_pos
        self.path_history.append(new_pos)
        
        if self.reached_goal() and not self.goal_reported:
            logger.info(f'Agent (start: {self.start}) reached goal {self.final_goal}')
            self.goal_reported = True

    def reached_goal(self):
        return self.position == self.final_goal

    def find_path(self, env):
        """Fallback pathfinding using wavefront algorithm"""
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
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if not self._is_valid_move(next_pos, env) or next_pos in visited:
                    continue
                    
                additional_cost = COST_MOVE
                if env.grid[next_pos[0]][next_pos[1]] == 1:
                    additional_cost = BASE_COST_SAND
                elif env.grid[next_pos[0]][next_pos[1]] == -2:
                    additional_cost = BASE_COST_PIT_FILL * env.pit_depth.get(next_pos, 1)
                    
                new_cost = curr_cost + additional_cost
                if new_cost < wavefront[next_pos[0]][next_pos[1]]:
                    wavefront[next_pos[0]][next_pos[1]] = new_cost
                    queue.append(next_pos)

        # Reconstruct path
        path = []
        current = self.position
        path.append(current)
        steps = 0
        
        while current != self.final_goal and steps < MAX_PATH_STEPS:
            min_cost = float('inf')
            next_step = None
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (self._is_valid_move(neighbor, env) and 
                    neighbor not in path and 
                    wavefront[neighbor[0]][neighbor[1]] < min_cost):
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
        return (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and 
                env.grid[x][y] != -1 and env.grid[x][y] != 2)  # Not wall or agent

# ============================================================================
# Simulation Runner and Statistics
# ============================================================================

def run_simulation(obstacle_density, headless=True):
    """Run a single simulation with given parameters"""
    pygame.init()
    if not headless:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Multi-Agent Pathfinding Simulation")
        clock = pygame.time.Clock()
    else:
        screen = pygame.Surface((WIDTH, HEIGHT))
    
    env = Environment(obstacle_density)
    
    # Run simulation
    while not env.is_done():
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
        
        env.step(screen)
        
        if not headless:
            screen.fill((0, 0, 0))
            env.draw(screen)
            pygame.display.flip()
            clock.tick(10)  # Limit to 10 FPS for visualization
    
    # Calculate performance metrics
    agents_reached = sum(1 for agent in env.agents if agent.reached_goal())
    success_rate = agents_reached / NUM_AGENTS
    total_energy = sum(agent.energy_cost for agent in env.agents)
    avg_energy = total_energy / NUM_AGENTS
    
    # Path efficiency calculation
    total_efficiency = 0
    for agent in env.agents:
        if agent.reached_goal():
            efficiency = agent.shortest_path_length / max(agent.distance_traveled, 1)
            total_efficiency += efficiency
    avg_efficiency = total_efficiency / max(agents_reached, 1)
    
    results = {
        'obstacle_density': obstacle_density,
        'agents_reached': agents_reached,
        'success_rate': success_rate,
        'total_steps': env.step_count,
        'total_energy': total_energy,
        'avg_energy': avg_energy,
        'pits_filled': env.pits_filled,
        'avg_efficiency': avg_efficiency,
        'total_distance': env.total_distance
    }
    
    if not headless:
        pygame.quit()
    
    return results

def run_batch_simulations(num_runs=10):
    """Run multiple simulations across different obstacle densities"""
    all_results = []
    
    for density in OBSTACLE_DENSITIES:
        logger.info(f"Running simulations for obstacle density: {density}")
        density_results = []
        
        for run in range(num_runs):
            logger.info(f"  Run {run + 1}/{num_runs}")
            result = run_simulation(density, headless=True)
            if result:
                density_results.append(result)
        
        if density_results:
            # Calculate averages for this density
            avg_result = {
                'obstacle_density': density,
                'avg_success_rate': np.mean([r['success_rate'] for r in density_results]),
                'avg_total_steps': np.mean([r['total_steps'] for r in density_results]),
                'avg_total_energy': np.mean([r['total_energy'] for r in density_results]),
                'avg_energy_per_agent': np.mean([r['avg_energy'] for r in density_results]),
                'avg_pits_filled': np.mean([r['pits_filled'] for r in density_results]),
                'avg_path_efficiency': np.mean([r['avg_efficiency'] for r in density_results]),
                'avg_total_distance': np.mean([r['total_distance'] for r in density_results]),
                'std_success_rate': np.std([r['success_rate'] for r in density_results]),
                'std_total_energy': np.std([r['total_energy'] for r in density_results])
            }
            all_results.append(avg_result)
    
    return all_results

def plot_results(results):
    """Create visualization plots for the simulation results"""
    if not results:
        logger.error("No results to plot")
        return
    
    densities = [r['obstacle_density'] for r in results]
    success_rates = [r['avg_success_rate'] for r in results]
    total_energies = [r['avg_total_energy'] for r in results]
    path_efficiencies = [r['avg_path_efficiency'] for r in results]
    total_steps = [r['avg_total_steps'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success Rate vs Obstacle Density
    ax1.plot(densities, success_rates, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Obstacle Density')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs Obstacle Density')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Total Energy vs Obstacle Density
    ax2.plot(densities, total_energies, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Obstacle Density')
    ax2.set_ylabel('Average Total Energy')
    ax2.set_title('Energy Consumption vs Obstacle Density')
    ax2.grid(True, alpha=0.3)
    
    # Path Efficiency vs Obstacle Density
    ax3.plot(densities, path_efficiencies, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Obstacle Density')
    ax3.set_ylabel('Average Path Efficiency')
    ax3.set_title('Path Efficiency vs Obstacle Density')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # Total Steps vs Obstacle Density
    ax4.plot(densities, total_steps, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Obstacle Density')
    ax4.set_ylabel('Average Total Steps')
    ax4.set_title('Simulation Steps vs Obstacle Density')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the simulation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Agent Pathfinding Simulation')
    parser.add_argument('--mode', choices=['single', 'batch', 'visual'], default='visual',
                        help='Simulation mode: single run, batch analysis, or visual demo')
    parser.add_argument('--density', type=float, default=0.2,
                        help='Obstacle density for single/visual mode (0.0-1.0)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs per density for batch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'visual':
        logger.info("Running visual simulation...")
        result = run_simulation(args.density, headless=False)
        if result:
            logger.info(f"Simulation completed: {result}")
    
    elif args.mode == 'single':
        logger.info(f"Running single simulation with density {args.density}...")
        result = run_simulation(args.density, headless=True)
        if result:
            logger.info(f"Results: {result}")
    
    elif args.mode == 'batch':
        logger.info(f"Running batch simulations ({args.runs} runs per density)...")
        results = run_batch_simulations(args.runs)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("BATCH SIMULATION RESULTS")
        logger.info("="*80)
        logger.info(f"{'Density':<10} {'Success':<10} {'Energy':<12} {'Efficiency':<12} {'Steps':<10}")
        logger.info("-"*80)
        
        for result in results:
            logger.info(f"{result['obstacle_density']:<10.2f} "
                       f"{result['avg_success_rate']:<10.3f} "
                       f"{result['avg_total_energy']:<12.1f} "
                       f"{result['avg_path_efficiency']:<12.3f} "
                       f"{result['avg_total_steps']:<10.1f}")
        
        # Create plots
        plot_results(results)
        
        # Save results to file
        import json
        with open('simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to simulation_results.json")

if __name__ == "__main__":
    main()