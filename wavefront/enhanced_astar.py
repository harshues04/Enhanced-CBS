#!/usr/bin/env python3
"""
Enhanced A* Implementation with Environment Awareness

This module extends the original A* to handle:
- Dynamic environment changes (pit filling)
- Sandbag movement costs
- Environment-aware heuristics
"""

import heapq
import time as timer
import numpy as np
import copy
from itertools import product

def move(loc, dir):
    """Move from location in given direction"""
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def compute_heuristics_with_env(my_map, goal, env_manager=None):
    """
    Enhanced heuristic computation that considers environment modifications
    Uses Dijkstra to build shortest-path tree considering potential pit filling
    """
    # Convert goal to tuple if it's a list
    if isinstance(goal, list):
        goal = tuple(goal)
    
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            
            if (child_loc[0] < 0 or child_loc[0] >= len(my_map) or
                child_loc[1] < 0 or child_loc[1] >= len(my_map[0])):
                continue
            
            # Check if location is blocked
            if my_map[child_loc[0]][child_loc[1]]:
                # If it's a pit that could potentially be filled, allow with higher cost
                if env_manager and env_manager.is_pit_at(child_loc):
                    # Add penalty for pit filling requirement
                    child_cost = cost + 5  # Higher cost to account for pit filling
                else:
                    continue  # Skip permanent obstacles
            
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # Build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

def compute_heuristics(my_map, goal):
    """Original heuristic computation for backward compatibility"""
    return compute_heuristics_with_env(my_map, goal, None)

def get_location(path, time):
    """Get agent location at specific time step"""
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location
    
def get_path(goal_node, meta_agent):
    """Extract path from goal node"""
    path = []
    for i in range(len(meta_agent)):
        path.append([])
    curr = goal_node
    while curr is not None:
        for i in range(len(meta_agent)):
            path[i].append(curr['loc'][i])
        curr = curr['parent']
    for i in range(len(meta_agent)):
        path[i].reverse()
        assert path[i] is not None

        if len(path[i]) > 1: 
            # remove trailing duplicates
            while len(path[i]) > 1 and path[i][-1] == path[i][-2]:
                path[i].pop()
                if len(path[i]) <= 1:
                    break

    assert path is not None
    return path

class Sandbag:
    """Represents a sandbag in the environment"""
    
    def __init__(self, position, move_cost=2.0, assigned_agent=None):
        self.position = tuple(position) if isinstance(position, list) else position
        self.move_cost = move_cost
        self.assigned_agent = assigned_agent
        self.is_filled = False  # Whether it's used to fill a pit
        
    def assign_to_agent(self, agent_id):
        """Assign sandbag to an agent"""
        self.assigned_agent = agent_id
        
    def move_to(self, new_position):
        """Move sandbag to new position"""
        self.position = tuple(new_position) if isinstance(new_position, list) else new_position
        
    def fill_pit(self):
        """Mark sandbag as used to fill a pit"""
        self.is_filled = True

class Pit:
    """Represents a pit in the environment"""
    
    def __init__(self, position, is_filled=False):
        self.position = tuple(position) if isinstance(position, list) else position
        self.is_filled = is_filled
        self.filled_by_sandbag = None
        
    def fill_with_sandbag(self, sandbag):
        """Fill pit with a sandbag"""
        self.is_filled = True
        self.filled_by_sandbag = sandbag
        sandbag.fill_pit()

class EnvironmentManager:
    """Manages dynamic environment elements like pits and sandbags"""
    
    def __init__(self, my_map):
        self.my_map = my_map
        self.pits = {}  # position -> Pit object
        self.sandbags = {}  # position -> Sandbag object
        self.original_map = copy.deepcopy(my_map)
        
    def add_pit(self, position):
        """Add a pit at given position"""
        pos = tuple(position) if isinstance(position, list) else position
        self.pits[pos] = Pit(pos)
        if 0 <= pos[0] < len(self.my_map) and 0 <= pos[1] < len(self.my_map[0]):
            self.my_map[pos[0]][pos[1]] = True  # Mark as obstacle
            
    def add_sandbag(self, position, move_cost=2.0, assigned_agent=None):
        """Add a sandbag at given position"""
        pos = tuple(position) if isinstance(position, list) else position
        self.sandbags[pos] = Sandbag(pos, move_cost, assigned_agent)
        
    def is_pit_at(self, position):
        """Check if there's a pit at given position"""
        pos = tuple(position) if isinstance(position, list) else position
        return pos in self.pits and not self.pits[pos].is_filled
        
    def is_sandbag_at(self, position):
        """Check if there's a sandbag at given position"""
        pos = tuple(position) if isinstance(position, list) else position
        return pos in self.sandbags and not self.sandbags[pos].is_filled
        
    def get_sandbag_at(self, position):
        """Get sandbag at given position"""
        pos = tuple(position) if isinstance(position, list) else position
        return self.sandbags.get(pos)
        
    def get_pit_at(self, position):
        """Get pit at given position"""
        pos = tuple(position) if isinstance(position, list) else position
        return self.pits.get(pos)
        
    def move_sandbag(self, from_pos, to_pos, agent_id):
        """Move sandbag from one position to another"""
        from_pos = tuple(from_pos) if isinstance(from_pos, list) else from_pos
        to_pos = tuple(to_pos) if isinstance(to_pos, list) else to_pos
        
        if from_pos in self.sandbags:
            sandbag = self.sandbags[from_pos]
            if sandbag.assigned_agent == agent_id:
                del self.sandbags[from_pos]
                sandbag.move_to(to_pos)
                self.sandbags[to_pos] = sandbag
                return True
        return False
        
    def fill_pit_with_sandbag(self, pit_pos, sandbag_pos, agent_id):
        """Fill a pit with a sandbag"""
        pit_pos = tuple(pit_pos) if isinstance(pit_pos, list) else pit_pos
        sandbag_pos = tuple(sandbag_pos) if isinstance(sandbag_pos, list) else sandbag_pos
        
        if (pit_pos in self.pits and sandbag_pos in self.sandbags and 
            self.sandbags[sandbag_pos].assigned_agent == agent_id):
            
            pit = self.pits[pit_pos]
            sandbag = self.sandbags[sandbag_pos]
            
            pit.fill_with_sandbag(sandbag)
            del self.sandbags[sandbag_pos]
            
            # Update map to make pit traversable
            if 0 <= pit_pos[0] < len(self.my_map) and 0 <= pit_pos[1] < len(self.my_map[0]):
                self.my_map[pit_pos[0]][pit_pos[1]] = False
                
            return True
        return False
        
    def reset_environment(self):
        """Reset environment to original state"""
        self.my_map = copy.deepcopy(self.original_map)
        for pit in self.pits.values():
            pit.is_filled = False
            pit.filled_by_sandbag = None
        for sandbag in self.sandbags.values():
            sandbag.is_filled = False

class EnhancedAStar(object):
    """Enhanced A* with environment awareness and sandbag handling"""
    
    def __init__(self, my_map, starts, goals, heuristics, agents, constraints, env_manager=None):
        self.my_map = my_map
        self.num_generated = 0
        self.num_expanded = 0
        self.CPU_time = 0
        self.open_list = []
        self.closed_list = dict()
        self.constraints = constraints
        self.agents = agents
        self.env_manager = env_manager

        # check if meta_agent is only a simple agent (from basic CBS)
        if not isinstance(agents, list):
            self.agents = [agents]

            # add meta_agent keys to constraints
            for c in self.constraints:
                c['meta_agent'] = {c['agent']}

        # FILTER BY INDEX FOR STARTS AND GOALS AND HEURISTICS
        self.starts = [starts[a] for a in self.agents]
        self.heuristics = [heuristics[a] for a in self.agents]
        self.goals = [goals[a] for a in self.agents]

        self.c_table = []
        self.max_constraints = np.zeros((len(self.agents),), dtype=int)

    def push_node(self, node):
        f_value = node['g_val'] + node['h_val']
        heapq.heappush(self.open_list, (f_value, node['h_val'], node['loc'], self.num_generated, node))
        self.num_generated += 1
        
    def pop_node(self):
        _,_,_, id, curr = heapq.heappop(self.open_list)
        self.num_expanded += 1
        return curr

    def build_constraint_table(self, agent):
        """Build constraint table for agent"""
        constraint_table = dict()

        if not self.constraints:
            return constraint_table
        for constraint in self.constraints:
            timestep = constraint['timestep']

            t_constraint = []
            if timestep in constraint_table:
                t_constraint = constraint_table[timestep]

            # positive constraint for agent
            if constraint['positive'] and constraint['agent'] == agent:
                t_constraint.append(constraint)
                constraint_table[timestep] = t_constraint
            # and negative (external) constraint for agent
            elif not constraint['positive'] and constraint['agent'] == agent:
                t_constraint.append(constraint)
                constraint_table[timestep] = t_constraint
            # enforce positive constraints from other agents (i.e. create neg constraint)
            elif constraint['positive']: 
                neg_constraint = copy.deepcopy(constraint)
                neg_constraint['agent'] = agent
                # if edge collision
                if len(constraint['loc']) == 2:
                    # switch traversal direction
                    prev_loc = constraint['loc'][1]
                    curr_loc = constraint['loc'][0]
                    neg_constraint['loc'] = [prev_loc, curr_loc]
                neg_constraint['positive'] = False
                t_constraint.append(neg_constraint)
                constraint_table[timestep] = t_constraint
        
        return constraint_table

    def constraint_violated(self, curr_loc, next_loc, timestep, c_table_agent, agent):
        """Check if move violates any constraints"""
        if timestep not in c_table_agent:
            return None
        
        for constraint in c_table_agent[timestep]:
            if agent == constraint['agent']:
                # vertex constraint
                if len(constraint['loc']) == 1:
                    # positive constraint
                    if constraint['positive'] and next_loc != constraint['loc'][0]:
                        return constraint
                    # negative constraint
                    elif not constraint['positive'] and next_loc == constraint['loc'][0]:
                        return constraint
                # edge constraint
                else:
                    if constraint['positive'] and constraint['loc'] != [curr_loc, next_loc]:
                        return constraint
                    if not constraint['positive'] and constraint['loc'] == [curr_loc, next_loc]:
                        return constraint

        return None

    def future_constraint_violated(self, curr_loc, timestep, max_timestep, c_table_agent, agent):
        """Check if staying at location violates future constraints"""
        for t in range(timestep+1, max_timestep+1):
            if t not in c_table_agent:
                continue

            for constraint in c_table_agent[t]:
                if agent == constraint['agent']:
                    # vertex constraint
                    if len(constraint['loc']) == 1:
                        # positive constraint
                        if constraint['positive'] and curr_loc != constraint['loc'][0]:
                            return True
                        # negative constraint
                        elif not constraint['positive'] and curr_loc == constraint['loc'][0]:
                            return True

        return False

    def get_movement_cost(self, curr_loc, next_loc, agent_id):
        """
        Calculate movement cost considering sandbag movement
        
        Returns:
            float: Movement cost (1.0 for normal movement, higher for sandbag movement)
        """
        base_cost = 1.0
        
        # Check if agent is moving a sandbag
        if self.env_manager:
            sandbag = self.env_manager.get_sandbag_at(curr_loc)
            if sandbag and sandbag.assigned_agent == self.agents[agent_id]:
                # Higher cost for moving sandbag
                return base_cost * sandbag.move_cost
        
        return base_cost
            
    def generate_child_nodes(self, curr):
        """Generate child nodes with environment awareness"""
        children = []
        ma_dirs = product(list(range(5)), repeat=len(self.agents))
        
        for dirs in ma_dirs: 
            invalid_move = False
            child_loc = []
            movement_costs = []
            
            # move each agent for new timestep & check for (internal) conflicts with each other
            for i, a in enumerate(self.agents):           
                aloc = move(curr['loc'][i], dirs[i])
                # vertex collision; check for duplicates in child_loc
                if aloc in child_loc:
                    invalid_move = True
                    break
                child_loc.append(move(curr['loc'][i], dirs[i]))
                
                # Calculate movement cost for this agent
                move_cost = self.get_movement_cost(curr['loc'][i], aloc, i)
                movement_costs.append(move_cost)

            if invalid_move:
                continue

            for i, a in enumerate(self.agents):   
                # edge collision: check for matching locs in curr_loc and child_loc between two agents
                for j, a in enumerate(self.agents):   
                    if i != j:
                        if child_loc[i] == curr['loc'][j] and child_loc[j] == curr['loc'][i]:
                            invalid_move = True             
            
            if invalid_move:
                continue

            # check map constraints and external constraints
            for i, a in enumerate(self.agents):  
                next_loc= child_loc[i]
                # agent out of map bounds
                if next_loc[0]<0 or next_loc[0]>=len(self.my_map) or next_loc[1]<0 or next_loc[1]>=len(self.my_map[0]):
                    invalid_move = True
                # agent collision with map obstacle
                elif self.my_map[next_loc[0]][next_loc[1]]:
                    # Special handling for pits if environment manager is available
                    if self.env_manager and self.env_manager.is_pit_at(next_loc):
                        # Agent cannot move into unfilled pit
                        invalid_move = True
                    else:
                        invalid_move = True
                # Check for sandbag collisions
                elif self.env_manager and self.env_manager.is_sandbag_at(next_loc):
                    sandbag = self.env_manager.get_sandbag_at(next_loc)
                    if sandbag.assigned_agent != self.agents[i]:
                        # Cannot move into cell with unassigned sandbag
                        invalid_move = True
                # agent is constrained by a negative external constraint
                elif self.constraint_violated(curr['loc'][i],next_loc,curr['timestep']+1,self.c_table[i], self.agents[i]):
                    invalid_move = True
                if invalid_move:
                    break

            if invalid_move:
                continue

            # find h_values for current moves - convert to tuples for hashing
            h_value = 0
            for i in range(len(self.agents)):
                child_pos = tuple(child_loc[i]) if isinstance(child_loc[i], list) else child_loc[i]
                h_value += self.heuristics[i][child_pos]

            # Calculate g_value considering movement costs
            num_moves = curr['reached_goal'].count(False)
            total_movement_cost = sum(movement_costs[i] for i in range(len(self.agents)) if not curr['reached_goal'][i])
            g_value = curr['g_val'] + total_movement_cost

            reached_goal = [False for i in range(len(self.agents))]

            for i, a in enumerate(self.agents):
                if not reached_goal[i] and child_loc[i] == self.goals[i]:
                    if curr['timestep']+1 <= self.max_constraints[i]:
                        if not self.future_constraint_violated(child_loc[i], curr['timestep']+1, self.max_constraints[i] ,self.c_table[i], self.agents[i]):
                            reached_goal[i] = True
                    else:
                        reached_goal[i] = True

            child = {'loc': child_loc,
                    'g_val': g_value,
                    'h_val': h_value,
                    'parent': curr,
                    'timestep': curr['timestep']+1,
                    'reached_goal': copy.deepcopy(reached_goal)
                    } 

            children.append(child)

        return children

    def compare_nodes(self, n1, n2):
        return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

    def find_paths(self):
        """Find paths using enhanced A* with environment awareness"""
        self.start_time = timer.time()

        for i, a in enumerate(self.agents):
            table_i = self.build_constraint_table(a)
            self.c_table.append(table_i)
            if table_i.keys():
                self.max_constraints[i] = max(table_i.keys())

        # Convert starts and goals to tuples for hashing
        starts_tuples = [tuple(start) if isinstance(start, list) else start for start in self.starts]
        goals_tuples = [tuple(goal) if isinstance(goal, list) else goal for goal in self.goals]

        h_value = sum([self.heuristics[i][starts_tuples[i]] for i in range(len(self.agents))])

        root = {'loc': [self.starts[j] for j in range(len(self.agents))],
                'g_val': 0, 
                'h_val': h_value, 
                'parent': None,
                'timestep': 0,
                'reached_goal': [False for i in range(len(self.agents))]
                }

        # check if any any agents are already at goal loc
        for i, a in enumerate(self.agents):
            if self.starts[i] == self.goals[i]:
                if root['timestep'] <= self.max_constraints[i]:
                    if not self.future_constraint_violated(self.starts[i], root['timestep'], self.max_constraints[i] ,self.c_table[i], self.agents[i]):
                        root['reached_goal'][i] = True
                        self.max_constraints[i] = 0

        self.push_node(root)
        self.closed_list[(tuple(tuple(pos) for pos in root['loc']),root['timestep'])] = root

        while len(self.open_list) > 0:
            curr = self.pop_node()

            solution_found = all(curr['reached_goal'][i] for i in range(len(self.agents)))

            if solution_found:
                self.CPU_time = timer.time() - self.start_time
                return get_path(curr, self.agents)

            children = self.generate_child_nodes(curr)

            for child in children:
                # Create hashable key for child location
                child_key = (tuple(tuple(pos) for pos in child['loc']), child['timestep'])
                
                if child_key in self.closed_list:
                    existing = self.closed_list[child_key]
                    if (child['g_val'] + child['h_val'] < existing['g_val'] + existing['h_val']) and (child['g_val'] < existing['g_val']) and child['reached_goal'].count(False) <= existing['reached_goal'].count(False):
                        self.closed_list[child_key] = child
                        self.push_node(child)
                else:
                    self.closed_list[child_key] = child
                    self.push_node(child)

        self.CPU_time = timer.time() - self.start_time
        print('No solution found')
        return None

    def get_statistics(self):
        """Get search statistics"""
        return {
            'expanded': self.num_expanded,
            'generated': self.num_generated,
            'cpu_time': self.CPU_time
        }

# Backward compatibility - use enhanced A* but maintain original interface
class A_Star(EnhancedAStar):
    """Backward compatible A* class"""
    def __init__(self, my_map, starts, goals, heuristics, agents, constraints):
        super().__init__(my_map, starts, goals, heuristics, agents, constraints, env_manager=None)

def create_environment_manager(my_map, pits=None, sandbags=None):
    """
    Utility function to create an EnvironmentManager with initial pits and sandbags
    
    Args:
        my_map: The grid map
        pits: List of pit positions [(row, col), ...]
        sandbags: List of dictionaries with sandbag info [{'pos': (row, col), 'cost': float, 'agent': int}, ...]
    
    Returns:
        EnvironmentManager instance
    """
    env_manager = EnvironmentManager(my_map)
    
    if pits:
        for pit_pos in pits:
            env_manager.add_pit(pit_pos)
    
    if sandbags:
        for sandbag_info in sandbags:
            pos = sandbag_info['pos']
            cost = sandbag_info.get('cost', 2.0)
            agent = sandbag_info.get('agent', None)
            env_manager.add_sandbag(pos, cost, agent)
    
    return env_manager

def validate_path(path, my_map, env_manager=None):
    """
    Validate that a path is feasible given the map and environment constraints
    
    Args:
        path: List of positions [(row, col), ...]
        my_map: Grid map
        env_manager: Optional EnvironmentManager
    
    Returns:
        bool: True if path is valid, False otherwise
    """
    if not path:
        return False
    
    for i, pos in enumerate(path):
        # Check bounds
        if pos[0] < 0 or pos[0] >= len(my_map) or pos[1] < 0 or pos[1] >= len(my_map[0]):
            return False
        
        # Check obstacles
        if my_map[pos[0]][pos[1]]:
            # If environment manager exists, check if it's a filled pit
            if env_manager:
                pit = env_manager.get_pit_at(pos)
                if not (pit and pit.is_filled):
                    return False
            else:
                return False
        
        # Check movement constraints (no diagonal moves)
        if i > 0:
            prev_pos = path[i-1]
            distance = abs(pos[0] - prev_pos[0]) + abs(pos[1] - prev_pos[1])
            if distance > 1:
                return False
    
    return True
