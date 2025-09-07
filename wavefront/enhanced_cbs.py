#!/usr/bin/env python3
"""
Enhanced DCPCBS Solver with Environment Modification Support

This module extends the original CBS implementation to handle:
- Removable obstacles (pits) and movable resources (sandbags) 
- Dynamic environment modifications during search
- Cost-benefit analysis for obstacle removal vs. detour
- Integration with environment management module
"""

import heapq
import random
import multiprocessing
from functools import partial
import copy
import time as timer
from enhanced_astar import A_Star, compute_heuristics, get_location
from environment import EnvironmentManager, Pit, Sandbag

# Utility functions from original cbs.py
def get_sum_of_cost(paths):
    """Calculate the sum of costs for all paths."""
    rst = 0
    for path in paths:
        rst += len(path) - 1
        if len(path) > 1:
            assert path[-1] != path[-2]
    return rst

def detect_collision(path1, path2):
    """Detect if there is a collision between two paths."""
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
    """Compute the criticality score for a collision."""
    a1 = collision['a1']
    a2 = collision['a2']
    timestep = collision['timestep']
    loc = collision['loc'][0] if len(collision['loc']) == 1 else collision['loc'][0]
    loc = tuple(loc)
    
    # Compute Manhattan distance to goals
    dist_i = abs(loc[0] - goals[a1][0]) + abs(loc[1] - goals[a1][1])
    dist_j = abs(loc[0] - goals[a2][0]) + abs(loc[1] - goals[a2][1])
    total_dist = dist_i + dist_j
    
    if total_dist == 0:
        dist_score = w2 * 1.0
    else:
        dist_score = w2 * (1.0 / total_dist)
    
    # Conflict frequency score
    freq_key = tuple(sorted([a1, a2]))
    freq_score = w1 * (1 if conflict_freq.get(freq_key, 0) >= T_freq else 0)
    
    # Timestep score
    time_score = w3 * (horizon - timestep)
    
    return freq_score + dist_score + time_score

def detect_collisions(paths, goals, conflict_freq, k=5, horizon=100):
    """Detect up to k earliest collisions and select the one with highest criticality score."""
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
    
    # Sort collisions by timestep to get earliest k
    collisions.sort(key=lambda x: x['timestep'])
    collisions = collisions[:min(k, len(collisions))]
    
    # Compute criticality scores and select the highest
    scored_collisions = []
    for collision in collisions:
        score = compute_criticality_score(collision, goals, conflict_freq, horizon=horizon)
        scored_collisions.append((score, collision))
    
    # Select collision with maximum score
    selected_collision = max(scored_collisions, key=lambda x: x[0])[1]
    
    # Update conflict frequency
    freq_key = tuple(sorted([selected_collision['a1'], selected_collision['a2']]))
    conflict_freq[freq_key] = conflict_freq.get(freq_key, 0) + 1
    
    return [selected_collision]

def standard_splitting(collision):
    """Generate constraints for standard splitting."""
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

def disjoint_splitting(collision):
    """Generate constraints for disjoint splitting."""
    constraints = []
    agent = random.randint(0, 1)
    a = 'a' + str(agent + 1)
    if len(collision['loc']) == 1:
        constraints.append({'agent': collision[a],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': True})
        constraints.append({'agent': collision[a],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
    else:
        if agent == 0:
            constraints.append({'agent': collision[a],
                                'loc': [collision['loc'][0], collision['loc'][1]],
                                'timestep': collision['timestep'],
                                'positive': True})
            constraints.append({'agent': collision[a],
                                'loc': [collision['loc'][0], collision['loc'][1]],
                                'timestep': collision['timestep'],
                                'positive': False})
        else:
            constraints.append({'agent': collision[a],
                                'loc': [collision['loc'][1], collision['loc'][0]],
                                'timestep': collision['timestep'],
                                'positive': True})
            constraints.append({'agent': collision[a],
                                'loc': [collision['loc'][1], collision['loc'][0]],
                                'timestep': collision['timestep'],
                                'positive': False})
    return constraints

def paths_violate_constraint(constraint, paths):
    """Check which paths violate a positive constraint."""
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

def detect_pit_blockages(paths, env_manager):
    """
    Detect if any agent paths are blocked by unfilled pits
    
    Returns:
        list: List of (agent_id, pit_position, timestep) tuples
    """
    blockages = []
    
    for agent_id, path in enumerate(paths):
        for t, position in enumerate(path):
            pos_tuple = tuple(position) if isinstance(position, list) else position
            if env_manager.is_pit_at(pos_tuple):
                blockages.append((agent_id, pos_tuple, t))
    
    return blockages

def evaluate_environment_modifications(paths, env_manager, goals):
    """
    Evaluate whether environment modifications would improve solution cost
    
    Returns:
        tuple: (should_modify, modifications_list)
    """
    # Get current agent positions (start positions)
    agent_positions = [path[0] if path else None for path in paths]
    agent_positions = [pos for pos in agent_positions if pos is not None]
    
    if len(agent_positions) != len(goals):
        return False, []
    
    # Check for pit blockages
    blockages = detect_pit_blockages(paths, env_manager)
    
    if not blockages:
        # No blockages, check if modifications could improve overall cost
        current_cost = get_sum_of_cost(paths)
        
        # Evaluate potential modifications
        modifications = env_manager.evaluate_pit_modifications(
            agent_positions, goals, max_evaluations=3
        )
        
        if modifications:
            best_modification = modifications[0]  # Already sorted by benefit
            if best_modification[3] > current_cost * 0.1:  # 10% improvement threshold
                return True, [best_modification]
    else:
        # Direct blockages - must modify to proceed
        # Find modifications for blocked pits
        blocked_pits = set(blockage[1] for blockage in blockages)
        modifications = []
        
        for pit_pos in blocked_pits:
            sandbag_id, agent_id, cost = env_manager.find_best_sandbag_for_pit(
                pit_pos, agent_positions, goals
            )
            if sandbag_id is not None:
                modifications.append((pit_pos, sandbag_id, agent_id, -cost))  # Negative cost = required
        
        if modifications:
            return True, modifications
    
    return False, []

# Modified process_node function for parallel processing with environment management
def process_node_with_env(args, my_map, starts, goals, heuristics, splitter, k=5, horizon=100, env_manager=None):
    """
    Process a CT node with environment modification support
    """
    node, conflict_freq = args
    
    # Clone environment manager for this node
    if env_manager:
        local_env = env_manager.clone_state()
        current_map = local_env.get_current_map()
    else:
        local_env = None
        current_map = my_map
    
    # Check for solution
    if not node['collisions']:
        # Before declaring solution, check if environment modifications could improve it
        if local_env:
            should_modify, modifications = evaluate_environment_modifications(
                node['paths'], local_env, goals
            )
            
            if should_modify and modifications:
                # Apply best modification and replan
                best_mod = modifications[0]
                pit_pos, sandbag_id, agent_id, benefit = best_mod
                
                if local_env.apply_pit_modification(pit_pos, sandbag_id, agent_id):
                    # Replan all agents with updated map
                    updated_map = local_env.get_current_map()
                    updated_paths = []
                    
                    for i in range(len(node['paths'])):
                        astar = A_Star(updated_map, starts, goals, heuristics, i, node['constraints'])
                        path = astar.find_paths()
                        if path is None:
                            break
                        updated_paths.append(path[0])
                    else:
                        # All paths found successfully
                        updated_node = copy.deepcopy(node)
                        updated_node['paths'] = updated_paths
                        updated_node['cost'] = get_sum_of_cost(updated_paths)
                        updated_node['collisions'] = detect_collisions(
                            updated_paths, goals, updated_node['conflict_freq'], k, horizon
                        )
                        updated_node['env_manager'] = local_env
                        updated_node['modifications'] = node.get('modifications', []) + [best_mod]
                        
                        if not updated_node['collisions']:
                            return True, updated_node  # Better solution found
                        else:
                            return False, [updated_node]  # Continue search with modified environment
        
        return True, node  # Original solution
    
    collision = node['collisions'][0]
    constraints = splitter(collision)
    children = []
    
    for constraint in constraints:
        child = {
            'cost': 0,
            'constraints': node['constraints'] + [constraint],
            'paths': node['paths'].copy(),
            'conflict_freq': dict(conflict_freq),
            'collisions': [],
            'env_manager': local_env,
            'modifications': node.get('modifications', [])
        }
        ai = constraint['agent']
        astar = A_Star(current_map, starts, goals, heuristics, ai, child['constraints'])
        path = astar.find_paths()
        
        if path is not None:
            child['paths'][ai] = path[0]
            if constraint.get('positive', False):
                vol = paths_violate_constraint(constraint, child['paths'])
                for v in vol:
                    astar_v = A_Star(current_map, starts, goals, heuristics, v, child['constraints'])
                    path_v = astar_v.find_paths()
                    if path_v is None:
                        break
                    child['paths'][v] = path_v[0]
                else:
                    child['collisions'] = detect_collisions(child['paths'], goals, child['conflict_freq'], k, horizon)
                    child['cost'] = get_sum_of_cost(child['paths'])
                    
                    # Check if environment modifications could help
                    if local_env:
                        should_modify, modifications = evaluate_environment_modifications(
                            child['paths'], local_env, goals
                        )
                        
                        if should_modify and modifications:
                            # Try applying the best modification
                            best_mod = modifications[0]
                            pit_pos, sandbag_id, agent_id, benefit = best_mod
                            
                            if local_env.apply_pit_modification(pit_pos, sandbag_id, agent_id):
                                # Replan affected agents
                                updated_map = local_env.get_current_map()
                                updated_paths = []
                                
                                for i in range(len(child['paths'])):
                                    astar_updated = A_Star(updated_map, starts, goals, heuristics, i, child['constraints'])
                                    path_updated = astar_updated.find_paths()
                                    if path_updated is None:
                                        local_env.rollback_last_modification()
                                        break
                                    updated_paths.append(path_updated[0])
                                else:
                                    # All paths found successfully with modification
                                    child['paths'] = updated_paths
                                    child['cost'] = get_sum_of_cost(updated_paths)
                                    child['collisions'] = detect_collisions(
                                        updated_paths, goals, child['conflict_freq'], k, horizon
                                    )
                                    child['modifications'].append(best_mod)
                    
                    children.append(child)
                    continue
            
            child['collisions'] = detect_collisions(child['paths'], goals, child['conflict_freq'], k, horizon)
            child['cost'] = get_sum_of_cost(child['paths'])
            
            # Check if environment modifications could help this child
            if local_env:
                should_modify, modifications = evaluate_environment_modifications(
                    child['paths'], local_env, goals
                )
                
                if should_modify and modifications:
                    # Try applying the best modification
                    best_mod = modifications[0]
                    pit_pos, sandbag_id, agent_id, benefit = best_mod
                    
                    if local_env.apply_pit_modification(pit_pos, sandbag_id, agent_id):
                        # Replan all agents with updated environment
                        updated_map = local_env.get_current_map()
                        updated_paths = []
                        
                        for i in range(len(child['paths'])):
                            astar_updated = A_Star(updated_map, starts, goals, heuristics, i, child['constraints'])
                            path_updated = astar_updated.find_paths()
                            if path_updated is None:
                                local_env.rollback_last_modification()
                                break
                            updated_paths.append(path_updated[0])
                        else:
                            # All paths found successfully with modification
                            child['paths'] = updated_paths
                            child['cost'] = get_sum_of_cost(updated_paths)
                            child['collisions'] = detect_collisions(
                                updated_paths, goals, child['conflict_freq'], k, horizon
                            )
                            child['modifications'].append(best_mod)
            
            children.append(child)
    
    return False, children


# Enhanced CBS Solver Class
class CBSSolverWithEnvironment:
    def __init__(self, my_map, starts, goals, max_time=300, max_nodes=50000, 
                 pits=None, sandbags=None):
        """
        Initialize the enhanced CBS solver with environment management
        """
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.open_list = []
        self.conflict_freq = {}
        
        # Environment management
        self.env_manager = EnvironmentManager(my_map, pits, sandbags)
        self.my_map = self.env_manager.get_current_map()
        
        # Add timeout and node limits
        self.max_time = max_time
        self.max_nodes = max_nodes
        self.start_time = None

        # Compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(self.my_map, goal))

    def push_node(self, node):
        """Push a node into the OPEN list."""
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        """Pop the best node from the OPEN list."""
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def should_terminate(self):
        """Check if we should terminate the search due to resource limits."""
        if self.start_time is None:
            return False
            
        # Check time limit
        if timer.time() - self.start_time > self.max_time:
            print(f"Terminating: Time limit of {self.max_time}s exceeded")
            return True
        
        # Check node limit
        if self.num_of_expanded > self.max_nodes:
            print(f"Terminating: Node limit of {self.max_nodes} exceeded")
            return True
        
        return False

    def get_adaptive_batch_size(self, base_batch_size):
        """Calculate adaptive batch size based on problem complexity."""
        batch_size = base_batch_size
        
        # Reduce batch size for complex problems
        agent_factor = max(1, self.num_of_agents // 5)
        depth_factor = max(1, self.num_of_expanded // 1000)
        
        batch_size = max(1, batch_size // (agent_factor * depth_factor))
        
        # Memory-based adjustment
        if len(self.open_list) > 10000:
            batch_size = max(1, batch_size // 2)
        
        return min(batch_size, len(self.open_list))

    def find_solution(self, disjoint=False, batch_size=None, k=5, horizon=100):
        """
        Find paths for all agents using enhanced CBS with environment modifications
        """
        self.start_time = timer.time()

        if disjoint:
            splitter = disjoint_splitting
        else:
            splitter = standard_splitting

        print("USING:", splitter.__name__)
        print("Environment features enabled: Pits and Sandbags")

        # Set batch size to CPU count if not specified
        if batch_size is None:
            batch_size = multiprocessing.cpu_count()
        
        print(f"Using parallel CBS with environment modifications (base batch: {batch_size})")

        # Generate the root node
        root = {
            'cost': 0,
            'constraints': [],
            'paths': [],
            'collisions': [],
            'conflict_freq': {},
            'env_manager': self.env_manager,
            'modifications': []
        }
        
        current_map = self.env_manager.get_current_map()
        for i in range(self.num_of_agents):
            astar = A_Star(current_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])
        
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'], self.goals, root['conflict_freq'], k, horizon)
        
        # Check if initial environment modifications would help
        should_modify, modifications = evaluate_environment_modifications(
            root['paths'], self.env_manager, self.goals
        )
        
        if should_modify and modifications:
            print(f"Evaluating {len(modifications)} potential environment modifications...")
            best_mod = modifications[0]
            pit_pos, sandbag_id, agent_id, benefit = best_mod
            
            if self.env_manager.apply_pit_modification(pit_pos, sandbag_id, agent_id):
                print(f"Applied modification: Agent {agent_id} fills pit at {pit_pos} with sandbag {sandbag_id}")
                
                # Replan all agents with updated environment
                updated_map = self.env_manager.get_current_map()
                for i in range(self.num_of_agents):
                    astar = A_Star(updated_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
                    path = astar.find_paths()
                    if path is None:
                        # Rollback if planning fails
                        self.env_manager.rollback_last_modification()
                        break
                    root['paths'][i] = path[0]
                else:
                    # All replanning successful
                    root['cost'] = get_sum_of_cost(root['paths'])
                    root['collisions'] = detect_collisions(root['paths'], self.goals, root['conflict_freq'], k, horizon)
                    root['modifications'].append(best_mod)
                    print(f"Environment modification improved solution cost to {root['cost']}")
        
        self.push_node(root)

        # Track best solution found so far
        best_solution = None
        best_cost = float('inf')

        # Process nodes with environment management
        with multiprocessing.Pool(processes=batch_size) as pool:
            while self.open_list:
                # Check termination conditions
                if self.should_terminate():
                    break
                
                # Get adaptive batch size
                current_batch_size = self.get_adaptive_batch_size(batch_size)
                
                # Collect nodes for processing
                nodes = []
                for _ in range(current_batch_size):
                    if self.open_list:
                        node = self.pop_node()
                        # Check if this is a solution
                        if not node['collisions']:
                            print("Found solution!")
                            self.print_results(node)
                            return node['paths'], self.num_of_generated, self.num_of_expanded, node.get('modifications', [])
                        nodes.append(node)
                    else:
                        break
                
                if not nodes:
                    break

                # Update best solution found so far
                for node in nodes:
                    if node['cost'] < best_cost:
                        best_cost = node['cost']
                        if not node['collisions']:
                            best_solution = node

                print(f"Expanding {len(nodes)} nodes (batch size: {current_batch_size}, "
                    f"open list: {len(self.open_list)}, expanded: {self.num_of_expanded})")

                # Create partial function with environment manager
                process_node_partial = partial(
                    process_node_with_env,
                    my_map=self.my_map,
                    starts=self.starts,
                    goals=self.goals,
                    heuristics=self.heuristics,
                    splitter=splitter,
                    k=k,
                    horizon=horizon,
                    env_manager=self.env_manager
                )

                # Process nodes in parallel
                try:
                    results = pool.map(process_node_partial, [(node, node['conflict_freq']) for node in nodes])
                    
                    for is_solution, data in results:
                        if is_solution:
                            print("Found solution!")
                            self.print_results(data)
                            return data['paths'], self.num_of_generated, self.num_of_expanded, data.get('modifications', [])
                        for child in data:
                            self.push_node(child)
                            
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                    continue

        # If we exit the loop without finding optimal solution
        if best_solution is not None:
            print("Returning best solution found within limits:")
            self.print_results(best_solution)
            return best_solution['paths'], self.num_of_generated, self.num_of_expanded, best_solution.get('modifications', [])
        
        print("No solution found within resource limits")
        return None

    def print_results(self, node):
        """Print the results of the solution."""
        print("\nFound a solution!\n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
        
        modifications = node.get('modifications', [])
        if modifications:
            print(f"Environment modifications: {len(modifications)}")
            for i, mod in enumerate(modifications):
                pit_pos, sandbag_id, agent_id, benefit = mod
                print(f"  {i+1}: Agent {agent_id} fills pit at {pit_pos} with sandbag {sandbag_id} (benefit: {benefit:.2f})")
        else:
            print("No environment modifications applied")
            
        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])
