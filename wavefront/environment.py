#!/usr/bin/env python3
"""
Environment Management Module for DCPCBS with Removable Obstacles and Movable Resources

This module handles:
- Pits (removable obstacles that can be filled with sandbags)
- Sandbags (movable resources that can fill pits)
- Dynamic grid updates and state management
- Agent-sandbag-pit assignment using wavefront propagation and cost-benefit analysis
"""

import numpy as np
import heapq
from collections import defaultdict, deque
import copy
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.spatial import KDTree

@dataclass
class Pit:
    """Represents a removable obstacle (pit) in the environment"""
    position: Tuple[int, int]
    filled: bool = False
    sandbag_id: Optional[int] = None  # ID of sandbag filling this pit
    
    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        return isinstance(other, Pit) and self.position == other.position

@dataclass 
class Sandbag:
    """Represents a movable resource (sandbag) in the environment"""
    id: int
    position: Tuple[int, int]
    assigned_pit: Optional[Tuple[int, int]] = None
    assigned_agent: Optional[int] = None
    move_cost: float = 2.0  # Cost multiplier for moving sandbags
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Sandbag) and self.id == other.id

class EnvironmentManager:
    """Manages dynamic environment modifications for DCPCBS"""
    
    def __init__(self, my_map, pits=None, sandbags=None):
        """
        Initialize environment manager
        
        Args:
            my_map: 2D grid (1=obstacle, 0=free space)
            pits: List of Pit objects
            sandbags: List of Sandbag objects
        """
        self.original_map = copy.deepcopy(my_map)
        self.current_map = copy.deepcopy(my_map)
        
        # Initialize pits and sandbags
        self.pits = {pit.position: pit for pit in (pits or [])}
        self.sandbags = {sb.id: sb for sb in (sandbags or [])}
        
        # Track modifications for rollback capability
        self.modification_history = []
        self.state_id = 0
        
        # Wavefront cache for efficiency
        self._wavefront_cache = {}
        
        # Initialize KDTree for nearest neighbor queries
        self._update_kdtree()
    
    def _update_kdtree(self):
        """Update KDTree for efficient nearest sandbag queries"""
        if self.sandbags:
            positions = [sb.position for sb in self.sandbags.values() 
                        if sb.assigned_pit is None]  # Only unassigned sandbags
            if positions:
                self.sandbag_kdtree = KDTree(positions)
                self.available_sandbag_positions = positions
            else:
                self.sandbag_kdtree = None
                self.available_sandbag_positions = []
        else:
            self.sandbag_kdtree = None
            self.available_sandbag_positions = []
    
    def get_current_map(self):
        """Get current state of the map"""
        return copy.deepcopy(self.current_map)
    
    def get_state_id(self):
        """Get unique identifier for current environment state"""
        return self.state_id
    
    def is_pit_at(self, position):
        """Check if there's an unfilled pit at the given position"""
        return position in self.pits and not self.pits[position].filled
    
    def is_sandbag_at(self, position):
        """Check if there's a sandbag at the given position"""
        for sandbag in self.sandbags.values():
            if sandbag.position == position:
                return True
        return False
    
    def get_sandbag_at(self, position):
        """Get sandbag at the given position"""
        for sandbag in self.sandbags.values():
            if sandbag.position == position:
                return sandbag
        return None
    
    def compute_wavefront_from_goal(self, goal, max_distance=50):
        """
        Compute wavefront distances from goal using Dijkstra-like propagation
        
        Returns:
            dict: position -> distance mapping
        """
        if goal in self._wavefront_cache:
            return self._wavefront_cache[goal]
        
        distances = {}
        queue = [(0, goal)]
        visited = set()
        
        while queue:
            dist, pos = heapq.heappop(queue)
            
            if pos in visited or dist > max_distance:
                continue
                
            visited.add(pos)
            distances[pos] = dist
            
            # Check all 4 directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # Check bounds
                if (new_pos[0] < 0 or new_pos[0] >= len(self.current_map) or
                    new_pos[1] < 0 or new_pos[1] >= len(self.current_map[0])):
                    continue
                
                # Skip obstacles (but allow pits for potential filling)
                if (self.current_map[new_pos[0]][new_pos[1]] and 
                    not self.is_pit_at(new_pos)):
                    continue
                
                if new_pos not in visited:
                    heapq.heappush(queue, (dist + 1, new_pos))
        
        self._wavefront_cache[goal] = distances
        return distances
    
    def compute_pit_filling_benefit(self, pit_pos, agent_goals, agent_positions):
        """
        Compute the benefit of filling a specific pit
        
        Args:
            pit_pos: Position of the pit to evaluate
            agent_goals: List of agent goal positions
            agent_positions: Current agent positions
            
        Returns:
            float: Benefit score (higher = more beneficial)
        """
        if not self.is_pit_at(pit_pos):
            return 0.0
        
        benefit = 0.0
        
        # Calculate benefit based on agents that could use this path
        for i, (agent_pos, agent_goal) in enumerate(zip(agent_positions, agent_goals)):
            # Get wavefront distances from agent's goal
            goal_distances = self.compute_wavefront_from_goal(agent_goal)
            
            if pit_pos in goal_distances:
                # Distance from agent current position to pit
                agent_to_pit = abs(agent_pos[0] - pit_pos[0]) + abs(agent_pos[1] - pit_pos[1])
                
                # Distance from pit to goal
                pit_to_goal = goal_distances[pit_pos]
                
                # Estimate total distance if using this pit
                total_with_pit = agent_to_pit + pit_to_goal
                
                # Estimate detour distance (approximate)
                direct_distance = abs(agent_pos[0] - agent_goal[0]) + abs(agent_pos[1] - agent_goal[1])
                
                # Benefit is the detour savings minus filling cost
                detour_savings = max(0, direct_distance - total_with_pit)
                benefit += detour_savings
        
        return benefit
    
    def find_best_sandbag_for_pit(self, pit_pos, agent_positions, agent_goals):
        """
        Find the best sandbag and agent combination to fill a specific pit
        
        Returns:
            tuple: (sandbag_id, agent_id, total_cost) or (None, None, float('inf'))
        """
        if not self.sandbag_kdtree or not self.is_pit_at(pit_pos):
            return None, None, float('inf')
        
        best_cost = float('inf')
        best_sandbag = None
        best_agent = None
        
        # Find nearest sandbags using KDTree
        try:
            distances, indices = self.sandbag_kdtree.query([pit_pos], k=min(3, len(self.available_sandbag_positions)))
            
            if not hasattr(distances, '__iter__'):
                distances = [distances]
                indices = [indices]
        except:
            return None, None, float('inf')
        
        for dist, idx in zip(distances, indices):
            sandbag_pos = self.available_sandbag_positions[idx]
            
            # Find the sandbag object
            sandbag = None
            for sb in self.sandbags.values():
                if sb.position == sandbag_pos and sb.assigned_pit is None:
                    sandbag = sb
                    break
            
            if not sandbag:
                continue
            
            # Find best agent to move this sandbag
            for agent_id, agent_pos in enumerate(agent_positions):
                # Cost for agent to reach sandbag
                agent_to_sandbag = abs(agent_pos[0] - sandbag.position[0]) + abs(agent_pos[1] - sandbag.position[1])
                
                # Cost to push sandbag to pit
                sandbag_to_pit = abs(sandbag.position[0] - pit_pos[0]) + abs(sandbag.position[1] - pit_pos[1])
                
                # Total cost including sandbag movement penalty
                total_cost = agent_to_sandbag + (sandbag_to_pit * sandbag.move_cost)
                
                # Bias toward agents whose direction aligns with the task
                agent_goal = agent_goals[agent_id]
                goal_direction = (agent_goal[0] - agent_pos[0], agent_goal[1] - agent_pos[1])
                sandbag_direction = (sandbag.position[0] - agent_pos[0], sandbag.position[1] - agent_pos[1])
                
                # Dot product for direction alignment (normalized)
                if abs(goal_direction[0]) + abs(goal_direction[1]) > 0:
                    alignment = (goal_direction[0] * sandbag_direction[0] + goal_direction[1] * sandbag_direction[1]) / (
                        (abs(goal_direction[0]) + abs(goal_direction[1])) * max(1, abs(sandbag_direction[0]) + abs(sandbag_direction[1]))
                    )
                    # Reduce cost for better alignment
                    total_cost *= (1.0 - 0.3 * max(0, alignment))
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_sandbag = sandbag.id
                    best_agent = agent_id
        
        return best_sandbag, best_agent, best_cost
    
    def evaluate_pit_modifications(self, agent_positions, agent_goals, max_evaluations=5):
        """
        Evaluate which pits should be filled based on cost-benefit analysis
        
        Returns:
            list: List of (pit_pos, sandbag_id, agent_id, net_benefit) sorted by benefit
        """
        modifications = []
        
        # Get unfilled pits sorted by potential benefit
        unfilled_pits = [(pos, pit) for pos, pit in self.pits.items() if not pit.filled]
        
        # Limit evaluations for performance
        if len(unfilled_pits) > max_evaluations:
            # Sort by distance to nearest agent to prioritize accessible pits
            def pit_priority(pit_item):
                pos, _ = pit_item
                min_dist = min(abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]) 
                              for agent_pos in agent_positions)
                return min_dist
            
            unfilled_pits.sort(key=pit_priority)
            unfilled_pits = unfilled_pits[:max_evaluations]
        
        for pit_pos, pit in unfilled_pits:
            # Calculate benefit of filling this pit
            benefit = self.compute_pit_filling_benefit(pit_pos, agent_goals, agent_positions)
            
            # Find best sandbag and agent for this pit
            sandbag_id, agent_id, cost = self.find_best_sandbag_for_pit(pit_pos, agent_positions, agent_goals)
            
            if sandbag_id is not None:
                net_benefit = benefit - cost
                modifications.append((pit_pos, sandbag_id, agent_id, net_benefit))
        
        # Sort by net benefit (highest first)
        modifications.sort(key=lambda x: x[3], reverse=True)
        return modifications
    
    def apply_pit_modification(self, pit_pos, sandbag_id, agent_id):
        """
        Apply a pit filling modification to the environment
        
        Returns:
            bool: True if modification was successful
        """
        if pit_pos not in self.pits or self.pits[pit_pos].filled:
            return False
        
        if sandbag_id not in self.sandbags:
            return False
        
        # Record modification for potential rollback
        modification = {
            'type': 'fill_pit',
            'pit_pos': pit_pos,
            'sandbag_id': sandbag_id,
            'agent_id': agent_id,
            'sandbag_old_pos': self.sandbags[sandbag_id].position
        }
        
        # Apply the modification
        self.pits[pit_pos].filled = True
        self.pits[pit_pos].sandbag_id = sandbag_id
        
        # Update sandbag status
        self.sandbags[sandbag_id].position = pit_pos
        self.sandbags[sandbag_id].assigned_pit = pit_pos
        self.sandbags[sandbag_id].assigned_agent = agent_id
        
        # Update map - pit becomes passable
        self.current_map[pit_pos[0]][pit_pos[1]] = 0
        
        # Update state tracking
        self.modification_history.append(modification)
        self.state_id += 1
        
        # Update KDTree
        self._update_kdtree()
        
        # Clear wavefront cache
        self._wavefront_cache.clear()
        
        return True
    
    def rollback_last_modification(self):
        """Rollback the last environmental modification"""
        if not self.modification_history:
            return False
        
        modification = self.modification_history.pop()
        
        if modification['type'] == 'fill_pit':
            pit_pos = modification['pit_pos']
            sandbag_id = modification['sandbag_id']
            
            # Restore pit state
            self.pits[pit_pos].filled = False
            self.pits[pit_pos].sandbag_id = None
            
            # Restore sandbag state
            self.sandbags[sandbag_id].position = modification['sandbag_old_pos']
            self.sandbags[sandbag_id].assigned_pit = None
            self.sandbags[sandbag_id].assigned_agent = None
            
            # Restore map
            self.current_map[pit_pos[0]][pit_pos[1]] = 1
        
        # Update state tracking
        self.state_id += 1
        
        # Update KDTree
        self._update_kdtree()
        
        # Clear wavefront cache
        self._wavefront_cache.clear()
        
        return True
    
    def get_modification_actions(self, pit_pos, sandbag_id, agent_id):
        """
        Generate action sequence for an agent to fill a pit with a sandbag
        
        Returns:
            list: Sequence of actions for the agent
        """
        if (pit_pos not in self.pits or sandbag_id not in self.sandbags):
            return []
        
        sandbag = self.sandbags[sandbag_id]
        actions = []
        
        # Action to reach sandbag
        actions.append({
            'type': 'move_to_sandbag',
            'agent': agent_id,
            'target': sandbag.position,
            'sandbag_id': sandbag_id
        })
        
        # Action to push sandbag to pit
        actions.append({
            'type': 'push_sandbag',
            'agent': agent_id,
            'sandbag_id': sandbag_id,
            'from': sandbag.position,
            'to': pit_pos
        })
        
        return actions
    
    def clone_state(self):
        """Create a deep copy of current environment state"""
        new_env = EnvironmentManager(self.original_map)
        new_env.current_map = copy.deepcopy(self.current_map)
        new_env.pits = copy.deepcopy(self.pits)
        new_env.sandbags = copy.deepcopy(self.sandbags)
        new_env.modification_history = copy.deepcopy(self.modification_history)
        new_env.state_id = self.state_id
        new_env._update_kdtree()
        return new_env
