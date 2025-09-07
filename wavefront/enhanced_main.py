#!/usr/bin/env python3
"""
Enhanced DCPCBS Main Script with Environment Modifications

This script demonstrates the complete system with:
- Removable obstacles (pits) and movable resources (sandbags)
- Dynamic environment modifications during CBS search
- Enhanced visualization showing all interactions
"""

from enhanced_cbs import CBSSolverWithEnvironment
from environment import Pit, Sandbag, EnvironmentManager
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

max_time = 600  # 10 minutes
max_nodes = 100000  # Maximum nodes to expand

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def __getitem__(self, index):
        return [self.x, self.y][index]
    
    def __repr__(self):
        return f"({self.x}, {self.y})"

class Agent:
    def __init__(self, id, start_pos, goal_pos):
        self.id = id
        self.start = start_pos
        self.goal = goal_pos
    
    def __repr__(self):
        return f"Agent {self.id}: {self.start} -> {self.goal}"

def create_test_environment():
    """Create test environment with pits and sandbags strategically placed"""
    
    # Define pits at strategic locations that could block common paths
    pits = [
        Pit(position=(15, 10)),  # Central bottleneck
        Pit(position=(8, 15)),   # Another potential bottleneck
        Pit(position=(20, 20)),  # Path blocker
        Pit(position=(25, 8)),   # Alternative route blocker
        Pit(position=(12, 25)),  # Northern passage
    ]
    
    # Define sandbags positioned to potentially fill the pits
    sandbags = [
        Sandbag(id=1, position=(14, 12), move_cost=2.5),  # Near central pit
        Sandbag(id=2, position=(6, 16), move_cost=2.0),   # Near second pit  
        Sandbag(id=3, position=(22, 18), move_cost=2.2),  # Near third pit
        Sandbag(id=4, position=(26, 6), move_cost=2.3),   # Near fourth pit
        Sandbag(id=5, position=(10, 27), move_cost=2.1),  # Near fifth pit
        Sandbag(id=6, position=(18, 14), move_cost=2.4),  # Extra sandbag for flexibility
    ]
    
    return pits, sandbags

def create_simple_test_environment():
    """Create a simpler test environment for debugging"""
    
    # Simple test with fewer pits and sandbags
    pits = [
        Pit(position=(15, 16)),  # Blocks direct path for some agents
        Pit(position=(10, 20)),  # Another strategic location
    ]
    
    sandbags = [
        Sandbag(id=1, position=(16, 15), move_cost=2.0),  # Close to first pit
        Sandbag(id=2, position=(9, 19), move_cost=2.0),   # Close to second pit
    ]
    
    return pits, sandbags

class SimpleVisualization:
    """Simple matplotlib-based visualization for the pathfinding results"""
    
    def __init__(self, my_map, paths, starts, goals, pits=None, sandbags=None, modifications=None):
        self.my_map = my_map
        self.paths = paths
        self.starts = starts
        self.goals = goals
        self.pits = pits or []
        self.sandbags = sandbags or []
        self.modifications = modifications or []
        
        # Colors for agents
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
    def create_static_plot(self):
        """Create a static plot showing the final paths"""
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Draw map
        map_array = np.array(self.my_map)
        ax.imshow(map_array, cmap='gray_r', origin='upper', alpha=0.7)
        
        # Draw pits
        for pit in self.pits:
            if pit.filled:
                ax.plot(pit.position[1], pit.position[0], 's', color='brown', markersize=8, label='Filled Pit' if pit == self.pits[0] else "")
            else:
                ax.plot(pit.position[1], pit.position[0], 's', color='black', markersize=8, label='Unfilled Pit' if pit == self.pits[0] else "")
        
        # Draw sandbags
        for sandbag in self.sandbags:
            if sandbag.assigned_pit:
                # Sandbag is in a pit
                ax.plot(sandbag.position[1], sandbag.position[0], 'D', color='brown', markersize=6)
            else:
                ax.plot(sandbag.position[1], sandbag.position[0], 'D', color='yellow', markersize=6, label='Sandbag' if sandbag.id == 1 else "")
        
        # Draw agent paths
        for i, path in enumerate(self.paths):
            if not path:
                continue
                
            color = self.colors[i % len(self.colors)]
            
            # Draw path
            path_x = [pos[1] for pos in path]
            path_y = [pos[0] for pos in path]
            ax.plot(path_x, path_y, '-', color=color, linewidth=2, alpha=0.7, label=f'Agent {i}')
            
            # Draw start
            ax.plot(self.starts[i][1], self.starts[i][0], 'o', color=color, markersize=10)
            
            # Draw goal
            ax.plot(self.goals[i][1], self.goals[i][0], '*', color=color, markersize=15)
        
        ax.set_title('Enhanced DCPCBS Solution with Environment Modifications')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def create_animation(self):
        """Create an animated visualization of the solution"""
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Find maximum path length
        max_length = max(len(path) for path in self.paths if path)
        
        def animate(frame):
            ax.clear()
            
            # Draw map
            map_array = np.array(self.my_map)
            ax.imshow(map_array, cmap='gray_r', origin='upper', alpha=0.7)
            
            # Draw pits
            for pit in self.pits:
                if pit.filled:
                    ax.plot(pit.position[1], pit.position[0], 's', color='brown', markersize=8)
                else:
                    ax.plot(pit.position[1], pit.position[0], 's', color='black', markersize=8)
            
            # Draw sandbags
            for sandbag in self.sandbags:
                ax.plot(sandbag.position[1], sandbag.position[0], 'D', color='yellow', markersize=6)
            
            # Draw agent positions at current frame
            for i, path in enumerate(self.paths):
                if not path:
                    continue
                    
                color = self.colors[i % len(self.colors)]
                
                # Get current position
                if frame < len(path):
                    pos = path[frame]
                else:
                    pos = path[-1]  # Stay at goal
                
                # Draw agent
                ax.plot(pos[1], pos[0], 'o', color=color, markersize=12, label=f'Agent {i}')
                
                # Draw goal
                ax.plot(self.goals[i][1], self.goals[i][0], '*', color=color, markersize=15, alpha=0.5)
                
                # Draw path history
                if frame > 0:
                    history_x = [path[min(t, len(path)-1)][1] for t in range(min(frame, len(path)))]
                    history_y = [path[min(t, len(path)-1)][0] for t in range(min(frame, len(path)))]
                    ax.plot(history_x, history_y, '-', color=color, linewidth=1, alpha=0.3)
            
            ax.set_title(f'Enhanced DCPCBS Animation - Step {frame}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(-1, len(self.my_map[0]))
            ax.set_ylim(-1, len(self.my_map))
        
        anim = animation.FuncAnimation(fig, animate, frames=max_length+10, interval=500, repeat=True)
        return anim

def main():
    print("Enhanced DCPCBS with Environment Modifications")
    print("=" * 60)
    
    # Define the custom maze from the provided map
    my_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # Define agents with start and goal positions from the provided data
    agents = [
        Agent(1, Position(17, 21), Position(15, 16)),
        Agent(2, Position(23, 23), Position(10, 19)),
        Agent(3, Position(1, 20), Position(8, 2)),
        Agent(4, Position(2, 29), Position(31, 22)),
        Agent(5, Position(28, 26), Position(25, 16)),
        Agent(6, Position(13, 5), Position(5, 1)),
        Agent(7, Position(13, 9), Position(14, 19)),
        Agent(8, Position(1, 29), Position(14, 8)),
        Agent(9, Position(14, 24), Position(19, 4)),
        Agent(10, Position(17, 19), Position(25, 14)),
    ]
    
    print(f"Number of agents: {len(agents)}")
    print("Agent configurations:")
    for agent in agents:
        print(f"  {agent}")
    
    # Extract starts and goals for the solver
    starts = [list(agent.start) for agent in agents]
    goals = [list(agent.goal) for agent in agents]
    
    # Create environment with pits and sandbags
    print("\nSetting up environment...")
    use_simple = input("Use simple environment? (y/n, default=n): ").lower().strip() == 'y'
    
    if use_simple:
        pits, sandbags = create_simple_test_environment()
        print("Using simple test environment")
    else:
        pits, sandbags = create_test_environment()
        print("Using complex test environment")
    
    print(f"Pits: {len(pits)} at positions {[pit.position for pit in pits]}")
    print(f"Sandbags: {len(sandbags)} at positions {[sb.position for sb in sandbags]}")
    
    # Create and run the enhanced CBS solver
    print(f"\nRunning Enhanced CBS solver (max time: {max_time}s, max nodes: {max_nodes})...")
    print("This may take a while for complex problems...\n")
    
    try:
        solver = CBSSolverWithEnvironment(my_map, starts, goals, max_time, max_nodes, pits, sandbags)
        
        start_time = time.time()
        result = solver.find_solution(disjoint=False, batch_size=4)
        solve_time = time.time() - start_time
        
        if result is None:
            print("No solution found within the time/node limits!")
            print(f"Search terminated after {solve_time:.2f} seconds")
            return
        
        paths, num_generated, num_expanded, modifications = result
        
        print(f"\nSolution found in {solve_time:.2f} seconds!")
        print(f"Generated nodes: {num_generated}")
        print(f"Expanded nodes: {num_expanded}")
        
        if modifications:
            print(f"\nEnvironment modifications applied: {len(modifications)}")
            for i, mod in enumerate(modifications):
                pit_pos, sandbag_id, agent_id, benefit = mod
                print(f"  {i+1}: Agent {agent_id} fills pit at {pit_pos} with sandbag {sandbag_id} (benefit: {benefit:.2f})")
        else:
            print("\nNo environment modifications were needed")
        
        # Calculate final costs
        total_cost = sum(len(path) - 1 for path in paths if path)
        print(f"\nTotal solution cost: {total_cost}")
        print(f"Average cost per agent: {total_cost / len(agents):.2f}")
        
        print("\nIndividual agent paths:")
        for i, path in enumerate(paths):
            if path:
                print(f"  Agent {i}: length {len(path)} - {path[0]} -> {path[-1]}")
            else:
                print(f"  Agent {i}: No path found")
        
        # Visualization
        print("\nGenerating visualization...")
        try:
            # Update environment state if modifications were applied
            env_manager = solver.env_manager
            for mod in modifications:
                pit_pos, sandbag_id, agent_id, benefit = mod
                env_manager.apply_pit_modification(pit_pos, sandbag_id, agent_id)
            
            viz = SimpleVisualization(my_map, paths, starts, goals, 
                                    env_manager.pits.values(), 
                                    env_manager.sandbags.values(), 
                                    modifications)
            
            # Create static plot
            fig = viz.create_static_plot()
            plt.show()
            
            # Ask if user wants animation
            show_anim = input("\nShow animation? (y/n, default=n): ").lower().strip() == 'y'
            if show_anim:
                print("Generating animation... (close window to continue)")
                anim = viz.create_animation()
                plt.show()
                
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Solution data:")
            for i, path in enumerate(paths):
                print(f"Agent {i}: {path}")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
    except Exception as e:
        print(f"\nError during solving: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
