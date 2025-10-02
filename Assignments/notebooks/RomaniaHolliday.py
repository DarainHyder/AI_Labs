import heapq 
from typing import List, Tuple, Dict, Optional, Set 
import math 
 
class RomaniaProblem: 
    """ 
    Romania problem representation with cities and distances. 
    Based on the classic AI problem of finding paths between Romanian cities. 
    """ 
     
    def __init__(self): 
        # Graph representation: city -> {neighbor: cost} 
        self.graph = { 
            'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118}, 
            'Zerind': {'Arad': 75, 'Oradea': 71}, 
            'Oradea': {'Zerind': 71, 'Sibiu': 151}, 
            'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80}, 
            'Timisoara': {'Arad': 118, 'Lugoj': 111}, 
            'Lugoj': {'Timisoara': 111, 'Mehadia': 70}, 
            'Mehadia': {'Lugoj': 70, 'Drobeta': 75}, 
            'Drobeta': {'Mehadia': 75, 'Craiova': 120}, 
            'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138}, 
            'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97}, 
            'Fagaras': {'Sibiu': 99, 'Bucharest': 211}, 
            'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101}, 
            'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85}, 
            'Giurgiu': {'Bucharest': 90}, 
            'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98}, 
            'Hirsova': {'Urziceni': 98, 'Eforie': 86}, 
            'Eforie': {'Hirsova': 86}, 
            'Vaslui': {'Urziceni': 142, 'Iasi': 92}, 
            'Iasi': {'Vaslui': 92, 'Neamt': 87}, 
            'Neamt': {'Iasi': 87} 
        } 
         
        # Straight-line distances to Bucharest (for heuristic) 
        self.straight_line_distances = { 
            'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242, 
            'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151, 
            'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234, 
            'Oradea': 380, 'Pitesti': 100, 'Rimnicu': 193, 'Sibiu': 253, 
            'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374 
        } 
     
    def get_neighbors(self, city: str) -> Dict[str, int]: 
        """Get all neighboring cities and their distances.""" 
        return self.graph.get(city, {}) 
     
    def get_heuristic(self, city: str) -> int: 
        """Get straight-line distance heuristic to Bucharest.""" 
        return self.straight_line_distances.get(city, float('inf')) 
 
class SearchNode: 
    """Node for search algorithms tracking path and cost.""" 
     
    def __init__(self, state: str, parent=None, cost: float = 0, heuristic: float = 0): 
        self.state = state 
        self.parent = parent 
        self.cost = cost  # g(n) - cost from start to current node 
        self.heuristic = heuristic  # h(n) - heuristic cost to goal 
         
    def __lt__(self, other): 
        # For priority queue comparison 
        return (self.cost + self.heuristic) < (other.cost + other.heuristic) 
     
    def get_path(self) -> List[str]: 
        """Reconstruct path from start to current node.""" 
        path = [] 
        current = self 
        while current: 
            path.append(current.state) 
            current = current.parent 
        return path[::-1] 
 
class RomaniaSolver: 
    """Solver for Romania pathfinding problem using different search algorithms.""" 
     
    def __init__(self, problem: RomaniaProblem): 
        self.problem = problem 
     
    def uniform_cost_search(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int]: 
        """Uniform Cost Search implementation.""" 
        frontier = [] 
        heapq.heappush(frontier, (0, SearchNode(start, cost=0))) 
        explored = set() 
        cost_so_far = {start: 0} 
        nodes_expanded = 0 
         
        while frontier: 
            current_cost, current_node = heapq.heappop(frontier) 
             
            if current_node.state in explored: 
                continue 
                 
            if current_node.state == goal: 
                return current_node.get_path(), current_cost, nodes_expanded 
             
            explored.add(current_node.state) 
            nodes_expanded += 1 
             
            for neighbor, step_cost in self.problem.get_neighbors(current_node.state).items(): 
                new_cost = current_cost + step_cost 
                 
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: 
                    cost_so_far[neighbor] = new_cost 
                    new_node = SearchNode(neighbor, current_node, new_cost) 
                    heapq.heappush(frontier, (new_cost, new_node)) 
         
        return None, float('inf'), nodes_expanded 
     
    def greedy_best_first_search(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int]: 
        """Greedy Best-First Search implementation.""" 
        frontier = [] 
        start_heuristic = self.problem.get_heuristic(start) 
        heapq.heappush(frontier, (start_heuristic, SearchNode(start, cost=0, heuristic=start_heuristic))) 
        explored = set() 
        nodes_expanded = 0 
         
        while frontier: 
            current_heuristic, current_node = heapq.heappop(frontier) 
             
            if current_node.state in explored: 
                continue 
                 
            if current_node.state == goal: 
                # Calculate actual path cost 
                path = current_node.get_path() 
                actual_cost = self._calculate_path_cost(path) 
                return path, actual_cost, nodes_expanded 
             
            explored.add(current_node.state) 
            nodes_expanded += 1 
             
            for neighbor, step_cost in self.problem.get_neighbors(current_node.state).items(): 
                if neighbor not in explored: 
                    neighbor_heuristic = self.problem.get_heuristic(neighbor) 
                    new_node = SearchNode(neighbor, current_node,  
                                        current_node.cost + step_cost,  
                                        neighbor_heuristic) 
                    heapq.heappush(frontier, (neighbor_heuristic, new_node)) 
         
        return None, float('inf'), nodes_expanded 
     
    def a_star_search(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int]: 
        """A* Search implementation.""" 
        frontier = [] 
        start_heuristic = self.problem.get_heuristic(start) 
        heapq.heappush(frontier, (start_heuristic, SearchNode(start, cost=0, heuristic=start_heuristic))) 
         
        cost_so_far = {start: 0} 
        nodes_expanded = 0 
         
        while frontier: 
            current_priority, current_node = heapq.heappop(frontier) 
             
            if current_node.state == goal: 
                return current_node.get_path(), current_node.cost, nodes_expanded 
             
            nodes_expanded += 1 
             
            for neighbor, step_cost in self.problem.get_neighbors(current_node.state).items(): 
                new_cost = current_node.cost + step_cost 
                 
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: 
                    cost_so_far[neighbor] = new_cost 
                    neighbor_heuristic = self.problem.get_heuristic(neighbor) 
                    priority = new_cost + neighbor_heuristic 
                    new_node = SearchNode(neighbor, current_node, new_cost, neighbor_heuristic) 
                    heapq.heappush(frontier, (priority, new_node)) 
         
        return None, float('inf'), nodes_expanded 
     
    def _calculate_path_cost(self, path: List[str]) -> float: 
        """Calculate total cost of a path.""" 
        if not path or len(path) < 2: 
            return 0 
         
        total_cost = 0 
        for i in range(len(path) - 1): 
            total_cost += self.problem.graph[path[i]][path[i + 1]] 
        return total_cost 
 
# print section
def print_results(algorithm_name: str, path: Optional[List[str]], cost: float, nodes_expanded: int):
    """Print search results in a structured way."""
    print(f"\n{'='*70}")
    print(f"{algorithm_name.upper():^70}")  # Centered title
    print(f"{'='*70}")
    
    if path:
        print(f" Path        : {' -> '.join(path)}")
        print(f" Total Cost  : {cost} km")
        print(f" Path Length : {len(path)} cities")
    else:
        print(" No path found!")
    
    print(f" Nodes Expanded : {nodes_expanded}")
    print(f"{'='*70}")


def main(): 
    """Main function to demonstrate all search algorithms with clearer output.""" 
    problem = RomaniaProblem() 
    solver = RomaniaSolver(problem) 
     
    test_cases = [ 
        ('Arad', 'Bucharest'), 
        ('Timisoara', 'Bucharest'), 
        ('Oradea', 'Bucharest') 
    ] 
     
    for start, goal in test_cases: 
        print(f"\n{'#'*80}") 
        print(f" SEARCHING ROUTE: {start} -> {goal} ") 
        print(f"{'#'*80}") 
         
        # Run algorithms 
        ucs_path, ucs_cost, ucs_nodes = solver.uniform_cost_search(start, goal) 
        greedy_path, greedy_cost, greedy_nodes = solver.greedy_best_first_search(start, goal) 
        astar_path, astar_cost, astar_nodes = solver.a_star_search(start, goal) 
         
        # Print results 
        print_results("Uniform Cost Search", ucs_path, ucs_cost, ucs_nodes) 
        print_results("Greedy Best-First Search", greedy_path, greedy_cost, greedy_nodes) 
        print_results("A* Search", astar_path, astar_cost, astar_nodes) 
         
        # Comparison table 
        print(f"\nCOMPARISON SUMMARY: {start} -> {goal}") 
        print(f"{'-'*70}") 
        print(f"{'Algorithm':<25} {'Cost (km)':<12} {'Nodes Expanded':<18} {'Optimal'}") 
        print(f"{'-'*70}") 
        print(f"{'Uniform Cost Search':<25} {ucs_cost:<12} {ucs_nodes:<18} {'Yes'}") 
        print(f"{'Greedy Best-First':<25} {greedy_cost:<12} {greedy_nodes:<18} {'No'}") 
        print(f"{'A* Search':<25} {astar_cost:<12} {astar_nodes:<18} {'Yes'}") 
        print(f"{'-'*70}") 
 
if __name__ == "__main__": 
    main() 
