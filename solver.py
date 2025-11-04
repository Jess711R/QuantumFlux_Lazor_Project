"""
Advanced Solver for Lazor Puzzles
=================================
This module contains the AdvancedSolver class, which implements multiple
strategies to solve Lazor puzzles of varying complexity.
"""

import itertools
import time
import math
import random
from typing import List, Tuple, Optional

from lazor_board import LazorBoard, Block
from laser_engine import LaserEngine


class AdvancedSolver:
    """Multi-strategy solver for all puzzle complexities."""
    
    def __init__(self, board: LazorBoard):
        """Initialize the AdvancedSolver with a board.
        
        Args:
            board (LazorBoard): The puzzle board to solve
            
        Returns:
            None
        """
        self.board = board
        self.engine = LaserEngine(board.grid_width, board.grid_height)
        self.solution: Optional[List[Tuple[int, int, str]]] = None
        self.solve_time = 0
        self.strategy_used = ""
    
    def solve(self, timeout: int = 300) -> bool:
        """Solve using adaptive strategy selection.
        
        Args:
            timeout (int): Maximum time to spend solving in seconds (default: 300)
            
        Returns:
            bool: True if a solution was found, False otherwise
        """
        start_time = time.time()
        
        blocks_needed = [bt for bt, count in self.board.available_blocks.items() for _ in range(count)]
        
        if not blocks_needed:
            success = self.engine.simulate_all_lasers(
                self.board.lasers, self.board.fixed_blocks, set(self.board.targets)
            )
            if success:
                self.solution = []
                self.strategy_used = "No placement needed"
            self.solve_time = time.time() - start_time
            return success
        
        positions = self.board.open_positions
        if len(blocks_needed) > len(positions):
            return False
        
        complexity = self.estimate_complexity(len(positions), len(blocks_needed))
        print(f"  Complexity estimate: {complexity:,}")
        
        puzzle_name = self.board.name.lower()
        
        if 'numbered_6' in puzzle_name or 'showstopper_4' in puzzle_name:
            success = self.solve_simulated_annealing(blocks_needed, positions, start_time, timeout)
            if not success and timeout - (time.time() - start_time) > 60:
                print("  Falling back to Genetic Algorithm...")
                success = self.solve_genetic(blocks_needed, positions, time.time(), timeout - (time.time() - start_time))
        elif complexity <= 1_000_000:
            success = self.solve_direct(blocks_needed, positions, start_time, timeout)
            if not success and timeout - (time.time() - start_time) > 30:
                print("  Trying optimized approach as backup...")
                success = self.solve_optimized(blocks_needed, positions, time.time(), timeout - (time.time() - start_time))
        elif complexity <= 2_000_000:
            success = self.solve_optimized(blocks_needed, positions, start_time, timeout * 0.7)
            if not success and timeout - (time.time() - start_time) > 60:
                print("  Trying heuristic approach as final attempt...")
                success = self.solve_heuristic(blocks_needed, positions, time.time(), timeout - (time.time() - start_time))
        else:
            success = self.solve_heuristic(blocks_needed, positions, start_time, timeout * 0.8)
            if not success and timeout - (time.time() - start_time) > 60:
                print("  Final attempt with genetic algorithm...")
                success = self.solve_genetic(blocks_needed, positions, time.time(), timeout - (time.time() - start_time))
        
        self.solve_time = time.time() - start_time
        return success

    def solve_simulated_annealing(self, blocks_needed, positions, start_time, timeout):
        """Solve using Simulated Annealing.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when solving started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        print("  Strategy: Simulated Annealing")
        self.strategy_used = "Simulated Annealing"

        initial_temp, final_temp, alpha, iterations_per_temp = 100.0, 0.1, 0.98, 100

        def get_energy(placement):
            if not placement: return float('inf')
            all_blocks = self.board.fixed_blocks + [Block(bt, x, y) for x, y, bt in placement]
            hit_points = self.engine.get_all_hit_points(self.board.lasers, all_blocks)
            return len(set(self.board.targets) - hit_points)

        def generate_neighbor(placement):
            neighbor = placement[:]
            if not neighbor: return None
            
            possible_moves = []
            if len(positions) > len(blocks_needed): possible_moves.append('move')
            if len(neighbor) > 1: possible_moves.append('swap_pos')
            if len(set(b[2] for b in neighbor)) > 1: possible_moves.append('swap_type')
            if not possible_moves: return neighbor

            move_type = random.choice(possible_moves)
            
            if move_type == 'move':
                idx_to_move = random.randrange(len(neighbor))
                used_positions = {p[0:2] for p in neighbor}
                available_new_pos = [p for p in positions if p not in used_positions]
                if available_new_pos:
                    new_pos = random.choice(available_new_pos)
                    neighbor[idx_to_move] = (*new_pos, neighbor[idx_to_move][2])
            elif move_type == 'swap_pos':
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                pos1, pos2 = neighbor[idx1][0:2], neighbor[idx2][0:2]
                bt1, bt2 = neighbor[idx1][2], neighbor[idx2][2]
                neighbor[idx1], neighbor[idx2] = (*pos2, bt1), (*pos1, bt2)
            elif move_type == 'swap_type':
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                if neighbor[idx1][2] != neighbor[idx2][2]:
                    bt1, bt2 = neighbor[idx1][2], neighbor[idx2][2]
                    neighbor[idx1] = (*neighbor[idx1][0:2], bt2)
                    neighbor[idx2] = (*neighbor[idx2][0:2], bt1)
            return neighbor

        current_placement = random.sample(positions, len(blocks_needed))
        random.shuffle(blocks_needed)
        current_solution = [(pos[0], pos[1], bt) for pos, bt in zip(current_placement, blocks_needed)]
        current_energy = get_energy(current_solution)
        
        temp = initial_temp
        while temp > final_temp and time.time() - start_time < timeout:
            for _ in range(iterations_per_temp):
                if time.time() - start_time > timeout: break
                if current_energy == 0:
                    self.solution = current_solution
                    return True
                neighbor_solution = generate_neighbor(current_solution)
                if not neighbor_solution: continue
                neighbor_energy = get_energy(neighbor_solution)
                energy_delta = neighbor_energy - current_energy
                if energy_delta < 0 or (current_energy > 0 and random.random() < math.exp(-energy_delta / temp)):
                    current_solution, current_energy = neighbor_solution, neighbor_energy
            temp *= alpha
            
        if get_energy(current_solution) == 0:
            self.solution = current_solution
            return True
        return False

    def estimate_complexity(self, positions: int, blocks: int) -> int:
        """Estimate solution space size.
        
        Args:
            positions (int): Number of available positions
            blocks (int): Number of blocks to place
            
        Returns:
            int: Estimated number of combinations to check
        """
        try:
            return math.comb(positions, blocks) * math.factorial(blocks)
        except ValueError:
            return positions ** blocks
    
    def solve_direct(self, blocks_needed, positions, start_time, timeout):
        """Enhanced direct brute force.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when solving started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        print("  Strategy: Enhanced brute force")
        self.strategy_used = "Enhanced Brute Force"
        
        max_attempts = 500_000
        ranked_positions = self.rank_positions(positions)
        sorted_positions = [pos for _, pos in ranked_positions]
        
        for i, pos_combo in enumerate(itertools.combinations(sorted_positions, len(blocks_needed))):
            for block_perm in itertools.permutations(blocks_needed):
                if i * math.factorial(len(blocks_needed)) > max_attempts or time.time() - start_time > timeout:
                    return False
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, block_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found in {i+1} main loops")
                    return True
        return False

    def solve_optimized(self, blocks_needed, positions, start_time, timeout):
        """Smart optimized search.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when solving started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        print("  Strategy: Smart optimized search")
        self.strategy_used = "Smart Optimized"
        
        strategies = [
            (self._try_top_positions, 0.4),
            (self._try_random_sampling, 0.3),
            (self._try_block_priority, 0.3)
        ]
        
        for strategy_func, time_fraction in strategies:
            if time.time() - start_time > timeout: break
            if strategy_func(blocks_needed, positions, time.time(), timeout * time_fraction):
                return True
        return False
    
    def _try_top_positions(self, blocks_needed, positions, start_time, timeout):
        """Try top-ranked positions.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when this strategy started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        ranked_positions = self.rank_positions(positions)
        top_positions = [pos for _, pos in ranked_positions[:min(20, len(positions))]]
        
        for i, pos_combo in enumerate(itertools.combinations(top_positions, len(blocks_needed))):
            for block_perm in itertools.permutations(blocks_needed):
                if i * math.factorial(len(blocks_needed)) > 50000 or time.time() - start_time > timeout: return False
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, block_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found with top positions")
                    return True
        return False
    
    def _try_random_sampling(self, blocks_needed, positions, start_time, timeout):
        """Enhanced random sampling.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when this strategy started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        ranked_positions = self.rank_positions(positions)
        top_60_count = max(1, int(len(ranked_positions) * 0.6))
        biased_positions = [pos for _, pos in ranked_positions[:top_60_count]]
        
        for i in range(50000):
            if time.time() - start_time > timeout: return False
            if len(positions) >= len(blocks_needed):
                pos_combo = random.sample(biased_positions if biased_positions else positions, min(len(blocks_needed), len(biased_positions if biased_positions else positions)))
                block_perm = random.sample(blocks_needed, len(blocks_needed))
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, block_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found with enhanced random sampling")
                    return True
        return False
    
    def _try_block_priority(self, blocks_needed, positions, start_time, timeout):
        """Try with smart block type ordering.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when this strategy started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        block_priority = {'B': 0, 'A': 1, 'C': 2}
        sorted_blocks = sorted(blocks_needed, key=lambda b: block_priority.get(b, 3))
        ranked_positions = self.rank_positions(positions)
        good_positions = [pos for _, pos in ranked_positions[:min(15, len(positions))]]
        
        for i, pos_combo in enumerate(itertools.combinations(good_positions, len(blocks_needed))):
            for block_perm in itertools.permutations(sorted_blocks):
                if i * math.factorial(len(blocks_needed)) > 25000 or time.time() - start_time > timeout: return False
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, block_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found with block priority")
                    return True
        return False
    
    def solve_heuristic(self, blocks_needed, positions, start_time, timeout):
        """Multi-phase heuristic search.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when solving started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        print("  Strategy: Multi-phase heuristic")
        self.strategy_used = "Multi-Phase Heuristic"
        
        phases = [("random", 0.5), ("targeted", 0.2), ("systematic", 0.3)]
        
        for phase_name, time_fraction in phases:
            if time.time() - start_time > timeout: break
            phase_timeout = timeout * time_fraction
            if phase_name == "targeted" and self._heuristic_targeted(blocks_needed, positions, time.time(), phase_timeout): return True
            if phase_name == "systematic" and self._heuristic_systematic(blocks_needed, positions, time.time(), phase_timeout): return True
            if phase_name == "random" and self._heuristic_random(blocks_needed, positions, time.time(), phase_timeout): return True
        return False
    
    def _heuristic_targeted(self, blocks_needed, positions, start_time, timeout):
        """Focus on most promising positions.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when this phase started
            timeout: Maximum time allowed for this phase
            
        Returns:
            bool: True if solution found, False otherwise
        """
        ranked_positions = self.rank_positions(positions)
        top_positions = [pos for _, pos in ranked_positions[:min(12, len(positions))]]
        block_priority = {'B': 0, 'A': 1, 'C': 2}
        sorted_blocks = sorted(blocks_needed, key=lambda b: block_priority.get(b, 3))
        
        for i, pos_combo in enumerate(itertools.combinations(top_positions, len(blocks_needed))):
            for block_perm in itertools.permutations(sorted_blocks):
                if i * math.factorial(len(blocks_needed)) > 15000 or time.time() - start_time > timeout: return False
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, block_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found with targeted search")
                    return True
        return False
    
    def _heuristic_systematic(self, blocks_needed, positions, start_time, timeout):
        """Systematic but limited exploration.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when this phase started
            timeout: Maximum time allowed for this phase
            
        Returns:
            bool: True if solution found, False otherwise
        """
        ranked_positions = self.rank_positions(positions)
        good_positions = [pos for _, pos in ranked_positions[:min(20, len(positions))]]
        unique_blocks = list(set(blocks_needed))
        
        for i, pos_combo in enumerate(itertools.combinations(good_positions, len(blocks_needed))):
            for base_perm in itertools.permutations(unique_blocks):
                if i * math.factorial(len(unique_blocks)) > 25000 or time.time() - start_time > timeout: return False
                block_counts = {bt: blocks_needed.count(bt) for bt in unique_blocks}
                full_perm = [bt for b in base_perm for bt in [b]*block_counts[b]]
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, full_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found with systematic search")
                    return True
        return False
    
    def _heuristic_random(self, blocks_needed, positions, start_time, timeout):
        """Pure random exploration.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when this phase started
            timeout: Maximum time allowed for this phase
            
        Returns:
            bool: True if solution found, False otherwise
        """
        max_attempts = 500_000
        if 'showstopper_4' in self.board.name.lower() or 'numbered_6' in self.board.name.lower():
            max_attempts = 1_000_000
        
        for i in range(max_attempts):
            if time.time() - start_time > timeout: return False
            if len(positions) >= len(blocks_needed):
                pos_combo = random.sample(positions, len(blocks_needed))
                block_perm = random.sample(blocks_needed, len(blocks_needed))
                placement = [(pos[0], pos[1], bt) for pos, bt in zip(pos_combo, block_perm)]
                if self.test_solution(placement):
                    self.solution = placement
                    print(f"  Solution found with random search in {i+1} attempts")
                    return True
        return False
    
    def solve_genetic(self, blocks_needed, positions, start_time, timeout):
        """Genetic algorithm for complex puzzles.
        
        Args:
            blocks_needed: List of block types needed for the solution
            positions: List of available positions for block placement
            start_time: Time when solving started
            timeout: Maximum time allowed for this strategy
            
        Returns:
            bool: True if solution found, False otherwise
        """
        print("  Strategy: Genetic Algorithm")
        self.strategy_used = "Genetic Algorithm"
        
        population_size = 200
        max_generations = 1000
        if 'showstopper_4' in self.board.name.lower() or 'numbered_6' in self.board.name.lower():
            population_size, max_generations = 600, 2500
        
        def generate_individual():
            """Generate a random individual for the genetic algorithm.
            
            Args:
                None
                
            Returns:
                List[Tuple[int, int, str]] or None: Random block placement or None if insufficient positions
            """
            if len(positions) < len(blocks_needed): return None
            selected_positions = random.sample(positions, len(blocks_needed))
            block_types = random.sample(blocks_needed, len(blocks_needed))
            return [(x, y, bt) for (x, y), bt in zip(selected_positions, block_types)]
        
        def fitness(individual):
            """Calculate fitness score for an individual solution.
            
            Args:
                individual: List of block placements as (x, y, block_type) tuples
                
            Returns:
                int: Number of targets hit by this individual's laser configuration
            """
            if individual is None or len(set(p[0:2] for p in individual)) != len(blocks_needed): return 0
            try:
                all_blocks = self.board.fixed_blocks + [Block(bt, x, y) for x, y, bt in individual]
                hit_points = self.engine.get_all_hit_points(self.board.lasers, all_blocks)
                return len(set(self.board.targets).intersection(hit_points))
            except: return 0
        
        def tournament_selection(population, fitnesses):
            """Select an individual using tournament selection.
            
            Args:
                population: List of individuals in the population
                fitnesses: List of fitness scores corresponding to the population
                
            Returns:
                Individual with highest fitness from random tournament sample
            """
            return max(random.sample(list(zip(population, fitnesses)), 5), key=lambda x: x[1])[0]

        def crossover(p1, p2):
            """Perform crossover between two parent individuals.
            
            Args:
                p1: First parent individual
                p2: Second parent individual
                
            Returns:
                Tuple[List, List]: Two offspring individuals created from crossover
            """
            size = len(p1)
            p1_pos, p2_pos = [i[0:2] for i in p1], [i[0:2] for i in p2]
            c1, c2 = [None] * size, [None] * size
            cx1, cx2 = sorted(random.sample(range(size), 2))
            c1[cx1:cx2], c2[cx1:cx2] = p1[cx1:cx2], p2[cx1:cx2]
            c1_pos_set, c2_pos_set = {i[0:2] for i in c1 if i}, {i[0:2] for i in c2 if i}

            for i in range(size):
                if c1[i] is None:
                    gene, pos = p2[i], p2[i][0:2]
                    while pos in c1_pos_set:
                        gene, pos = p2[p1_pos.index(pos)], p2[p1_pos.index(pos)][0:2]
                    c1[i], c1_pos_set = gene, c1_pos_set | {pos}
            for i in range(size):
                if c2[i] is None:
                    gene, pos = p1[i], p1[i][0:2]
                    while pos in c2_pos_set:
                        gene, pos = p1[p2_pos.index(pos)], p1[p2_pos.index(pos)][0:2]
                    c2[i], c2_pos_set = gene, c2_pos_set | {pos}
            return c1, c2

        def mutate(individual):
            """Apply mutation to an individual with small probability.
            
            Args:
                individual: Individual to mutate (list of block placements)
                
            Returns:
                List: Mutated individual with possible position or block type swaps
            """
            mutated = individual[:]
            if random.random() < 0.1 and len(mutated) > 1:
                idx1, idx2 = random.sample(range(len(mutated)), 2)
                pos1, pos2 = mutated[idx1][0:2], mutated[idx2][0:2]
                bt1, bt2 = mutated[idx1][2], mutated[idx2][2]
                mutated[idx1], mutated[idx2] = (*pos2, bt1), (*pos1, bt2)
            if random.random() < 0.1 and len(mutated) > 1 and len(set(g[2] for g in mutated)) > 1:
                idx1, idx2 = random.sample(range(len(mutated)), 2)
                while mutated[idx1][2] == mutated[idx2][2]: idx1, idx2 = random.sample(range(len(mutated)), 2)
                bt1, bt2 = mutated[idx1][2], mutated[idx2][2]
                mutated[idx1], mutated[idx2] = (*mutated[idx1][0:2], bt2), (*mutated[idx2][0:2], bt1)
            return mutated
        
        population = [generate_individual() for _ in range(population_size)]
        for generation in range(max_generations):
            if time.time() - start_time > timeout: break
            fitnesses = [fitness(ind) for ind in population]
            if max(fitnesses) == len(self.board.targets):
                best_ind = population[fitnesses.index(max(fitnesses))]
                if self.test_solution(best_ind):
                    self.solution = best_ind
                    print(f"  Solution found in generation {generation}")
                    return True
            new_population = []
            for _ in range(population_size // 2):
                p1, p2 = tournament_selection(population, fitnesses), tournament_selection(population, fitnesses)
                c1, c2 = crossover(p1, p2)
                new_population.extend([mutate(c1), mutate(c2)])
            population = new_population
        return False
    
    def test_solution(self, placement: List[Tuple[int, int, str]]) -> bool:
        """Test if a placement solves the puzzle.
        
        Args:
            placement (List[Tuple[int, int, str]]): List of block placements as (x, y, block_type) tuples
            
        Returns:
            bool: True if the placement solves the puzzle, False otherwise
        """
        if len(set(p[0:2] for p in placement)) != len(placement): return False
        try:
            all_blocks = self.board.fixed_blocks + [Block(bt, x, y) for x, y, bt in placement]
            return self.engine.simulate_all_lasers(self.board.lasers, all_blocks, set(self.board.targets))
        except: return False

    def rank_positions(self, positions: List[Tuple[int, int]]) -> List[Tuple[float, Tuple[int, int]]]:
        """Enhanced position ranking.
        
        Args:
            positions (List[Tuple[int, int]]): List of available positions as (x, y) coordinates
            
        Returns:
            List[Tuple[float, Tuple[int, int]]]: List of (score, position) tuples sorted by score descending
        """
        scored = []
        for x, y in positions:
            score = 0
            for tx, ty in self.board.targets:
                score += max(0, 30 - (abs(x - tx) + abs(y - ty)) * 2)
            for lx, ly, vx, vy in self.board.lasers:
                if vx != 0 and abs((ly - y) * vx - (lx - x) * vy) <= 2: score += 15
                score += max(0, 20 - (abs(x - lx) + abs(y - ly)))
            if x <= 1 or x >= self.board.grid_width * 2 - 1: score -= 5
            if y <= 1 or y >= self.board.grid_height * 2 - 1: score -= 5
            score += max(0, 10 - abs(x - self.board.grid_width) - abs(y - self.board.grid_height))
            scored.append((score, (x, y)))
        return sorted(scored, reverse=True)
