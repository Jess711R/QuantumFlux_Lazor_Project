"""
Laser Engine for Lazor Puzzles
===============================
This module contains the LaserEngine class, which is responsible for
simulating the behavior of lasers, including reflection and refraction.
"""

from typing import List, Tuple, Set, Optional
from lazor_board import Block


class LaserEngine:
    """Robust laser simulation engine."""
    
    def __init__(self, grid_width: int, grid_height: int):
        """Initialize the LaserEngine with grid dimensions.
        
        Args:
            grid_width (int): Width of the game grid
            grid_height (int): Height of the game grid
            
        Returns:
            None
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = 500
        self.max_beams = 50
    
    def simulate_all_lasers(self, lasers: List[Tuple[int, int, int, int]], 
                           blocks: List[Block], targets: Set[Tuple[int, int]]) -> bool:
        """Main simulation function - returns True if all targets are hit.
        
        Args:
            lasers (List[Tuple[int, int, int, int]]): List of laser starting positions and directions
            blocks (List[Block]): List of all blocks on the board
            targets (Set[Tuple[int, int]]): Set of target coordinates that must be hit
            
        Returns:
            bool: True if all targets are hit by lasers, False otherwise
        """
        all_hit_points = self.get_all_hit_points(lasers, blocks)
        return targets.issubset(all_hit_points)
    
    def get_all_hit_points(self, lasers: List[Tuple[int, int, int, int]], 
                          blocks: List[Block]) -> Set[Tuple[int, int]]:
        """Return all hit points from lasers.
        
        Args:
            lasers (List[Tuple[int, int, int, int]]): List of laser starting positions and directions
            blocks (List[Block]): List of all blocks on the board
            
        Returns:
            Set[Tuple[int, int]]: Set of all coordinates hit by any laser beam
        """
        all_hit_points = set()
        
        for start_x, start_y, vx, vy in lasers:
            laser_points = self.trace_laser_path(start_x, start_y, vx, vy, blocks)
            all_hit_points.update(laser_points)
        
        return all_hit_points
    
    def trace_laser_path(self, x: int, y: int, vx: int, vy: int, 
                        blocks: List[Block]) -> Set[Tuple[int, int]]:
        """Trace a laser path with beam splitting support.
        
        Args:
            x (int): Starting X coordinate of the laser
            y (int): Starting Y coordinate of the laser
            vx (int): X component of laser direction
            vy (int): Y component of laser direction
            blocks (List[Block]): List of blocks that can interact with the laser
            
        Returns:
            Set[Tuple[int, int]]: Set of all points hit by this laser and its split beams
        """
        hit_points = set()
        laser_queue = [(x, y, vx, vy)]
        processed_beams = set()
        
        while laser_queue and len(laser_queue) <= self.max_beams:
            current_x, current_y, dx, dy = laser_queue.pop(0)
            beam_signature = (current_x, current_y, dx, dy)
            
            if beam_signature in processed_beams:
                continue
            
            processed_beams.add(beam_signature)
            beam_points = self.simulate_single_beam(current_x, current_y, dx, dy, blocks)
            
            refract_beams = self.check_for_refraction(current_x, current_y, dx, dy, blocks)
            for new_beam in refract_beams:
                if new_beam not in processed_beams and len(laser_queue) < self.max_beams:
                    laser_queue.append(new_beam)
            
            hit_points.update(beam_points)
        
        return hit_points
    
    def simulate_single_beam(self, x: int, y: int, dx: int, dy: int, 
                           blocks: List[Block]) -> Set[Tuple[int, int]]:
        """Simulate a single laser beam.
        
        Args:
            x (int): Starting X coordinate of the beam
            y (int): Starting Y coordinate of the beam
            dx (int): X direction component of the beam
            dy (int): Y direction component of the beam
            blocks (List[Block]): List of blocks that can interact with the beam
            
        Returns:
            Set[Tuple[int, int]]: Set of all points hit by this single beam
        """
        points = set()
        visited_states = set()
        
        for _ in range(self.max_steps):
            points.add((x, y))
            
            state = (x, y, dx, dy)
            if state in visited_states:
                break
            visited_states.add(state)
            
            next_x = x + dx
            next_y = y + dy
            
            if (next_x < 0 or next_x > self.grid_width * 2 or
                next_y < 0 or next_y > self.grid_height * 2):
                break
            
            hit_block = self.find_block_collision(next_x, next_y, blocks)
            
            if hit_block:
                if hit_block.block_type == 'A':
                    dx, dy = self.calculate_reflection(next_x, next_y, dx, dy, hit_block)
                elif hit_block.block_type == 'B':
                    break
                elif hit_block.block_type == 'C':
                    pass
            
            x, y = next_x, next_y
        
        return points
    
    def find_block_collision(self, x: int, y: int, blocks: List[Block]) -> Optional[Block]:
        """Check if laser position collides with any block.
        
        Args:
            x (int): X coordinate to check for collision
            y (int): Y coordinate to check for collision
            blocks (List[Block]): List of blocks to check against
            
        Returns:
            Optional[Block]: The block that collides with the position, or None if no collision
        """
        for block in blocks:
            if (abs(x - block.x) <= 1 and abs(y - block.y) <= 1):
                on_vertical_edge = (x == block.x - 1 or x == block.x + 1) and (block.y - 1 <= y <= block.y + 1)
                on_horizontal_edge = (y == block.y - 1 or y == block.y + 1) and (block.x - 1 <= x <= block.x + 1)
                
                if on_vertical_edge or on_horizontal_edge:
                    return block
        return None
    
    def calculate_reflection(self, hit_x: int, hit_y: int, dx: int, dy: int, 
                           block: Block) -> Tuple[int, int]:
        """Calculate reflection direction based on hit edge.
        
        Args:
            hit_x (int): X coordinate where laser hits the block
            hit_y (int): Y coordinate where laser hits the block
            dx (int): Current X direction of the laser
            dy (int): Current Y direction of the laser
            block (Block): The block being hit
            
        Returns:
            Tuple[int, int]: New direction vector (new_dx, new_dy) after reflection
        """
        hit_left = (hit_x == block.x - 1)
        hit_right = (hit_x == block.x + 1) 
        hit_top = (hit_y == block.y - 1)
        hit_bottom = (hit_y == block.y + 1)
        
        new_dx, new_dy = dx, dy
        
        if hit_left or hit_right:
            new_dx = -dx
        if hit_top or hit_bottom:
            new_dy = -dy
            
        return new_dx, new_dy
    
    def check_for_refraction(self, start_x: int, start_y: int, dx: int, dy: int, 
                           blocks: List[Block]) -> List[Tuple[int, int, int, int]]:
        """Check if beam path intersects refract blocks and return reflected beams.
        
        Args:
            start_x (int): Starting X coordinate of the beam
            start_y (int): Starting Y coordinate of the beam
            dx (int): X direction component of the beam
            dy (int): Y direction component of the beam
            blocks (List[Block]): List of blocks to check for refraction
            
        Returns:
            List[Tuple[int, int, int, int]]: List of new beam parameters (x, y, dx, dy) from refraction
        """
        refract_beams = []
        x, y = start_x, start_y
        visited_positions = set()
        
        for _ in range(min(self.max_steps, 100)):
            next_x = x + dx
            next_y = y + dy
            
            pos_state = (next_x, next_y, dx, dy)
            if pos_state in visited_positions:
                break
            visited_positions.add(pos_state)
            
            if (next_x < 0 or next_x > self.grid_width * 2 or
                next_y < 0 or next_y > self.grid_height * 2):
                break
            
            hit_block = self.find_block_collision(next_x, next_y, blocks)
            
            if hit_block:
                if hit_block.block_type == 'C':
                    ref_dx, ref_dy = self.calculate_reflection(next_x, next_y, dx, dy, hit_block)
                    if (ref_dx, ref_dy) != (dx, dy):
                        refract_beams.append((next_x, next_y, ref_dx, ref_dy))
                elif hit_block.block_type in ['A', 'B']:
                    break
            
            x, y = next_x, next_y
        
        return refract_beams
