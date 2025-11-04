"""
Lazor Board and Block Representation
====================================
This module defines the data structures for the Lazor puzzle, including
the Block and LazorBoard classes.
"""

from typing import List, Tuple, Dict


class Block:
    """Block class for Lazor game pieces."""
    
    def __init__(self, block_type: str, x: int, y: int, is_fixed: bool = False):
        """Initialize a Block object.
        
        Args:
            block_type (str): Type of block ('A' for reflect, 'B' for opaque, 'C' for refract)
            x (int): X coordinate of the block on the grid
            y (int): Y coordinate of the block on the grid
            is_fixed (bool): Whether the block is fixed in place (default: False)
            
        Returns:
            None
        """
        self.block_type = block_type
        self.x = x
        self.y = y
        self.is_fixed = is_fixed
    
    def __repr__(self):
        """Return string representation of the Block.
        
        Args:
            None
            
        Returns:
            str: String representation of the block
        """
        return f"Block({self.block_type}, {self.x}, {self.y})"


class LazorBoard:
    """Game board representation."""
    
    def __init__(self):
        """Initialize a LazorBoard object with empty attributes.
        
        Args:
            None
            
        Returns:
            None
        """
        self.grid_width = 0
        self.grid_height = 0
        self.fixed_blocks: List[Block] = []
        self.available_blocks: Dict[str, int] = {}
        self.open_positions: List[Tuple[int, int]] = []
        self.lasers: List[Tuple[int, int, int, int]] = []
        self.targets: List[Tuple[int, int]] = []
        self.name: str = ""
    
    def set_dimensions(self, width: int, height: int):
        """Set the dimensions of the game board.
        
        Args:
            width (int): Width of the board in grid units
            height (int): Height of the board in grid units
            
        Returns:
            None
        """
        self.grid_width = width
        self.grid_height = height
    
    def add_fixed_block(self, block_type: str, x: int, y: int):
        """Add a fixed block to the board.
        
        Args:
            block_type (str): Type of block ('A' for reflect, 'B' for opaque, 'C' for refract)
            x (int): X coordinate of the block
            y (int): Y coordinate of the block
            
        Returns:
            None
        """
        self.fixed_blocks.append(Block(block_type, x, y, is_fixed=True))
    
    def add_open_position(self, x: int, y: int):
        """Add an open position where blocks can be placed.
        
        Args:
            x (int): X coordinate of the open position
            y (int): Y coordinate of the open position
            
        Returns:
            None
        """
        self.open_positions.append((x, y))
    
    def set_available_blocks(self, available: Dict[str, int]):
        """Set the available blocks for puzzle solving.
        
        Args:
            available (Dict[str, int]): Dictionary mapping block types to their counts
            
        Returns:
            None
        """
        self.available_blocks = available
    
    def add_laser(self, x: int, y: int, vx: int, vy: int):
        """Add a laser to the board.
        
        Args:
            x (int): Starting X coordinate of the laser
            y (int): Starting Y coordinate of the laser
            vx (int): X component of laser direction vector
            vy (int): Y component of laser direction vector
            
        Returns:
            None
        """
        self.lasers.append((x, y, vx, vy))
    
    def add_target(self, x: int, y: int):
        """Add a target point that must be hit by a laser.
        
        Args:
            x (int): X coordinate of the target
            y (int): Y coordinate of the target
            
        Returns:
            None
        """
        self.targets.append((x, y))
