"""
BFF File Parser for Lazor Puzzles
=================================
This module provides a function to parse .bff files and create a
LazorBoard object from the file's contents.
"""

from pathlib import Path
from lazor_board import LazorBoard


def parse_bff_file(filename: str) -> LazorBoard:
    """Parse .bff file format.
    
    Args:
        filename (str): Path to the .bff file to parse
        
    Returns:
        LazorBoard: Populated LazorBoard object with puzzle data from the file
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip()
    
    lines = [line.strip() for line in content.split('\n')]
    lines = [line for line in lines if line and not line.startswith('#')]
    
    board = LazorBoard()
    board.name = Path(filename).name
    
    grid_start_idx = lines.index("GRID START")
    grid_stop_idx = lines.index("GRID STOP")
    grid_lines = lines[grid_start_idx + 1:grid_stop_idx]
    
    height = len(grid_lines)
    width = len(grid_lines[0].split()) if grid_lines else 0
    board.set_dimensions(width, height)
    
    for row_idx, line in enumerate(grid_lines):
        cells = line.split()
        for col_idx, cell in enumerate(cells):
            block_x = col_idx * 2 + 1
            block_y = row_idx * 2 + 1
            
            if cell in ['A', 'B', 'C']:
                board.add_fixed_block(cell, block_x, block_y)
            elif cell == 'o':
                board.add_open_position(block_x, block_y)
    
    available_blocks = {}
    for line in lines[grid_stop_idx + 1:]:
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        if parts[0] in ['A', 'B', 'C']:
            available_blocks[parts[0]] = int(parts[1])
        elif parts[0] == 'L' and len(parts) >= 5:
            board.add_laser(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        elif parts[0] == 'P' and len(parts) >= 3:
            board.add_target(int(parts[1]), int(parts[2]))
    
    board.set_available_blocks(available_blocks)
    return board
