"""
FINAL LAZOR SOLVER - MAIN ENTRY POINT
=====================================
This script orchestrates the solving of Lazor puzzles.

Authors: Pratiti Pradhan, Jay Modi, Jessica Ravindran
Date: 04th November 2025
"""

import time
from pathlib import Path
import os
from typing import List, Tuple, Optional

from bff_parser import parse_bff_file
from solver import AdvancedSolver


def solve_single_puzzle(filename: str) -> bool:
    """Solve one puzzle file.
    
    Args:
        filename (str): Path to the .bff puzzle file to solve
        
    Returns:
        bool: True if puzzle was solved successfully, False otherwise
    """
    puzzle_name = Path(filename).name
    print(f"\nSolving {puzzle_name}...")
    
    try:
        board = parse_bff_file(filename)
        solver = AdvancedSolver(board)
        
        total_blocks = sum(board.available_blocks.values())
        if total_blocks <= 3: timeout = 120
        elif total_blocks <= 5: timeout = 300
        else: timeout = 600

        hard_puzzles = {'mad_1', 'showstopper_4', 'yarn_5', 'numbered_6'}
        if Path(filename).stem in hard_puzzles:
            timeout = max(timeout, 45)
        
        print(f"  Using timeout: {timeout} seconds")
        
        success = solver.solve(timeout)
        
        if success:
            print(f"SOLVED in {solver.solve_time:.3f}s using {solver.strategy_used}")
            save_solution_file(filename, solver.solution, solver.solve_time)
            if solver.solution:
                block_names = {'A': 'Reflect', 'B': 'Opaque', 'C': 'Refract'}
                print("  Placement:")
                for x, y, bt in solver.solution:
                    print(f"    {block_names[bt]} at ({x}, {y})")
            else:
                print("  No blocks needed")
            return True
        else:
            print(f"FAILED after {solver.solve_time:.3f}s")
            return False
    
    except Exception:
        import traceback
        print(f"An unexpected error occurred while solving {puzzle_name}:")
        traceback.print_exc()
        return False


def save_solution_file(puzzle_file: str, solution: Optional[List[Tuple[int, int, str]]], solve_time: float):
    """Save solution to output file.
    
    Args:
        puzzle_file (str): Path to the original puzzle file
        solution (Optional[List[Tuple[int, int, str]]]): List of block placements as (x, y, block_type) tuples, or None if no solution
        solve_time (float): Time taken to solve the puzzle in seconds
        
    Returns:
        None
    """
    Path("solutions").mkdir(exist_ok=True)
    output_file = Path("solutions") / f"{Path(puzzle_file).stem}_solution.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Solution for {Path(puzzle_file).stem}\n"
                f"==================================================\n\n"
                f"Solved in {solve_time:.3f} seconds\n\n")
        
        if solution:
            f.write("Block placements:\n")
            block_names = {'A': 'Reflect', 'B': 'Opaque', 'C': 'Refract'}
            for x, y, block_type in solution:
                f.write(f"  {block_names[block_type]} block at position ({x}, {y})\n")
        else:
            f.write("No blocks needed to be placed.\n")


def main():
    """Main function that orchestrates solving all puzzle files in the bff_files directory.
    
    Args:
        None
        
    Returns:
        None
    """
    print("Lazor Puzzle Solver\n" + "=" * 50)
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    bff_files = sorted(Path("bff_files").glob("*.bff"))
    if not bff_files:
        print(f"No .bff files found in bff_files directory: {Path('bff_files').absolute()}")
        return
    
    print(f"Found {len(bff_files)} puzzle files")
    
    start_time = time.time()
    solved_count = sum(1 for bff_file in bff_files if solve_single_puzzle(str(bff_file)))
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 50 + f"\nSolved {solved_count}/{len(bff_files)} puzzles in {total_time:.3f} seconds")
    if solved_count == len(bff_files):
        print("All puzzles solved!")
    else:
        print(f"{len(bff_files) - solved_count} puzzles unsolved")


if __name__ == "__main__":
    main()