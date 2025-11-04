#!/usr/bin/env python3
"""
Test script for hard puzzles: showstopper_4 and numbered_6
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the function we need
from Lazor import solve_single_puzzle

def test_hard_puzzles():
    """Test the two hardest puzzles."""
    print("Testing Hard Puzzles: showstopper_4 and numbered_6")
    print("=" * 60)

    puzzles = [
        "bff_files/showstopper_4.bff",
        "bff_files/numbered_6.bff"
    ]

    solved = 0
    for puzzle in puzzles:
        if os.path.exists(puzzle):
            success = solve_single_puzzle(puzzle)
            if success:
                solved += 1
        else:
            print(f"File not found: {puzzle}")

    print(f"\nResults: {solved}/{len(puzzles)} hard puzzles solved")

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(__file__))
    test_hard_puzzles()