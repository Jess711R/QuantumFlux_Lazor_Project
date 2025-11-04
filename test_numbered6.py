#!/usr/bin/env python3
"""
Test script for numbered_6.bff
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the function we need
from Lazor import solve_single_puzzle

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(__file__))
    success = solve_single_puzzle("bff_files/numbered_6.bff")
    print(f"numbered_6 solved: {success}")