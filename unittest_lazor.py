"""
Unit tests for Lazor Puzzle Project
===================================
Ensures correctness of parser, board, engine, solver, and I/O logic.
"""

import unittest
import os
from pathlib import Path
from lazor_board import LazorBoard, Block
from bff_parser import parse_bff_file
from laser_engine import LaserEngine
from solver import AdvancedSolver
from Lazor import save_solution_file


# ----------------------------------------------------------
# 1️⃣  BFF File Parser Tests
# ----------------------------------------------------------
class TestBFFParser(unittest.TestCase):
    """Tests for parsing .bff puzzle files."""

    def test_parse_creates_board(self):
        """Ensure parser returns a LazorBoard instance.
        
        Args:
            None
            
        Returns:
            None
        """
        test_file = Path("bff_files") / "mad_1.bff"
        board = parse_bff_file(str(test_file))
        self.assertIsInstance(board, LazorBoard)

    def test_board_has_dimensions(self):
        """Parsed board must have width and height > 0.
        
        Args:
            None
            
        Returns:
            None
        """
        test_file = Path("bff_files") / "mad_1.bff"
        board = parse_bff_file(str(test_file))
        self.assertGreater(board.grid_width, 0)
        self.assertGreater(board.grid_height, 0)


# ----------------------------------------------------------
# 2️⃣  LazorBoard Tests
# ----------------------------------------------------------
class TestLazorBoard(unittest.TestCase):
    """Tests for LazorBoard structure and methods."""

    def setUp(self):
        """Set up test board for LazorBoard tests.
        
        Args:
            None
            
        Returns:
            None
        """
        self.board = LazorBoard()
        self.board.set_dimensions(4, 4)

    def test_add_fixed_block(self):
        """Adding a fixed block should increase fixed_blocks list.
        
        Args:
            None
            
        Returns:
            None
        """
        self.board.add_fixed_block('A', 1, 1)
        self.assertEqual(len(self.board.fixed_blocks), 1)

    def test_add_open_position(self):
        """Adding open positions should store coordinates.
        
        Args:
            None
            
        Returns:
            None
        """
        self.board.add_open_position(2, 2)
        self.assertIn((2, 2), self.board.open_positions)


# ----------------------------------------------------------
# 3️⃣  LaserEngine Tests
# ----------------------------------------------------------
class TestLaserEngine(unittest.TestCase):
    """Tests for laser simulation logic."""

    def setUp(self):
        """Set up test engine and blocks for LaserEngine tests.
        
        Args:
            None
            
        Returns:
            None
        """
        self.engine = LaserEngine(5, 5)
        self.blocks = [Block('A', 3, 3)]

    def test_reflection_logic(self):
        """Reflection must invert direction correctly.
        
        Args:
            None
            
        Returns:
            None
        """
        new_dx, new_dy = self.engine.calculate_reflection(2, 3, 1, 0, Block('A', 3, 3))
        self.assertEqual(new_dx, -1)

    def test_simulate_all_lasers_returns_bool(self):
        """simulate_all_lasers should always return a boolean.
        
        Args:
            None
            
        Returns:
            None
        """
        lasers = [(0, 0, 1, 0)]
        targets = {(4, 0)}
        result = self.engine.simulate_all_lasers(lasers, self.blocks, targets)
        self.assertIsInstance(result, bool)


# ----------------------------------------------------------
# 4️⃣  AdvancedSolver Tests
# ----------------------------------------------------------
class TestAdvancedSolver(unittest.TestCase):
    """Tests for solver strategies."""

    def setUp(self):
        """Set up test board and solver for AdvancedSolver tests.
        
        Args:
            None
            
        Returns:
            None
        """
        board = LazorBoard()
        board.set_dimensions(4, 4)
        board.lasers = [(0, 0, 1, 0)]
        board.targets = [(2, 0)]
        board.open_positions = [(1, 1), (2, 2)]
        board.set_available_blocks({'A': 1})
        self.solver = AdvancedSolver(board)

    def test_solver_returns_bool(self):
        """Solver.solve() should return a boolean.
        
        Args:
            None
            
        Returns:
            None
        """
        result = self.solver.solve(timeout=1)
        self.assertIsInstance(result, bool)

    def test_estimate_complexity_valid(self):
        """estimate_complexity() must return integer.
        
        Args:
            None
            
        Returns:
            None
        """
        c = self.solver.estimate_complexity(5, 2)
        self.assertIsInstance(c, int)

    def test_rank_positions_output(self):
        """rank_positions() must return list of tuples.
        
        Args:
            None
            
        Returns:
            None
        """
        ranked = self.solver.rank_positions([(1, 1), (2, 2)])
        self.assertIsInstance(ranked, list)
        self.assertIsInstance(ranked[0], tuple)


# ----------------------------------------------------------
# 5️⃣  File Output Tests
# ----------------------------------------------------------
class TestOutputFile(unittest.TestCase):
    """Tests for saving solutions."""

    def test_save_solution_creates_file(self):
        """Ensure solution file is written.
        
        Args:
            None
            
        Returns:
            None
        """
        sol_dir = Path("solutions")
        sol_dir.mkdir(exist_ok=True)
        file_path = sol_dir / "temp_solution.txt"

        save_solution_file("dummy_puzzle.bff", [(1, 1, 'A')], 0.123)
        self.assertTrue(any(sol_dir.iterdir()))


# ----------------------------------------------------------
# 6️⃣  Integration Test
# ----------------------------------------------------------
class TestIntegration(unittest.TestCase):
    """Test full pipeline from BFF to solver."""

    def test_parse_and_solve_pipeline(self):
        """Full integration check for one puzzle.
        
        Args:
            None
            
        Returns:
            None
        """
        test_file = Path("bff_files") / "mad_1.bff"
        if not test_file.exists():
            self.skipTest("mad_1.bff not found")
        board = parse_bff_file(str(test_file))
        solver = AdvancedSolver(board)
        result = solver.solve(timeout=3)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
