import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from game.board import Board

class TestBoard(unittest.TestCase):
    board_size = 3
    def setUp(self):
        self.board = Board(self.board_size)  # Using board_size variable
    
    def test_initialization(self):
        """Test if board initializes correctly with empty spaces"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                self.assertEqual(self.board.grid[row][col], ' ')
        self.assertEqual(self.board.size, self.board_size)
        
    def test_make_move(self):
        """Test making valid and invalid moves"""
        # Test valid move
        self.assertTrue(self.board.make_move(0, 0, 'X'))
        self.assertEqual(self.board.grid[0][0], 'X')
        
        # Test invalid move (cell already occupied)
        self.assertFalse(self.board.make_move(0, 0, 'O'))
        self.assertEqual(self.board.grid[0][0], 'X')  # Should remain unchanged
        
    def test_get_available_moves(self):
        """Test getting available moves"""
        # Initial board has all cells available
        self.assertEqual(len(self.board.get_available_moves()), self.board_size * self.board_size)
        
        # Make a move and check if available moves decreases
        self.board.make_move(0, 0, 'X')
        moves = self.board.get_available_moves()
        self.assertEqual(len(moves), self.board_size * self.board_size - 1)
        self.assertNotIn((0, 0), moves)
        
    def test_check_winner_horizontal(self):
        """Test horizontal winning condition"""
        # Create a winning horizontal line
        for col in range(self.board_size):
            self.board.make_move(0, col, 'X')
        self.assertEqual(self.board.check_winner(), 'X')
        
    def test_check_winner_vertical(self):
        """Test vertical winning condition"""
        # Create a winning vertical line
        for row in range(self.board_size):
            self.board.make_move(row, 0, 'O')
        self.assertEqual(self.board.check_winner(), 'O')
        
    def test_check_winner_diagonal(self):
        """Test diagonal winning condition"""
        # Create a winning diagonal line
        for i in range(self.board_size):
            self.board.make_move(i, i, 'X')
        self.assertEqual(self.board.check_winner(), 'X')
        
    def test_is_full(self):
        """Test checking if board is full"""
        self.assertFalse(self.board.is_full())
        
        # Fill the board
        for row in range(self.board_size):
            for col in range(self.board_size):
                self.board.make_move(row, col, 'X')
                
        self.assertTrue(self.board.is_full())

if __name__ == '__main__':
    unittest.main()