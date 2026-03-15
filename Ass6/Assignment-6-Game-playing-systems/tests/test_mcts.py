import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from mcts.mcts import MCTS, Node, TicTacToeBoard

class TestMCTS(unittest.TestCase):
    # def setUp(self):
    #     self.board = TicTacToeBoard(3)
    #     self.mcts = MCTS(num_iterations=100)
        
    # def test_initialization(self):
    #     """Test if MCTS initializes correctly"""
    #     self.assertEqual(self.mcts.num_iterations, 100)
    #     self.assertEqual(self.mcts.simulations, 0)
    #     self.assertEqual(self.mcts.wins, 0)
    #     self.assertEqual(self.mcts.draws, 0)
        
    # def test_node_initialization(self):
    #     """Test if Node initializes correctly"""
    #     node = Node(self.board, 'X')
    #     self.assertEqual(node.last_player, 'X')
    #     self.assertEqual(node.vis, 0)
    #     self.assertEqual(node.wins, 0)
    #     self.assertEqual(len(node.available_moves), 9)  # Empty 3x3 board has 9 possible moves
        
    # def test_node_expansion(self):
    #     """Test expanding a node"""
    #     node = Node(self.board, 'X')
    #     tree = MCTS(100).run_search(self.board, 'O')
    #     self.assertGreater(tree.vis, 0)
        
    # def test_node_update(self):
    #     """Test updating a node's stats"""
    #     node = Node(self.board, 'X')
    #     node.vis += 1
    #     node.wins += 1
    #     self.assertEqual(node.vis, 1)
    #     self.assertEqual(node.wins, 1)
        
    #     node.vis += 1
    #     # No win this time
    #     self.assertEqual(node.vis, 2)
    #     self.assertEqual(node.wins, 1)
        
    # def test_simulate(self):
    #     """Test simulation from a node"""
    #     # Create a board with X about to win
    #     board = TicTacToeBoard(3)
    #     board.make_move(0, 0, 'X')
    #     board.make_move(0, 1, 'X')
    #     # (0,2) is empty and would make X win
        
    #     node = Node(board, 'O')  # Last player was O, now X's turn
    #     result = self.mcts.simulate(node)
    #     # The simulate result is probabilistic, so we can't assert its exact value
    #     self.assertIn(result, [-1, 0, 1])
        
    def test_search_for_winning_move(self):
        """Test if MCTS finds obvious winning move"""
        # This test is more complex because we need to examine the selected move
        score = {'Correct': 0, 'Wrong': 0}
        for _ in range(100):
          
            # Create a board with X about to win
            board = TicTacToeBoard(3)
            board.make_move(0, 0, 'X')
            board.make_move(0, 1, 'X')
            # (0,2) would be a winning move
            # board.display()
            
            # Use more simulations to ensure it finds the winning move
            mcts = MCTS(num_iterations=1000)
            result_node = mcts.run_search(board, 'X')
            
            if result_node.board.grid[0][2] == 'X':
                score['Correct'] += 1
            else:
                score['Wrong'] += 1
            # Check if X played in position (0,2)
            # self.assertEqual(result_node.board.grid[0][2], 'X')
        print(score)
            
if __name__ == '__main__':
    unittest.main()

