import random
from mcts.mcts import MCTS

class Player:
    def __init__(self, symbol):
        if symbol.upper() not in ['X', 'O']:
            raise ValueError('Player symbol must be X or O')
        self.symbol = symbol.upper()

    def make_move(self, board):
        """prompt the user to make a move in the console"""
        while True:
            user_input = input("Enter row and col: ")
            parts = user_input.split(',')
            if len(parts) != 2:
                print("Please enter exactly two numbers separated by ','")
                continue
            try:
                row, col = int(parts[0]) - 1, int(parts[1]) - 1 # adjust for 0-indexing
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
                continue
            if row < 0 or row >= board.get_size() or col < 0 or col >= board.get_size():
                print("Invalid input. Please enter numbers within the board size.")
                continue
            
            if board.make_move(row, col, self.symbol):
                return True
            else:
                print("Invalid move. Try again.")
    

class RandomPlayer(Player):
    def __init__(self, symbol):
        super().__init__(symbol)

    def make_move(self, board):
        """randomly make a move"""
        size = board.get_size()
        while True:
            row = random.randint(0, size-1)
            col = random.randint(0, size-1)
            if board.make_move(row, col, self.symbol):
                return True
            
class MCTSPlayer(Player):
    def __init__(self, symbol, simulations=1000, exploration_constant=2):
        super().__init__(symbol)
        self.mcts = MCTS(simulations=simulations, exploration_constant=exploration_constant)
    
    def make_move(self, board):
        """use MCTS to select a move"""
        move = self.mcts.search(board, self.symbol)
        if move:
            board.make_move(move[0], move[1], self.symbol)
            return True
        return False