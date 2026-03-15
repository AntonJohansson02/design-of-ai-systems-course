# import math
# import random
# from copy import deepcopy

# class MCTS:
#     def __init__(self, simulations, exploration_constant):
#         self.simulations = simulations
#         self.exploration_constant = exploration_constant

#     def search(self, board, root_player):
#         """search for the best move from the root"""
#         # exhaustively check for an obvious move first, avoiding the expensive MCTS
#         obvious_move = self.obvious_move(board, root_player)
#         if obvious_move is not None:
#             return obvious_move

#         # if not found, start the MCTS
#         root = Node(board, root_player)
        
#         for _ in range(self.simulations):
#             # first select best child until a leaf node is reached
#             node = root
#             while node.is_fully_expanded() and node.children:
#                 node = node.get_best_child(self.exploration_constant)
            
#             # expand if it is possible, ie not last node and not already fully expanded
#             if not node.is_fully_expanded() and not node.board.check_winner() and not node.board.is_full():
#                 move = random.choice(node.untried_moves)
#                 node = node.add_child(move, node.board, node.player)
            
#             # simulate the game randomly
#             result = self.simulate(node, root.player)
            
#             # update every node up until the root with the result
#             self.backpropagate(node, result)
        
#         # when finished, return best move from root
#         return self.best_move(root)
    

#     def obvious_move(self, board, player):
#         moves = board.get_available_moves()
#         # first check if any available move is a winning move (makes the algoritm stronger, also the algoritm is much faster in the last move)
#         for move in moves:
#             test_board = deepcopy(board)
#             test_board.make_move(move[0], move[1], player)
#             if test_board.check_winner() == player:
#                 return move
#         # second check if the opponent has a winning move, if so block it
#         opponent = 'O' if player == 'X' else 'X'
#         for move in moves:
#             test_board = deepcopy(board)
#             test_board.make_move(move[0], move[1], opponent)
#             if test_board.check_winner() == opponent:
#                 return move
#         return None # return none if no obvious move is found
    
#     def simulate(self, node, root_player):
#         """randomly simulate the game until the end and return the result """
#         board = deepcopy(node.board) # to be able to simulate without changing the actual board
#         current_player = node.player  
        
#         while not board.check_winner() and not board.is_full():
#             moves = board.get_available_moves() # returns list of tuples
#             if not moves:
#                 break
#             move = random.choice(moves) 
#             board.make_move(move[0], move[1], current_player)
#             current_player = 'O' if current_player == 'X' else 'X' # swap player
        
#         winner = board.check_winner()
#         if winner == root_player:
#             return 1
#         elif winner is None: # draw 
#             return 0.5 # better to just force the model to try to win than rewarding for draw
#         else:
#             return 0 # loss

#     def backpropagate(self, node, result): 
#         """ update all the parent nodes"""
#         while node is not None:
#             node.update(result)
#             node = node.parent

#     def best_move(self, root):
#         """ return the coordinates of the move to the child with the best win rate"""
#         best_move = None
#         best_win_rate = -1 

#         for child in root.children:
#             if child.visits > 0:
#                 win_rate = child.wins / child.visits # win rate seems better than just number of visits
#             else:
#                 win_rate = 0

#             if win_rate > best_win_rate:
#                 best_win_rate = win_rate
#                 best_move = child.move

#         return best_move

# class Node:
#     def __init__(self, board, player, move=None):
#         self.board = deepcopy(board)
#         self.player = player        # the player whose turn it is at this node
#         self.visits = 0
#         self.wins = 0
#         self.move = move            # the move that led to this node from the parent
#         self.children = []
#         self.untried_moves = board.get_available_moves()
#         self.parent = None

#     def is_fully_expanded(self):
#         """ returns true if all possible moves have been tried """
#         return len(self.untried_moves) == 0

#     def get_best_child(self, exploration_constant):
#         """ returns the child with the best UCB1 score """
#         best_score = -float('inf')
#         best_child = None
        
#         for child in self.children:
#             exploitation = child.wins / child.visits if child.visits > 0 else 0
#             if child.visits < 3:  # boost nodes with few visits
#                 exploration = exploration_constant * 2 * math.sqrt(2 * math.log(self.visits) / max(child.visits, 1))
#             else:
#                 exploration = exploration_constant * math.sqrt(2 * math.log(self.visits) / max(child.visits, 1))
#             score = exploitation + exploration # UCB1
#             if score > best_score:
#                 best_score = score
#                 best_child = child
#         return best_child

#     def add_child(self, move, board, player):
#         """ add a child node with the given move and inverse its player """
#         new_board = deepcopy(board)
#         new_board.make_move(move[0], move[1], player)
        
#         next_player = 'O' if player == 'X' else 'X' # the state after the move is made is the opposite player, thus the child gets the opposite symbol
        
#         child = Node(new_board, next_player, move)
#         child.parent = self
#         self.children.append(child)
        
#         self.untried_moves.remove(move) # remove the move from the list of untried moves        
#         return child

#     def update(self, result):
#         self.visits += 1
#         self.wins += result
