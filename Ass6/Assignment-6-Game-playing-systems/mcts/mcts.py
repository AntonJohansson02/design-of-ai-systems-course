from copy import deepcopy
from typing import List
import math
import random

class TicTacToeBoard:
    def __init__(self, size=3):
        self.grid = [[' ' for _ in range(size)] for _ in range(size)] # empty grid
        self.size = size

    def get_board(self):
        """Return a copy of the board"""
        return self.grid.copy()
    
    def get_size(self):
        """Return the size of the board"""
        return self.size
    
    def is_full(self):
        """Check if the board is full"""
        for row in self.grid:
            for cell in row:
                if cell == ' ':
                    return False
        return True
    
    def make_move(self, row, col, symbol):
        """Make a move of the slot is empty"""
        if self.grid[row][col] == ' ':
            self.grid[row][col] = symbol
            return True
        return False
    
    def get_available_moves(self):
        """Return a list of available moves as tuples"""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == ' ':
                    moves.append((row, col))
        return moves
    
    def winner(self):
        """Check if there's a winner and return their symbol"""
        if self.size < 4:
            win_length = self.size
        else:
            win_length = 4  # if its larger than 4 then win by 4 anyways
        
        # check rows
        for row in range(self.size):
            for col in range(self.size - win_length + 1):
                if self.grid[row][col] != ' ':
                    if all(self.grid[row][col] == self.grid[row][col + i] for i in range(win_length)):
                        return self.grid[row][col]
        
        # check columns
        for col in range(self.size):
            for row in range(self.size - win_length + 1):
                if self.grid[row][col] != ' ':
                    if all(self.grid[row][col] == self.grid[row + i][col] for i in range(win_length)):
                        return self.grid[row][col]
        
        # check diagonals (top-left to bottom-right)
        for row in range(self.size - win_length + 1):
            for col in range(self.size - win_length + 1):
                if self.grid[row][col] != ' ':
                    if all(self.grid[row][col] == self.grid[row + i][col + i] for i in range(win_length)):
                        return self.grid[row][col]
        
        # check diagonals (top-right to bottom-left)
        for row in range(self.size - win_length + 1):
            for col in range(win_length - 1, self.size):
                if self.grid[row][col] != ' ':
                    if all(self.grid[row][col] == self.grid[row + i][col - i] for i in range(win_length)):
                        return self.grid[row][col]
        return None
        
    def display(self):
        """Print the board in the console"""
        h_line = "-" * (4 * self.size - 1) # horizontal separator line
        
        for row in range(self.size):
            row_display = " " + " | ".join(self.grid[row]) # row display with proper spacing
            print(row_display)
            
            if row < self.size - 1: # horizontal line between rows (except after the last row)
                print(h_line)
        print()
        print()

class Node:
    def __init__(self, state: TicTacToeBoard, last_player: str = None):
        self.board = deepcopy(state)
        self.vis: int = 0
        self.wins: int = 0
        self.last_player = last_player
        self.available_moves: list = state.get_available_moves()
    
    def available_moves_left(self) -> bool:
        empty_spaces = len(self.available_moves) > 0
        no_winner = self.board.winner() is None
        return empty_spaces and no_winner
    
    def __hash__(self):
        return hash(str(self.board.grid))
    
class Tree:
    def __init__(self, root: Node):
        self.root = root
        self.adjacency_matrix = {root: []}
        self.inverted_adjacency_matrix = {}

    def add_node(self, node: Node) -> None:
        self.adjacency_matrix[node] = []
       
    def add_edge(self, node_1: Node, node_2: Node) -> None:
        if node_1 not in self.adjacency_matrix:
            self.add_node(node_1)
        if node_2 not in self.adjacency_matrix:
            self.add_node(node_2)
        
        self.adjacency_matrix[node_1].append(node_2)
        self.inverted_adjacency_matrix[node_2] = node_1
    
    def has_parent(self, node: Node) -> bool:
        if node in self.inverted_adjacency_matrix.keys():
            return True
        else:
            return False

    def get_parent(self, node: Node) -> Node:
        if node in self.inverted_adjacency_matrix.keys():
            return self.inverted_adjacency_matrix[node]
        else:
            raise ValueError("This node does not have a parent")
    
    def get_children(self, node: Node) -> List[Node]:
        return self.adjacency_matrix[node]

    def get_best_child_simple(self, node: Node) -> Node:
        children = self.adjacency_matrix[node]
        best_child = max(children, key=lambda x: -x.vis)
        return best_child
    
    def get_best_child_ucb(self, node: Node, c = math.sqrt(2)) -> Node:
        children: List[Node] = self.adjacency_matrix[node]
        best_score = -float("inf")
        best_child: Node = None
        for child in children:
            exploitation = child.wins / max(child.vis, 1)  # Win ratio
            exploration = c * math.sqrt(math.log(node.vis) / max(child.vis, 1))            

            child_score = exploitation + exploration  # Upper confidence trees forumla from lecture
            
            # Update which node is the best child if it is better than all previous
            if child_score > best_score:
                best_score = child_score
                best_child = child

        return best_child

    

class MCTS:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
        self.simulations = 0
        self.wins = 0
        self.draws = 0
    
    def run_search(self, tic_tac_toe_board: TicTacToeBoard, start_player_symbol):
        """Returns the move that should be made"""
        
        # Check if game is already over
        if tic_tac_toe_board.winner() is not None or tic_tac_toe_board.is_full():
            print("Game is already over - no search needed")
            root_node = Node(tic_tac_toe_board, "X" if start_player_symbol == "O" else "O")
            return root_node
            
        last_player = "X" if start_player_symbol == "O" else "O"
        
        # Continue with normal search
        root_node = Node(tic_tac_toe_board, last_player)
        tree: Tree = Tree(root=root_node)

        for _ in range(self.num_iterations):
            node = self.select(tree, root_node)
            node = self.expand(tree, node)
            result = self.simulate(node)
            self.backwards(tree, node, result)
        
        return self.get_best_move(tree)
            
    
    def select(self, tree: Tree, root_node: Node) -> Node:
        node: Node = root_node
        while not node.available_moves_left() and len(tree.adjacency_matrix[node]) > 0:
            node = tree.get_best_child_ucb(node, 1.4)     
        return node

    def expand(self, tree: Tree, node: Node) -> Node:
        if not node.available_moves_left():
            return node
        
        if node.board.winner() is not None:
            return node
        
        if node.board.is_full():
            return node
        
        move = random.choice(node.available_moves)
        player_symbol = "X" if node.last_player == "O" else "O"

        node.available_moves.remove(move)

        new_board = deepcopy(node.board)
        new_board.make_move(move[0], move[1], player_symbol)

        new_node = Node(new_board, player_symbol)
        tree.add_node(new_node)
        tree.add_edge(node, new_node)

        return new_node
      
    def simulate(self, node: Node):
        copy_of_board = deepcopy(node.board)
        current_player_symbol = "X" if node.last_player == "O" else "O"
        
        # Check if the move just made resulted in a win already
        winning_symbol = copy_of_board.winner()
        if winning_symbol is not None:
            self.simulations += 1
            if winning_symbol == node.last_player:
                self.wins += 1
                return 1
            else:
                return -1
        
        # Continue with normal simulation but with improved rollout policy
        while not copy_of_board.is_full() and not copy_of_board.winner():
            available_moves = copy_of_board.get_available_moves()
            if not available_moves:
                break
                
            # 1. Check for winning moves for current player
            for move in available_moves:
                test_board = deepcopy(copy_of_board)
                test_board.make_move(move[0], move[1], current_player_symbol)
                if test_board.winner() == current_player_symbol:
                    # Found a winning move, use it
                    copy_of_board.make_move(move[0], move[1], current_player_symbol)
                    current_player_symbol = "X" if current_player_symbol == "O" else "O"
                    break
            else:  # No break occurred, no winning move found
                # 2. Check for blocking opponent's winning moves
                opponent = "X" if current_player_symbol == "O" else "O"
                for move in available_moves:
                    test_board = deepcopy(copy_of_board)
                    test_board.make_move(move[0], move[1], opponent)
                    if test_board.winner() == opponent:
                        # Found a move that blocks opponent's win, use it
                        copy_of_board.make_move(move[0], move[1], current_player_symbol)
                        current_player_symbol = "X" if current_player_symbol == "O" else "O"
                        break
                else:  # No break occurred, no blocking move found
                    # 3. Fallback to random move
                    move = random.choice(available_moves)
                    copy_of_board.make_move(move[0], move[1], current_player_symbol)
                    current_player_symbol = "X" if current_player_symbol == "O" else "O"

        # Rest of your existing code for evaluating the result
        winning_symbol = copy_of_board.winner()
        
        self.simulations += 1
        if winning_symbol == node.last_player:
            self.wins += 1
            return 1
        elif winning_symbol is None:
            self.draws += 1
            return 0
        else:
            return -1

    def backwards(self, tree: Tree, node: Node, value: float):
        """
        Update statistics for nodes using JavaScript-style win tracking
        """
        current_node = node
        winner_player = current_node.last_player if value == 1 else ("X" if current_node.last_player == "O" else "O")
        
        while True:
            current_node.vis += 1
            
            # Instead of negating values, directly evaluate if move helped player win
            if current_node != tree.root:  # Skip root node adjustments
                if current_node.last_player == winner_player:
                    current_node.wins += 1  # Player who made move won
                elif winner_player is not None:  # Not a draw
                    current_node.wins -= 1  # Player who made move lost
            
            if tree.has_parent(current_node):
                current_node = tree.get_parent(current_node)
            else:
                break

    def get_best_move(self, tree: Tree):
        children = tree.get_children(tree.root)
        
        if not children:
            print("No valid moves available - game might be over")
            return tree.root
        
        # # Check for immediate winning moves first (keep this optimization)
        # current_player = "X" if tree.root.last_player == "O" else "O"
        # for child in children:
        #     if child.board.winner() == current_player:
        #         print("\nImmediate winning move found!")
        #         child.board.display()
        #         return child
        
        # Use visit count instead of win rate for best move selection
        best_child = max(children, key=lambda x: x.vis)
        
        print("\nBest move selected:")
        win_rate = best_child.wins / max(best_child.vis, 1)
        print(f"Visits: {best_child.vis} (Win score: {best_child.wins})")
        best_child.board.display()
        
        print("\n\n")
        print(f"Total simulations: {self.simulations}")
        return best_child
    

# root_board = TicTacToeBoard()

# root_board.display()

# root_node = Node(root_board)

# tree = Tree(root_node)


# mcts = MCTS(1000)

# node = mcts.run_search(root_board, "X")
# node.board.display()

# node = mcts.run_search(node.board, "O")
# node.board.display()

# node = mcts.run_search(node.board, "X")
# node.board.display()

# node = mcts.run_search(node.board, "O")
# node.board.display()

# node = mcts.run_search(node.board, "X")
# node.board.display()

# node = mcts.run_search(node.board, "O")
# node.board.display()

# node = mcts.run_search(node.board, "X")
# node.board.display()





