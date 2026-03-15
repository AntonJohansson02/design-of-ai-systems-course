class Board:
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
    
    def check_winner(self):
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

